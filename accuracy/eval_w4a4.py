#!/usr/bin/env python3
"""W4A4 accuracy of Qwen3.8-27B, simulated on one 24 GB GPU.

Every linear layer of the target's 64 decoder layers gets int4 per-channel weights and int4
per-token activations, quantized exactly as mi50emu's spec does it (spec_np.py: activations
asymmetric over [min(x,0), max(x,0)], 15 steps, round half to even; weights symmetric codes in
[-8, 7]). Weights are round-to-nearest with a per-channel clip search, or GPTQ (`-gptq`). The
embedding, lm_head, norms, DeltaNet recurrence, convolutions and attention stay BF16, as on the
MI50 plan.

Rotations, QuaRot-style:
  - the residual stream is rotated by R1, a randomized Hadamard (5120 = 20 x 256, Paley x
    Sylvester), folded into every weight that reads or writes it, with the zero-centered RMSNorm
    scales (1 + w) folded into the readers;
  - online randomized Hadamards on the inputs of o_proj and out_proj (6144 = 12 x 512) and
    down_proj (17408 = 68 x 256), folded into those weights. The attention output gate and the
    DeltaNet z-gate sit in front of o_proj / out_proj, so these can't be folded away; they are
    the three the cost model charges for.

Configs (all run in the same pass, so they see the same tokens):
  bf16       reference
  bf16-rot   rotations folded, nothing quantized: must match bf16 (checks the folding)
  w4a16      int4 weights, BF16 activations, no rotation
  w4a4       int4 weights and activations, no rotation
  w4a8-rot   int4 weights, int8 activations, rotated (the W4A8 fallback)
  w4a4-rot   int4 weights and activations, rotated (the method)
  w4a4-rot-gptq  the same with GPTQ weights: Hessians from 128 x seqlen tokens of wikitext-2
             train run through the already quantized layers, act order, no true-sequential

Layer by layer: one decoder layer on the GPU at a time, every config's hidden states for the eval
set in host memory. It reports perplexity, mean KL(reference || config) per token, and top-1
agreement: how often the config's argmax is the reference's, a proxy for how often a drafter
trained on the reference target still gets accepted.

The reference is either the bf16 config, or, with --ref, a llama.cpp model: the log-probs that
`llama-perplexity --kl-divergence-base FILE` saved (e.g. unsloth's UD-Q8_K_XL). Then the eval runs
on that file's tokens and chunks and scores the same positions (the second half of each chunk)
with the same KL and top-1 definitions as `llama-perplexity --kl-divergence`, so GGUF quants
scored by llama.cpp and these configs land on one scale.

    .venv/bin/python accuracy/eval_w4a4.py [--seqs 64] [--seqlen 2048] [--configs ...]
    .venv/bin/python accuracy/eval_w4a4.py --ref ud-q8_k_xl.kld [--configs ...]
    .venv/bin/python accuracy/eval_w4a4.py --selftest        # rotation algebra, random weights
"""

import argparse
import glob
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO = "Qwen/Qwen3.8-27B"
PREFIX = "model.language_model."
HIDDEN = 5120


# --------------------------------------------------------------------------- rotations
def paley(q):
    """Hadamard matrix of order q + 1 (Paley construction I; q prime, q = 3 mod 4)."""
    squares = {(i * i) % q for i in range(1, q)}
    chi = torch.tensor([0.0] + [1.0 if i in squares else -1.0 for i in range(1, q)])
    S = torch.zeros(q + 1, q + 1)
    S[0, 1:], S[1:, 0] = 1, -1
    S[1:, 1:] = torch.stack([torch.roll(chi, i) for i in range(q)])  # chi[(j - i) mod q]
    H = torch.eye(q + 1) + S
    assert torch.equal(H @ H.T, (q + 1) * torch.eye(q + 1))
    return H


def sylvester(m):
    H = torch.ones(1, 1)
    while H.shape[0] < m:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


FACTORS = {5120: (20, 19, 256), 6144: (12, 11, 512), 17408: (68, 67, 256)}  # n: (paley order, q, 2^k)


class Rotation:
    """x -> x @ H for row vectors, H = diag(signs) (H_p kron H_m) / sqrt(n): orthogonal."""

    def __init__(self, n, seed, device):
        p, q, m = FACTORS[n]
        self.p, self.m, self.n = p, m, n
        self.A, self.B = paley(q).to(device), sylvester(m).to(device)
        g = torch.Generator().manual_seed(seed)
        self.signs = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(device)

    def __call__(self, x):
        shape = x.shape
        X = (x.float() * self.signs).reshape(*shape[:-1], self.p, self.m)
        Y = torch.einsum("...ij,ia,jb->...ab", X, self.A, self.B)
        return Y.reshape(shape) / math.sqrt(self.n)

    def inverse(self, y):
        """y -> y @ H^T."""
        shape = y.shape
        Y = y.float().reshape(*shape[:-1], self.p, self.m)
        X = torch.einsum("...ab,ia,jb->...ij", Y, self.A, self.B)
        return X.reshape(shape) / math.sqrt(self.n) * self.signs


# --------------------------------------------------------------------------- quantizers
def fake_quant_act(x, bits, clip=1.0, group=0):
    """mi50emu spec_np.py activation quantizer, per token (or per token per `group` inputs), returned
    dequantized (fp32); the range is scaled by clip first, as RQ_ACT_CLIP does in our ggml."""
    if group:
        shape = x.shape
        return fake_quant_act(x.reshape(*shape[:-1], -1, group), bits, clip).reshape(shape)
    levels = 2**bits - 1
    x = x.float()
    mn = x.amin(-1, keepdim=True).clamp(max=0) * clip
    mx = x.amax(-1, keepdim=True).clamp(min=0) * clip
    both0 = (mn == 0) & (mx == 0)
    mn, mx = torch.where(both0, -1.0, mn), torch.where(both0, 1.0, mx)
    scale = (mx - mn) / levels
    inv = 1.0 / scale
    z = torch.clamp(torch.round(-mn * inv), 0, levels)
    q = torch.clamp(torch.round(x * inv) + z, 0, levels)
    return (q - z) * scale


def int4_scale(W, ratios=torch.linspace(1.0, 0.7, 13).tolist()):
    """Per output channel scale for symmetric codes in [-8, 7], clip ratio searched for least
    squared rounding error."""
    W = W.float()
    amax = W.abs().amax(1, keepdim=True).clamp_min(1e-12)
    best_s, best_err = None, None
    for r in ratios:
        s = amax * r / 7
        err = (torch.clamp(torch.round(W / s), -8, 7) * s - W).pow(2).sum(1, keepdim=True)
        if best_s is None:
            best_s, best_err = s, err
        else:
            better = err < best_err
            best_s, best_err = torch.where(better, s, best_s), torch.where(better, err, best_err)
    return best_s


def rtn_int4(W):
    s = int4_scale(W)
    return torch.clamp(torch.round(W.float() / s), -8, 7) * s


def gptq_int4(W, H, block=128, damp=0.01, group=0):
    """GPTQ onto the int4 grid; H = sum of x x^T over calibration inputs. group=0: one scale per row
    (the W4A4 type), act order. group=g: one scale per row per g inputs, set when GPTQ reaches the
    group, in natural order."""
    W = W.float().clone()
    H = H.clone()
    n = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    s = int4_scale(W)
    perm = torch.argsort(torch.diag(H), descending=True) if not group else torch.arange(n, device=W.device)
    W, H = W[:, perm], H[perm][:, perm]
    H += damp * torch.mean(torch.diag(H)) * torch.eye(n, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1, Hi = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Err = torch.zeros_like(W1)
        for i in range(i2 - i1):
            if group and (i1 + i) % group == 0:  # scale from the group's current (updated) weights
                s = int4_scale(torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :group])
            w = W1[:, i]
            q = torch.clamp(torch.round(w / s[:, 0]), -8, 7) * s[:, 0]
            Q[:, i1 + i] = q
            e = (w - q) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None, :]
            Err[:, i] = e
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, torch.argsort(perm)]


class QLinear(nn.Module):
    """A linear layer whose input may get an online rotation and fake activation quantization."""

    def __init__(self, weight, rot=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.rot, self.abits, self.clip, self.agroup = rot, None, 1.0, 0
        self.H = None  # set to zeros to collect GPTQ's Hessian of the (rotated) inputs

    def forward(self, x):
        if self.rot is not None or self.abits or self.H is not None:
            dt = x.dtype
            x = x.float()
            if self.rot is not None:
                x = self.rot(x)
            if self.H is not None:
                xf = x.reshape(-1, x.shape[-1])
                self.H.addmm_(xf.T, xf)
            if self.abits:
                x = fake_quant_act(x, self.abits, self.clip, self.agroup)
            x = x.to(dt)
        return F.linear(x, self.weight)


# --------------------------------------------------------------------------- the model, by layer
READ_IN = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b",
           "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]  # read input_layernorm's output
READ_POST = ["mlp.gate_proj", "mlp.up_proj"]  # read post_attention_layernorm's output
WRITE = {"linear_attn.out_proj": 6144, "self_attn.o_proj": 6144, "mlp.down_proj": 17408}  # write the residual
LINEARS = READ_IN + READ_POST + list(WRITE)


@dataclass(frozen=True)
class Config:
    name: str
    rot: bool
    wq: bool
    abits: int | None
    gptq: bool = False


CONFIGS = {c.name: c for c in [
    Config("bf16", False, False, None),
    Config("bf16-rot", True, False, None),
    Config("w4a16", False, True, None),
    Config("w4a4", False, True, 4),
    Config("w4a8-rot", True, True, 8),
    Config("w4a4-rot", True, True, 4),
    Config("w4a4-rot-gptq", True, True, 4, gptq=True),
]}


class Checkpoint:
    def __init__(self, path):
        from safetensors import safe_open
        self.dir = Path(path)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.where, self.open, self.safe_open = index, {}, safe_open

    def get(self, name):
        f = self.where[name]
        if f not in self.open:
            self.open[f] = self.safe_open(str(self.dir / f), framework="pt")
        return self.open[f].get_tensor(name)

    def layer(self, i):
        p = f"{PREFIX}layers.{i}."
        return {k[len(p):]: self.get(k) for k in self.where if k.startswith(p)}


def transform(sd, rot, wq, R1, Rin):
    """A layer's state dict for one weight variant: rotations folded, then weights quantized."""
    sd = {k: v.cuda().float() for k, v in sd.items()}
    if rot:
        s_in = 1 + sd["input_layernorm.weight"]
        s_post = 1 + sd["post_attention_layernorm.weight"]
        for names, s in ((READ_IN, s_in), (READ_POST, s_post)):
            for n in names:
                if n + ".weight" in sd:
                    sd[n + ".weight"] = R1(sd[n + ".weight"] * s)  # W diag(s) R1
        sd["input_layernorm.weight"].zero_()
        sd["post_attention_layernorm.weight"].zero_()
        for n, dim in WRITE.items():
            if n + ".weight" in sd:
                W = Rin[dim](sd[n + ".weight"])  # W H: the input gets x H online
                sd[n + ".weight"] = R1(W.T).T  # R1^T W H
    if wq:
        for n in LINEARS:
            if n + ".weight" in sd:
                sd[n + ".weight"] = rtn_int4(sd[n + ".weight"])
    return sd


def build_layer(M, tcfg, i, sd, rot, Rin):
    with torch.device("meta"):  # no random init: the weights are assigned below
        layer = M.Qwen3_5DecoderLayer(tcfg, i)
    keep32 = ("linear_attn.A_log", "linear_attn.dt_bias")
    layer.load_state_dict({k: v.cuda().to(torch.float32 if k in keep32 else torch.bfloat16) for k, v in sd.items()},
                          strict=True, assign=True)
    for n in LINEARS:
        parent_name, attr = n.rsplit(".", 1)
        parent = layer.get_submodule(parent_name) if hasattr(layer, parent_name.split(".")[0]) else None
        if parent is None or not hasattr(parent, attr):
            continue
        lin = getattr(parent, attr)
        setattr(parent, attr, QLinear(lin.weight.data, Rin[WRITE[n]] if rot and n in WRITE else None))
    return layer.eval()


def set_abits(layer, abits):
    for m in layer.modules():
        if isinstance(m, QLinear):
            m.abits = abits


# --------------------------------------------------------------------------- data
def wikitext_ids(tok, split):
    import pyarrow.parquet as pq
    pattern = f"wikitext-2-raw-v1/{split}-*.parquet"
    files = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/{pattern}"))
    if not files:
        from huggingface_hub import snapshot_download
        d = snapshot_download("wikitext", repo_type="dataset", allow_patterns=pattern)
        files = glob.glob(f"{d}/{pattern}")
    text = "\n\n".join(pq.read_table(files[0]).column("text").to_pylist())
    return tok(text, return_tensors="pt").input_ids[0]


def eval_tokens(tok, seqs, seqlen):
    ids = wikitext_ids(tok, "test")
    n = min(seqs, ids.numel() // seqlen)
    return ids[: n * seqlen].view(n, seqlen), ids.numel()


def calib_tokens(tok, seqs, seqlen, seed=0):
    """Random windows of wikitext-2 train, the usual GPTQ / QuaRot calibration set."""
    ids = wikitext_ids(tok, "train")
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - seqlen, (seqs,), generator=g)
    return torch.stack([ids[s:s + seqlen] for s in starts.tolist()])


class LlamaRef:
    """Log-probs saved by `llama-perplexity --kl-divergence-base`: "_logits_", int32 n_ctx, n_vocab,
    n_chunk, the n_chunk * n_ctx tokens, then for each chunk n_ctx - 1 - n_ctx/2 records (the
    positions from n_ctx/2 on) of [f32 scale, f32 min_log_prob, n_vocab x u16 q], with
    log_prob = scale * q + min_log_prob (tools/perplexity/perplexity.cpp)."""

    def __init__(self, path):
        import numpy as np
        self.np = np
        with open(path, "rb") as f:
            assert f.read(8) == b"_logits_", f"{path} is not a llama.cpp log-prob file"
            self.n_ctx, self.n_vocab, self.n_chunk = np.frombuffer(f.read(12), dtype=np.int32).tolist()
        self.tokens = np.memmap(path, dtype=np.int32, mode="r", offset=20, shape=(self.n_chunk, self.n_ctx))
        self.first = self.n_ctx // 2
        self.per_chunk = self.n_ctx - 1 - self.first
        nv = 2 * ((self.n_vocab + 1) // 2) + 4
        self.rec = np.memmap(path, dtype=np.uint16, mode="r", offset=20 + self.n_chunk * self.n_ctx * 4,
                             shape=(self.n_chunk, self.per_chunk, nv))

    def log_probs(self, chunk, t0, t1, device):
        """[t1 - t0, n_vocab] reference log-probs for record positions t0..t1 of a chunk."""
        raw = self.np.ascontiguousarray(self.rec[chunk, t0:t1])
        sm = torch.from_numpy(raw[:, :4].copy().view(self.np.float32)).to(device)  # scale, min_log_prob
        q = torch.from_numpy(raw[:, 4:4 + self.n_vocab].view(self.np.int16)).to(device).int() & 0xFFFF
        return q.float() * sm[:, :1] + sm[:, 1:2]


# --------------------------------------------------------------------------- main
@torch.no_grad()
def run(a):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    path = snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    cfg = json.loads((Path(path) / "config.json").read_text())
    tcfg = Qwen3_5TextConfig(**cfg["text_config"])
    tcfg._attn_implementation = "sdpa"
    layers = a.layers or tcfg.num_hidden_layers
    tok = AutoTokenizer.from_pretrained(path)
    ref = LlamaRef(a.ref) if a.ref else None
    if ref:
        import numpy as np
        ids = torch.from_numpy(np.array(ref.tokens[: a.seqs])).long()
        ids_in = ids.clone()
        if a.ref_bos is not None:  # llama-perplexity puts BOS over each chunk's first token
            ids_in[:, 0] = a.ref_bos
        what = f"{a.ref}: {ref.n_chunk} chunks of {ref.n_ctx}"
    else:
        ids, total = eval_tokens(tok, a.seqs, a.seqlen)
        ids_in, what = ids, f"wikitext-2 test: {total} tokens"
    nseq, L, B = ids.shape[0], ids.shape[1], a.batch
    names = a.configs or (["w4a8-rot", "w4a4-rot", "w4a4-rot-gptq"] if ref else list(CONFIGS))
    configs = [CONFIGS[c] for c in names]
    assert ref or configs[0].name == "bf16", "without --ref, bf16 must come first: it is the reference"
    print(f"{what}, evaluating {nseq} x {L} = {nseq * L}; layers {layers}; "
          f"configs {', '.join(c.name for c in configs)}", flush=True)

    ck = Checkpoint(path)
    dev = "cuda"
    R1 = Rotation(HIDDEN, 1, dev)
    Rin = {6144: Rotation(6144, 2, dev), 17408: Rotation(17408, 3, dev)}

    # hidden states per config, in host memory (rotated configs keep the rotated residual)
    embed = ck.get(PREFIX + "embed_tokens.weight").to(dev)
    states = {}
    for c in configs:
        st = torch.empty(nseq, L, HIDDEN, dtype=torch.bfloat16, pin_memory=True)
        for s in range(0, nseq, B):
            h = embed[ids_in[s:s + B].to(dev)]
            st[s:s + B] = (R1(h).to(torch.bfloat16) if c.rot else h).cpu()
        states[c.name] = st
    del embed

    # GPTQ configs carry a calibration stream through their own quantized layers
    calib = {}
    if any(c.gptq for c in configs):
        cids = calib_tokens(tok, a.calib_seqs, L)
        embed = ck.get(PREFIX + "embed_tokens.weight").to(dev)
        for c in (c for c in configs if c.gptq):
            st = torch.empty(cids.shape[0], L, HIDDEN, dtype=torch.bfloat16, pin_memory=True)
            for s in range(0, cids.shape[0], B):
                h = embed[cids[s:s + B].to(dev)]
                st[s:s + B] = (R1(h).to(torch.bfloat16) if c.rot else h).cpu()
            calib[c.name] = st
        del embed

    rotary = M.Qwen3_5TextRotaryEmbedding(tcfg).to(dev)
    pos = torch.arange(L, device=dev).view(1, 1, -1).expand(3, B, -1)
    pe = rotary(torch.empty(1, device=dev, dtype=torch.bfloat16), pos)

    def forward_all(layer, st):
        for s in range(0, st.shape[0], B):
            h = st[s:s + B].to(dev, non_blocking=True)
            n = h.shape[0]
            st[s:s + B].copy_(layer(h, position_embeddings=(pe[0][:n], pe[1][:n])), non_blocking=True)

    variants = sorted({(c.rot, c.wq) for c in configs if not c.gptq})
    log = []
    for i in range(layers):
        t0 = time.time()
        sd_cpu = ck.layer(i)
        for rot, wq in variants:
            layer = build_layer(M, tcfg, i, transform(sd_cpu, rot, wq, R1, Rin), rot, Rin)
            for c in (c for c in configs if (c.rot, c.wq) == (rot, wq) and not c.gptq):
                set_abits(layer, c.abits)
                forward_all(layer, states[c.name])
            del layer
            torch.cuda.empty_cache()
        for c in (c for c in configs if c.gptq):
            # Hessians from this config's calibration stream through the layer's unquantized
            # (rotated) weights, then GPTQ, then both streams through the quantized layer
            layer = build_layer(M, tcfg, i, transform(sd_cpu, c.rot, False, R1, Rin), c.rot, Rin)
            qlins = [m for m in layer.modules() if isinstance(m, QLinear)]
            for m in qlins:
                m.H = torch.zeros(m.weight.shape[1], m.weight.shape[1], device=dev)
            forward_all(layer, calib[c.name].clone())
            for m in qlins:
                m.weight.data = gptq_int4(m.weight.data, m.H).to(torch.bfloat16)
                m.H = None
            set_abits(layer, c.abits)
            forward_all(layer, calib[c.name])
            forward_all(layer, states[c.name])
            del layer, qlins
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # residual-stream error of each config against bf16, first batch (when bf16 runs)
        errs = {}
        if "bf16" in states:
            h_ref = states["bf16"][:B].to(dev).float()
            for c in (c for c in configs if c.name != "bf16"):
                h = states[c.name][:B].to(dev).float()
                if c.rot:
                    h = R1.inverse(h)
                errs[c.name] = ((h - h_ref).norm() / h_ref.norm()).item()
        log.append({"layer": i, "s": time.time() - t0, "rel_err": errs})
        print(f"layer {i:2d} ({tcfg.layer_types[i][:6]}) {time.time() - t0:5.1f} s  "
              + "  ".join(f"{k} {v:.4f}" for k, v in errs.items()), flush=True)

    # final norm + lm_head (BF16, not quantized), metrics against bf16
    norm = M.Qwen3_5RMSNorm(HIDDEN, eps=tcfg.rms_norm_eps).to(dev)
    norm.weight.copy_(ck.get(PREFIX + "norm.weight").to(dev))
    lm = ck.get("lm_head.weight").to(dev)
    acc = {c.name: {"nll": 0.0, "kl": 0.0, "top1": 0, "n": 0} for c in configs}
    chunk = a.chunk
    if ref:
        acc["ref"] = {"nll": 0.0, "kl": 0.0, "top1": 0, "n": 0}
        for s in range(nseq):
            hs = {}
            for c in configs:
                h = states[c.name][s].to(dev)
                hs[c.name] = R1.inverse(h).to(torch.bfloat16) if c.rot else h
            for t in range(0, ref.per_chunk, chunk):  # record t is position first + t
                e = min(t + chunk, ref.per_chunk)
                lp_ref = ref.log_probs(s, t, e, dev)
                tgt = ids[s, ref.first + t + 1:ref.first + e + 1].to(dev)
                top_ref = lp_ref.argmax(-1)
                p_ref = torch.where(lp_ref > -16, lp_ref.exp(), 0.0)  # llama.cpp skips the rest
                acc["ref"]["nll"] -= lp_ref.gather(-1, tgt[:, None]).sum().item()
                acc["ref"]["n"] += e - t
                for c in configs:
                    pos = slice(ref.first + t, ref.first + e)
                    lp = torch.log_softmax(F.linear(norm(hs[c.name][pos]), lm).float(), -1)
                    r = acc[c.name]
                    r["nll"] -= lp.gather(-1, tgt[:, None]).sum().item()
                    r["kl"] += (p_ref * (lp_ref - lp)).sum().item()
                    r["top1"] += (lp.argmax(-1) == top_ref).sum().item()
                    r["n"] += e - t
        configs = [Config("ref", False, False, None)] + configs
    for s in range(nseq if not ref else 0):
        tgt = ids[s, 1:].to(dev)
        hs = {}
        for c in configs:
            h = states[c.name][s].to(dev)
            hs[c.name] = (R1.inverse(h).to(torch.bfloat16) if c.rot else h)
        for t in range(0, L - 1, chunk):
            e = min(t + chunk, L - 1)
            lp_ref = None
            for c in configs:
                logits = F.linear(norm(hs[c.name][t:e]), lm).float()
                lp = torch.log_softmax(logits, -1)
                r = acc[c.name]
                r["nll"] -= lp.gather(-1, tgt[t:e, None]).sum().item()
                r["n"] += e - t
                if lp_ref is None:
                    lp_ref, top_ref = lp, lp.argmax(-1)
                else:
                    r["kl"] += (lp_ref.exp() * (lp_ref - lp)).sum().item()
                    r["top1"] += (lp.argmax(-1) == top_ref).sum().item()
    results = {}
    print(f"\n{'config':<10} {'ppl':>8} {'KL/token':>9} {'top-1 agree':>12}")
    for c in configs:
        r = acc[c.name]
        res = {"ppl": math.exp(r["nll"] / r["n"]), "kl": r["kl"] / r["n"], "top1": r["top1"] / r["n"],
               "tokens": r["n"]}
        results[c.name] = res
        agree = "    (ref)" if c.name in ("bf16", "ref") else f"{100 * res['top1']:11.2f}%"
        print(f"{c.name:<10} {res['ppl']:8.3f} {res['kl']:9.4f} {agree}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(a), "results": results, "layers": log}, indent=1))
    print(f"-> {out}")


@dataclass(frozen=True)
class Recipe:
    name: str
    group: int        # weight scale group (0 = per row, the W4A4 type)
    abits_read: int   # activation bits for linears reading the residual stream (attn/DeltaNet inputs, gate/up)
    abits_write: int  # activation bits for attn_output / ssm_out / ffn_down (after their online Hadamard)
    clip4: float = 0.8
    clip8: float = 1.0
    agroup: int = 0   # activation scale group (0 = per token)


# all W4A4 on dot8; groups cost one convert + one fma per group per output per token (g128: 1/8 of the
# dot8s, g256: 1/16), the same whether the weights, the activations or both are grouped
RECIPES = {r.name: r for r in [
    Recipe("w4a4", 0, 4, 4),
    Recipe("wg256", 256, 4, 4),
    Recipe("ag256", 0, 4, 4, clip4=0.9, agroup=256),
    Recipe("g256", 256, 4, 4, clip4=0.9, agroup=256),
    Recipe("g128", 128, 4, 4, clip4=0.9, agroup=128),
]}


@torch.no_grad()
def run_recipes(a):
    """Several GPTQ recipes in one pass against a llama.cpp reference. Rotated as in run(); one
    calibration stream through the rotated unquantized model gives every layer's Hessians (GPTQ's
    non-sequential mode), one GPTQ solve per weight grid serves every recipe on that grid."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    import numpy as np

    path = snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    tcfg = Qwen3_5TextConfig(**json.loads((Path(path) / "config.json").read_text())["text_config"])
    tcfg._attn_implementation = "sdpa"
    layers = a.layers or tcfg.num_hidden_layers
    tok = AutoTokenizer.from_pretrained(path)
    ref = LlamaRef(a.ref)
    ids = torch.from_numpy(np.array(ref.tokens[: a.seqs])).long()
    nseq, L, B = ids.shape[0], ids.shape[1], a.batch
    recipes = [RECIPES[r] for r in (a.recipes or list(RECIPES))]
    groups = sorted({r.group for r in recipes})
    cids = calib_tokens(tok, a.calib_seqs, L)
    print(f"{a.ref}: {nseq} x {L} eval, {cids.shape[0]} x {L} calibration; recipes "
          f"{', '.join(r.name for r in recipes)}", flush=True)

    ck = Checkpoint(path)
    dev = "cuda"
    R1 = Rotation(HIDDEN, 1, dev)
    Rin = {6144: Rotation(6144, 2, dev), 17408: Rotation(17408, 3, dev)}
    embed = ck.get(PREFIX + "embed_tokens.weight").to(dev)

    def embed_rot(t):
        st = torch.empty(t.shape[0], L, HIDDEN, dtype=torch.bfloat16, pin_memory=True)
        for s in range(0, t.shape[0], B):
            st[s:s + B] = R1(embed[t[s:s + B].to(dev)]).to(torch.bfloat16).cpu()
        return st

    calib = embed_rot(cids)
    states = {r.name: embed_rot(ids) for r in recipes}
    del embed
    rotary = M.Qwen3_5TextRotaryEmbedding(tcfg).to(dev)
    pe = rotary(torch.empty(1, device=dev, dtype=torch.bfloat16), torch.arange(L, device=dev).view(1, 1, -1).expand(3, B, -1))

    def forward_all(layer, st):
        for s in range(0, st.shape[0], B):
            h = st[s:s + B].to(dev, non_blocking=True)
            n = h.shape[0]
            st[s:s + B].copy_(layer(h, position_embeddings=(pe[0][:n], pe[1][:n])), non_blocking=True)

    def qlinears(layer):
        return {n: m for n, m in layer.named_modules() if isinstance(m, QLinear)}

    for i in range(layers):
        t0 = time.time()
        sd = transform(ck.layer(i), True, False, R1, Rin)
        fp = build_layer(M, tcfg, i, sd, True, Rin)
        for m in qlinears(fp).values():
            m.H = torch.zeros(m.weight.shape[1], m.weight.shape[1], device=dev)
        forward_all(fp, calib)  # Hessians, and the next layer's calibration inputs
        quant = {g: {n: gptq_int4(m.weight.data, m.H, group=g).to(torch.bfloat16) for n, m in qlinears(fp).items()}
                 for g in groups}
        del fp
        for r in recipes:
            layer = build_layer(M, tcfg, i, sd, True, Rin)
            for n, m in qlinears(layer).items():
                m.weight.data = quant[r.group][n]
                write = any(n.endswith(w) for w in WRITE)
                m.abits = r.abits_write if write else r.abits_read
                m.clip = r.clip4 if m.abits == 4 else r.clip8
                m.agroup = r.agroup
            forward_all(layer, states[r.name])
            del layer
        del quant, sd
        torch.cuda.empty_cache()
        print(f"layer {i:2d} ({tcfg.layer_types[i][:6]}) {time.time() - t0:5.1f} s", flush=True)

    norm = M.Qwen3_5RMSNorm(HIDDEN, eps=tcfg.rms_norm_eps).to(dev)
    norm.weight.copy_(ck.get(PREFIX + "norm.weight").to(dev))
    lm = ck.get("lm_head.weight").to(dev)
    acc = {r.name: [0.0, 0.0, 0, 0] for r in recipes}  # nll, kl, top1, n
    for s in range(nseq):
        hs = {r.name: R1.inverse(states[r.name][s].to(dev)).to(torch.bfloat16) for r in recipes}
        for t in range(0, ref.per_chunk, a.chunk):
            e = min(t + a.chunk, ref.per_chunk)
            lp_ref = ref.log_probs(s, t, e, dev)
            tgt = ids[s, ref.first + t + 1:ref.first + e + 1].to(dev)
            top_ref = lp_ref.argmax(-1)
            p_ref = torch.where(lp_ref > -16, lp_ref.exp(), 0.0)
            for r in recipes:
                lp = torch.log_softmax(F.linear(norm(hs[r.name][ref.first + t:ref.first + e]), lm).float(), -1)
                c = acc[r.name]
                c[0] -= lp.gather(-1, tgt[:, None]).sum().item()
                c[1] += (p_ref * (lp_ref - lp)).sum().item()
                c[2] += (lp.argmax(-1) == top_ref).sum().item()
                c[3] += e - t
    print(f"\n{'recipe':<12} {'ppl':>8} {'KL/token':>9} {'top-1 agree':>12}   (vs {a.ref}, first {nseq} chunks)")
    results = {}
    for r in recipes:
        nll, kl, top1, n = acc[r.name]
        results[r.name] = {"ppl": math.exp(nll / n), "kl": kl / n, "top1": top1 / n, "tokens": n, "recipe": vars(r)}
        print(f"{r.name:<12} {math.exp(nll / n):8.3f} {kl / n:9.4f} {100 * top1 / n:11.2f}%")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(a), "results": results}, indent=1))
    print(f"-> {out}")


def learn_r1(a):
    """A residual rotation fitted to the model's own activations: R1 (5120 x 5120 orthogonal, row
    convention y = x R1) minimizing the relative per-token W4A4 quantization error (spec quantizer,
    clip a.clip, straight-through rounding) of the normed residual vectors every reader quantizes:
    h / rms(h) at both norms of every layer. Starts from the randomized Hadamard used so far.
    Written as raw f32 for `ruquant convert --r1-file`."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    path = snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    tcfg = Qwen3_5TextConfig(**json.loads((Path(path) / "config.json").read_text())["text_config"])
    tcfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(path)
    ck = Checkpoint(path)
    dev, L, B = "cuda", a.seqlen, a.batch
    cids = calib_tokens(tok, a.calib_seqs, L, seed=1)
    keep = a.r1_tokens  # tokens kept per sequence per capture point
    g = torch.Generator().manual_seed(0)
    picks = torch.stack([torch.randperm(L, generator=g)[:keep] for _ in range(cids.shape[0])])
    rotary = M.Qwen3_5TextRotaryEmbedding(tcfg).to(dev)
    pe = rotary(torch.empty(1, device=dev, dtype=torch.bfloat16), torch.arange(L, device=dev).view(1, 1, -1).expand(3, B, -1))
    embed = ck.get(PREFIX + "embed_tokens.weight").to(dev)
    st = torch.stack([embed[cids[s].to(dev)].cpu() for s in range(cids.shape[0])]).to(torch.bfloat16)
    del embed
    caps = []  # normed residual samples, bf16 on the GPU
    t0 = time.time()
    grad_was = torch.is_grad_enabled()
    torch.set_grad_enabled(False)  # capture only; autograd is for the fit below
    for i in range(tcfg.num_hidden_layers):
        layer = build_layer(M, tcfg, i, transform(ck.layer(i), False, False, None, None), False, None)
        grabbed = []
        hooks = [m.register_forward_pre_hook(lambda mod, args: grabbed.append(args[0]))
                 for m in (layer.input_layernorm, layer.post_attention_layernorm)]
        for s in range(0, st.shape[0], B):
            h = st[s:s + B].to(dev)
            n = h.shape[0]
            grabbed.clear()
            st[s:s + B] = layer(h, position_embeddings=(pe[0][:n], pe[1][:n])).cpu()
            for x in grabbed:  # [n, L, 5120] residual entering a norm
                idx = picks[s:s + n].to(dev)
                xs = torch.gather(x, 1, idx[:, :, None].expand(-1, -1, HIDDEN)).reshape(-1, HIDDEN).float()
                caps.append((xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + tcfg.rms_norm_eps)).to(torch.bfloat16))
        for hk in hooks:
            hk.remove()
        del layer
    X = torch.cat(caps)
    del caps
    torch.set_grad_enabled(grad_was)
    torch.cuda.empty_cache()
    print(f"captured {X.shape[0]} normed residual vectors in {time.time() - t0:.0f} s", flush=True)

    def qerr(Y, clip):  # relative per-token quantization error, straight-through rounding
        mn = Y.amin(-1, keepdim=True).clamp(max=0) * clip
        mx = Y.amax(-1, keepdim=True).clamp(min=0) * clip
        scale = (mx - mn).clamp_min(1e-8) / 15
        z = torch.round(-mn / scale).detach().clamp(0, 15)
        u = Y / scale
        q = torch.clamp(u + (torch.round(u) - u).detach() + z, 0, 15)
        return (((q - z) * scale - Y).pow(2).sum(-1) / Y.pow(2).sum(-1).clamp_min(1e-12)).mean()

    perm = torch.randperm(X.shape[0], generator=g)
    hold, train = X[perm[:16384]], X[perm[16384:]]
    eye = torch.eye(HIDDEN, device=dev)
    R0 = Rotation(HIDDEN, 1, dev)(eye)  # the randomized Hadamard, as a dense matrix
    start = R0
    if a.r1_init:  # continue an earlier fit
        import numpy as np
        start = torch.from_numpy(np.fromfile(a.r1_init, dtype=np.float32).reshape(HIDDEN, HIDDEN)).to(dev)
    P = start.clone().requires_grad_(True)
    opt = torch.optim.Adam([P], lr=a.r1_lr)

    def ortho(P):
        Q, Rr = torch.linalg.qr(P)
        return Q * torch.sign(torch.diagonal(Rr))[None, :]

    with torch.no_grad():
        base = qerr(hold.float() @ R0, a.clip).item()
    print(f"held-out relative error: random Hadamard {base:.5f}", flush=True)
    for step in range(a.r1_steps):
        R = ortho(P)
        batch = train[torch.randint(0, train.shape[0], (a.r1_batch,), device=dev)].float()
        loss = qerr(batch @ R, a.clip)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == a.r1_steps - 1:
            with torch.no_grad():
                h = qerr(hold.float() @ ortho(P), a.clip).item()
            print(f"  step {step:4d}  train {loss.item():.5f}  held-out {h:.5f}  ({h / base:.3f}x)", flush=True)
    with torch.no_grad():
        R = ortho(P).contiguous()
        print(f"orthogonality error {(R @ R.T - eye).abs().max().item():.2e}")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        R.cpu().numpy().astype("float32").tofile(a.out)
    print(f"-> {a.out} ({HIDDEN} x {HIDDEN} f32, y = x R1)")


@torch.no_grad()
def selftest():
    """Rotation algebra on random weights: rotated layers must reproduce the unrotated ones."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    dev = "cuda"
    for n in FACTORS:
        R = Rotation(n, 7, dev)
        x = torch.randn(3, n, device=dev)
        assert torch.allclose(R.inverse(R(x)), x, atol=1e-4), n
        assert torch.allclose(R(x).norm(dim=-1), x.norm(dim=-1), rtol=1e-4), n
    tcfg = Qwen3_5TextConfig(hidden_size=HIDDEN, intermediate_size=17408, num_hidden_layers=4, num_attention_heads=24,
                             num_key_value_heads=4, head_dim=256, linear_num_key_heads=16, linear_num_value_heads=48,
                             linear_key_head_dim=128, linear_value_head_dim=128, attn_output_gate=True,
                             partial_rotary_factor=0.25, full_attention_interval=4,
                             rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                              "mrope_section": [11, 11, 10], "mrope_interleaved": True})
    tcfg._attn_implementation = "sdpa"
    R1 = Rotation(HIDDEN, 1, dev)
    Rin = {6144: Rotation(6144, 2, dev), 17408: Rotation(17408, 3, dev)}
    rotary = M.Qwen3_5TextRotaryEmbedding(tcfg).to(dev)
    L = 256
    pe = rotary(torch.empty(1, device=dev, dtype=torch.bfloat16), torch.arange(L, device=dev).view(1, 1, -1).expand(3, 2, -1))
    for i in (0, 3):  # one DeltaNet layer, one attention layer
        torch.manual_seed(i)
        ref = M.Qwen3_5DecoderLayer(tcfg, i)
        for p in ref.parameters():
            p.data.normal_(0, 0.02)
        sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
        if "linear_attn.A_log" in sd:
            sd["linear_attn.A_log"] = torch.log(torch.rand_like(sd["linear_attn.A_log"]) * 15 + 0.5)
        h = torch.randn(2, L, HIDDEN, device=dev)
        outs = {}
        for rot in (False, True):
            layer = build_layer(M, tcfg, i, transform(sd, rot, False, R1, Rin), rot, Rin)
            x = (R1(h) if rot else h).to(torch.bfloat16)
            y = layer(x, position_embeddings=pe).float()
            outs[rot] = R1.inverse(y) if rot else y
        err = ((outs[True] - outs[False]).norm() / outs[False].norm()).item()
        print(f"selftest layer {i} ({tcfg.layer_types[i]}): rotated vs plain rel. error {err:.2e}")
        assert err < 2e-2, err
    print("selftest OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seqs", type=int, default=64, help="eval sequences from wikitext-2 test")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch", type=int, default=4, help="sequences per forward")
    p.add_argument("--chunk", type=int, default=512, help="tokens per lm_head chunk")
    p.add_argument("--layers", type=int, default=0, help="only the first N decoder layers (a smoke test)")
    p.add_argument("--calib-seqs", type=int, default=128, help="GPTQ calibration sequences (wikitext-2 train)")
    p.add_argument("--configs", nargs="+", choices=list(CONFIGS),
                   help="default: all; with --ref, w4a8-rot w4a4-rot w4a4-rot-gptq")
    p.add_argument("--ref", help="llama-perplexity --kl-divergence-base file to score against")
    p.add_argument("--ref-bos", type=int, help="BOS id, if llama.cpp added one over each chunk's first token")
    p.add_argument("--out", default="accuracy/results/qwen3.8-27b-rtn.json")
    p.add_argument("--learn-r1", action="store_true", help="fit R1 to captured activations, write it to --out")
    p.add_argument("--r1-tokens", type=int, default=256, help="learn-r1: tokens sampled per sequence per norm")
    p.add_argument("--r1-steps", type=int, default=400)
    p.add_argument("--r1-batch", type=int, default=8192)
    p.add_argument("--r1-lr", type=float, default=2e-5)
    p.add_argument("--r1-init", help="learn-r1: start from this R1 file instead of the randomized Hadamard")
    p.add_argument("--clip", type=float, default=0.8, help="learn-r1: activation clip ratio")
    p.add_argument("--recipes", nargs="*", choices=list(RECIPES),
                   help="recipe mode (needs --ref): GPTQ recipes in one pass; no names = all")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.learn_r1:
        learn_r1(a)
    elif a.recipes is not None:
        run_recipes(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
