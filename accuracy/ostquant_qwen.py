#!/usr/bin/env python3
"""OSTQuant (ICLR 2025, github.com/BrotherHappy/OSTQuant) for Qwen3.8-27B, layer-streamed on one GPU.

The repo's W4A4 recipe (scripts/w4a4kv16.sh) where Qwen3.8 allows it:
  - R_res: one global residual rotation (5120 x 5120, a geoopt Stiefel parameter), KLT init (eigenvectors of
    the covariance of the norm-fused reader weights) @ randomized Hadamard;
  - smoothing scales: S_norm_qkv and S_norm_upgate (applied after the rotation: norm weight 1/s, readers
    W R diag(s)), S_up_down (up rows / s, down columns * s, before down's online Hadamard), S_ov (v rows / s,
    o_proj columns * s, GQA-expanded);
  - optimizer as the repo's default (opt_type RAdam): geoopt RiemannianAdam, rotation lr 0.0169, smoothing
    lr 0.00179, stabilize 10, linear decay, 100 steps;
  - loss kl_top with post_attn (k = 1000): KL of the BF16 model's softmax at its top-1000 tokens against the
    quantized model's full-vocabulary log-softmax at those tokens, mean over tokens;
  - weights full precision during training (train_enable_wquant=False), activations through the 4-bit
    per-token asymmetric quantizer, straight-through rounding, clip 1.0. GPTQ comes after, in Rust.
Not ported because Qwen3.8 can't fold them: R_ov (the attention output gate sits between V and o_proj; o_proj
gets our online Hadamard instead, as down_proj does) and S_qk (the KV cache stays fp). Quantized inputs are the
ones unsloth's Q4_0 map makes 4-bit: the two norm outputs (minus DeltaNet's F32 alpha/beta, which read the
unquantized norm output), down_proj's and attention o_proj's inputs. DeltaNet's out_proj stays Q5_K.

With full-precision weights, rotating the whole residual stream equals rotate -> scale -> quantize -> unscale ->
unrotate around each quantized input, so the student is the BF16 model with those wrappers; nothing is folded
per step. Per step: the teacher and the student run layer by layer (weights streamed from the checkpoint), the
student's layer inputs stay in host memory, the loss gives the gradient at the last hidden state, and each layer
is recomputed backwards from its saved input.

    .venv/bin/python accuracy/ostquant_qwen.py --out models/ost
Writes OUT/r1.f32 (5120 x 5120 row-major, y = x R1) and OUT/blk.N.{s_in,s_post,s_ud,s_ov}.f32 for
`ruquant convert --ost OUT`; OUT/scales.safetensors, OUT/opt.pt and OUT/log.json for --resume.
"""

import argparse
import json
import os
import random
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_w4a4 as E  # noqa: E402

HIDDEN = E.HIDDEN


# ---------------------------------------------------------------------------- quantizer with STE
def quant_ste(x, bits=4, clip=1.0):
    """The spec's per-token asymmetric quantizer in fp32 with straight-through rounding; the scale keeps its
    gradient through min / max, as the repo's quantizer does."""
    levels = 2**bits - 1
    x = x.float()
    mn = x.amin(-1, keepdim=True).clamp(max=0) * clip
    mx = x.amax(-1, keepdim=True).clamp(min=0) * clip
    scale = (mx - mn).clamp_min(1e-8) / levels
    z = torch.round(-mn / scale).detach().clamp(0, levels)
    u = x / scale
    q = torch.clamp(u + (torch.round(u) - u).detach() + z, 0, levels)
    return (q - z) * scale


def canonical_hadamard(n, dev):
    """ggml_rq_hadamard's H (Paley kron Sylvester / sqrt(n), no signs): the online rotation the runtime uses."""
    R = E.Rotation(n, 0, dev)
    R.signs = torch.ones_like(R.signs)
    return R


# ---------------------------------------------------------------------------- wrappers around the BF16 layer
class QNorm(nn.Module):
    """RMSNorm (zero-centered weight w) whose output the W4A4 readers quantize: x_n -> x_n R / s -> Q -> * s R^T,
    then * (1 + w), so the original reader weights apply unchanged. The unquantized output is kept in .fp."""

    def __init__(self, norm, R, s, clip):
        super().__init__()
        self.w, self.eps, self.R, self.s, self.clip = norm.weight, norm.eps, R, s, clip
        self.fp = None

    def forward(self, h):
        x = h.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        w = 1.0 + self.w.float()
        self.fp = (x * w).to(h.dtype)
        a = (x @ self.R) / self.s
        x = (quant_ste(a, 4, self.clip) * self.s) @ self.R.t()
        return (x * w).to(h.dtype)


class FpTap(nn.Module):
    """A reader kept in full precision (DeltaNet's F32 alpha / beta): reads the norm's unquantized output."""

    def __init__(self, lin, norm):
        super().__init__()
        self.lin, self.norm = lin, norm

    def forward(self, x):
        return self.lin(self.norm.fp)


class QIn(nn.Module):
    """A writer whose input gets smoothing s and the online Hadamard H before quantization."""

    def __init__(self, lin, H, s, clip):
        super().__init__()
        self.lin, self.H, self.s, self.clip = lin, H, s, clip

    def forward(self, x):
        dt = x.dtype
        b = self.H(x.float() / self.s)
        x = self.H.inverse(quant_ste(b, 4, self.clip)) * self.s
        return self.lin(x.to(dt))


def wrap_layer(layer, params, i, Hs, clip, n_kv, groups):
    p = params[i]
    qin = QNorm(layer.input_layernorm, params["R"], p["s_in"], clip)
    layer.input_layernorm = qin
    layer.post_attention_layernorm = QNorm(layer.post_attention_layernorm, params["R"], p["s_post"], clip)
    layer.mlp.down_proj = QIn(layer.mlp.down_proj, Hs[17408], p["s_ud"], clip)
    if hasattr(layer, "self_attn"):
        s_ov = p["s_ov"].view(n_kv, 1, -1).expand(n_kv, groups, -1).reshape(-1)  # o_proj input is per q head
        layer.self_attn.o_proj = QIn(layer.self_attn.o_proj, Hs[6144], s_ov, clip)
    else:
        la = layer.linear_attn
        la.in_proj_a, la.in_proj_b = FpTap(la.in_proj_a, qin), FpTap(la.in_proj_b, qin)
    return layer


# ---------------------------------------------------------------------------- weight streaming
class Prefetch:
    """posix_fadvise(WILLNEED) on a layer's byte ranges in the safetensors shards, so the kernel reads the next
    layer while this one computes."""

    def __init__(self, ck):
        self.ranges, self.fds = {}, {}
        for f in set(ck.where.values()):
            path = ck.dir / f
            self.fds[f] = os.open(path, os.O_RDONLY)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                for k, v in json.loads(fh.read(n)).items():
                    if k != "__metadata__":
                        s, e = v["data_offsets"]
                        self.ranges[k] = (f, 8 + n + s, e - s)

    def layer(self, i):
        p = f"{E.PREFIX}layers.{i}."
        for k, (f, off, n) in self.ranges.items():
            if k.startswith(p):
                os.posix_fadvise(self.fds[f], off, n, os.POSIX_FADV_WILLNEED)


# ---------------------------------------------------------------------------- KLT init
@torch.no_grad()
def klt_init(ck, n_layers, dev):
    """The repo's KLT R_res: eigenvectors of the covariance of every norm-fused reader row, @ a randomized
    Hadamard."""
    S = torch.zeros(HIDDEN, HIDDEN, device=dev, dtype=torch.float64)
    mean = torch.zeros(HIDDEN, device=dev, dtype=torch.float64)
    n = 0
    readers_in = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "self_attn.q_proj", "self_attn.k_proj",
                  "self_attn.v_proj"]
    for i in range(n_layers):
        sd = ck.layer(i)
        w_in = 1 + sd["input_layernorm.weight"].float().to(dev)
        w_post = 1 + sd["post_attention_layernorm.weight"].float().to(dev)
        for name, w in [(r, w_in) for r in readers_in] + [("mlp.gate_proj", w_post), ("mlp.up_proj", w_post)]:
            key = name + ".weight"
            if key not in sd:
                continue
            W = sd[key].float().to(dev) * w
            S += (W.T @ W).double()
            mean += W.sum(0).double()
            n += W.shape[0]
    mean /= n
    cov = (S - n * torch.outer(mean, mean)) / (n - 1)
    _, V = torch.linalg.eigh(cov)
    H = E.Rotation(HIDDEN, 1, dev)(torch.eye(HIDDEN, device=dev))  # diag(signs) H, the randomized Hadamard
    return (V.float() @ H).contiguous()


# ---------------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8, help="sequences per step (the repo: 32 on 8 GPUs)")
    ap.add_argument("--micro", type=int, default=2, help="sequences per forward / backward chunk")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--rotate-lr", type=float, default=0.01689753172873217)
    ap.add_argument("--smooth-lr", type=float, default=0.0017898822675917248)
    ap.add_argument("--clip", type=float, default=1.0, help="activation clip during training (repo default 1.0)")
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=0, help="first N layers only (a smoke test)")
    ap.add_argument("--out", default="models/ost")
    ap.add_argument("--resume", action="store_true", help="continue from OUT")
    a = ap.parse_args()

    import geoopt
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    torch.backends.cuda.matmul.allow_tf32 = True
    dev, L = "cuda", a.seqlen
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(E.REPO, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    tcfg = Qwen3_5TextConfig(**json.loads((Path(path) / "config.json").read_text())["text_config"])
    tcfg._attn_implementation = "sdpa"
    n_layers = a.layers or tcfg.num_hidden_layers
    n_kv, groups = tcfg.num_key_value_heads, tcfg.num_attention_heads // tcfg.num_key_value_heads
    tok = AutoTokenizer.from_pretrained(path)
    ck = E.Checkpoint(path)
    pf = Prefetch(ck)

    # training windows: random 2048-token windows of wikitext-2 train
    ids = E.wikitext_ids(tok, "train")
    nwin = ids.numel() // L
    order = list(range(nwin))
    random.Random(0).shuffle(order)
    wins = [order[j % nwin] for j in range(a.steps * a.batch)]
    print(f"{n_layers} layers, {a.steps} steps x {a.batch} sequences of {L} ({nwin} windows in wikitext-2 train)",
          flush=True)

    log, start = [], 0
    if a.resume:
        import numpy as np
        R0 = torch.from_numpy(np.fromfile(out / "r1.f32", dtype=np.float32).reshape(HIDDEN, HIDDEN)).to(dev)
        sc = {k: v.to(dev) for k, v in load_file(str(out / "scales.safetensors")).items()}
        log = json.loads((out / "log.json").read_text())
        start = len(log)
    else:
        t0 = time.time()
        R0 = klt_init(ck, tcfg.num_hidden_layers, dev)
        print(f"KLT init in {time.time() - t0:.0f} s", flush=True)
        sc = {}
    params = {"R": geoopt.ManifoldParameter(R0.clone(), manifold=geoopt.Stiefel())}
    smooth = []
    for i in range(n_layers):
        p = {}
        dims = [("s_in", HIDDEN), ("s_post", HIDDEN), ("s_ud", tcfg.intermediate_size)]
        if tcfg.layer_types[i] == "full_attention":
            dims.append(("s_ov", n_kv * tcfg.head_dim))
        for k, n in dims:
            p[k] = nn.Parameter(sc.get(f"{i}.{k}", torch.ones(n, device=dev)).clone())
            smooth.append(p[k])
        params[i] = p
    opt = geoopt.optim.RiemannianAdam([{"params": [params["R"]], "lr": a.rotate_lr},
                                       {"params": smooth, "lr": a.smooth_lr}],
                                      lr=a.rotate_lr, weight_decay=0, stabilize=10)
    if a.resume and (out / "opt.pt").exists():
        opt.load_state_dict(torch.load(out / "opt.pt"))
    base_lr = [a.rotate_lr, a.smooth_lr]

    Hs = {6144: canonical_hadamard(6144, dev), 17408: canonical_hadamard(17408, dev)}
    rotary = M.Qwen3_5TextRotaryEmbedding(tcfg).to(dev)
    pe = rotary(torch.empty(1, device=dev, dtype=torch.bfloat16),
                torch.arange(L, device=dev).view(1, 1, -1).expand(3, a.micro, -1))
    embed = ck.get(E.PREFIX + "embed_tokens.weight")  # stays in host memory
    norm_f = M.Qwen3_5RMSNorm(HIDDEN, eps=tcfg.rms_norm_eps).to(dev)
    norm_f.weight.data.copy_(ck.get(E.PREFIX + "norm.weight").to(dev))
    lm = ck.get("lm_head.weight").to(dev)

    def layer_fp(i, nxt):
        if 0 <= nxt < n_layers:
            pf.layer(nxt)
        return E.build_layer(M, tcfg, i, E.transform(ck.layer(i), False, False, None, None), False, None)

    def run_layer(layer, h):
        out_ = []
        for s in range(0, h.shape[0], a.micro):
            x = h[s:s + a.micro].to(dev, non_blocking=True)
            n = x.shape[0]
            out_.append(layer(x, position_embeddings=(pe[0][:n], pe[1][:n])).to("cpu"))
        return torch.cat(out_)

    t_start = time.time()
    for step in range(start, a.steps):
        t0 = time.time()
        batch = torch.stack([ids[w * L:(w + 1) * L] for w in wins[step * a.batch:(step + 1) * a.batch]])
        with torch.no_grad():
            h0 = embed[batch].to(torch.bfloat16)
            # teacher (BF16) and student forward, layer by layer; the student's layer inputs kept for backward
            ht, hs = h0.clone(), h0.clone()
            saved = []
            for i in range(n_layers):
                layer = layer_fp(i, i + 1)
                ht = run_layer(layer, ht)
                wrap_layer(layer, params, i, Hs, a.clip, n_kv, groups)
                saved.append(hs)
                hs = run_layer(layer, hs)
                del layer
        t_fwd = time.time() - t0
        # kl_top (post_attn) on the final hidden state
        hs_g = hs.to(dev).requires_grad_(True)
        loss_sum = 0.0
        for s in range(hs.shape[0]):
            for t in range(0, L, 512):
                with torch.no_grad():
                    pt = torch.softmax(F.linear(norm_f(ht[s, t:t + 512].to(dev)), lm).float(), -1)
                    top_p, top_i = pt.topk(a.topk, -1)
                    del pt
                ls = torch.log_softmax(F.linear(norm_f(hs_g[s, t:t + 512]), lm).float(), -1)
                loss = F.kl_div(ls.gather(-1, top_i), top_p, reduction="sum") / (hs.shape[0] * L)
                loss.backward()
                loss_sum += loss.item()
        grad = hs_g.grad.to(torch.bfloat16).cpu()
        del hs_g
        # backward, layer by layer, recomputing each layer from its saved input
        for i in reversed(range(n_layers)):
            layer = wrap_layer(layer_fp(i, i - 1), params, i, Hs, a.clip, n_kv, groups)
            gin = []
            for s in range(0, a.batch, a.micro):
                x = saved[i][s:s + a.micro].to(dev).requires_grad_(True)
                n = x.shape[0]
                y = layer(x, position_embeddings=(pe[0][:n], pe[1][:n]))
                y.backward(grad[s:s + a.micro].to(dev))
                gin.append(x.grad.to(torch.bfloat16).cpu())
            grad = torch.cat(gin)
            del layer
            saved[i] = None
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * (1.0 - step / a.steps)  # the HF Trainer's default linear decay
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            R = params["R"].data
            orth = (R.T @ R - torch.eye(HIDDEN, device=dev)).abs().max().item()
            s_min, s_max = min(v.min().item() for v in smooth), max(v.max().item() for v in smooth)
        el = time.time() - t0
        log.append({"step": step, "loss": loss_sum, "s": el, "orth": orth, "s_min": s_min, "s_max": s_max})
        done = step + 1 - start
        print(f"step {step:3d}  kl_top {loss_sum:.5f}  {el:4.0f} s (fwd {t_fwd:.0f})  orth {orth:.1e}  "
              f"s {s_min:.3f}..{s_max:.3f}  eta {(time.time() - t_start) / done * (a.steps - step - 1) / 60:.0f} min",
              flush=True)
        if step % 5 == 4 or step == a.steps - 1:
            save(out, params, n_layers, log, opt)
    print(f"done in {(time.time() - t_start) / 60:.0f} min -> {out}")


def save(out, params, n_layers, log, opt):
    from safetensors.torch import save_file

    def put(name, t):
        t.detach().float().cpu().numpy().tofile(out / (name + ".tmp"))
        (out / (name + ".tmp")).replace(out / name)

    put("r1.f32", params["R"].data)
    sc = {}
    for i in range(n_layers):
        for k, v in params[i].items():
            put(f"blk.{i}.{k}.f32", v.data)
            sc[f"{i}.{k}"] = v.data.float().cpu().contiguous()
    save_file(sc, str(out / "scales.safetensors.tmp"))
    (out / "scales.safetensors.tmp").replace(out / "scales.safetensors")
    torch.save(opt.state_dict(), out / "opt.pt.tmp")
    (out / "opt.pt.tmp").replace(out / "opt.pt")
    (out / "log.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
