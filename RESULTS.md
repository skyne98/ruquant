# ruquant: final results (2026-11-05)

Qwen3.8-27B in pure W4A4: int4 weights × int4 activations, multiplied with `v_dot8_i32_i4` on gfx906. It is served by
mx-llama.cpp with our patch and DFlash2 speculative decoding, on AMD MI50s. Everything below was measured on box 1
(4× MI50 32 GB, ROCm 5.7.1, PCIe 3.0 through the CPU) with the final build, unless a line says otherwise. The long
version, with every experiment and dead end, is in `REPORT-2026-11-02.md` (speed and TP) and
`REPORT-2026-11-03-quality.md` (quality).

## Bottom line

Box 1, one session on 2026-11-05 (`rig/final_mi50.sh`). The W4A4 model is v2 with its W4A4H DFlash2 drafter
(draft 7); the baseline is unsloth's Q4_0 (4-bit weights, 8-bit activations) with incoai's DFlash2 (draft 4, its
best).

| | 1 GPU | TP2 | TP4 |
|---|---|---|---|
| **W4A4 + DFlash2, t/s** | **80.2** | **89.5** | **109.6** |
| acceptance (tokens per step) / ms per step | 4.35 / 54.2 | 4.27 / 47.7 | 4.31 / 39.4 |
| math / code / chat, t/s | 103.9 / 84.3 / 62.4 | 119.8 / 86.3 / 74.2 | 137.2 / 102.5 / 82.7 (the 103.2 t/s first run) |
| Q4_0 + DFlash2, t/s | 43.5 | 60.5 | 70.5 |
| Q4_0 acceptance / ms per step | 3.67 / 84.4 | 3.68 / 60.8 | 3.68 / 52.1 |
| **verify forward (8 tokens), ms:** W4A4 / Q4_0 | **42.2** / 95.5 | **35.1** / 62.8 | **27.6** / 40.6 |
| **prefill pp512, t/s:** W4A4 / Q4_0 | **487.5** / 264.7 | **788.9** / 502.8 | **1172.0** / 804.9 |
| **prefill pp2048, t/s:** W4A4 / Q4_0 | **487.8** / 290.7 | **833.1** / 529.0 | **1324.0** / 901.4 |
| plain decode tg128, t/s: W4A4 / Q4_0 | 29.2 / 27.9 | 41.6 / 39.7 | 52.7 / 51.5 |

- **Speculative decoding: +84% over Q4_0 on one GPU, +48% at TP2, +55% at TP4.** A W4A4 verify of 8 tokens takes 2.3×
  less time than Q4_0's on one GPU, and the drafter accepts more tokens per step (4.3 vs 3.7).
- **Prefill: +68-84% over Q4_0 on one GPU, +46-57% at TP2 / TP4.** TP4 pp2048 at 1324 t/s is above castagna-veloce's
  published 943 (4 cards, UD-Q4_K_XL, PCIe 4.0).
- **Plain decode, one token per pass, is bandwidth-bound and about ties with Q4_0 (+2-5%).** The gain is in multi-token passes,
  which is why it comes with speculative decoding.
- **TP4 W4A4 + DFlash2 109.6 t/s is the mean of two back-to-back runs (109.2, 110.0).** The first TP4 speculative run of
  each session ran at 41.8 ms per step (101.8-103.2 t/s) with either the v2 or the previous model. The previous
  model measured 108.5 next to them, so v2 costs no speed.
- **Q4_0's DFlash2 under TP runs with its drafter replicated on every GPU.** incoai's drafter borrows the target's
  lm_head, which `-sm tensor` splits, so pinning it to one GPU aborts at load. Our converted drafter has its own
  W4A4H head and is pinned. For our drafter, replication costs 19%, so Q4_0's TP numbers carry part of that handicap.
- **Hardware:** box 1 is 4× MI50 32 GB (gfx906, 1 TB/s HBM2, ~826 GB/s measured), ROCm 5.7.1, GPUs on PCIe 3.0
  through the CPU (peer writes 5-6.5 GB/s), 12 CPU cores. One session; box 1 drifts by up to ±3% between sessions.

## Quality

KL divergence and top-1 agreement against UD-Q8_K_XL's logits, wikitext-2 test, context 2048.

| | KLD | top-1 | PPL / base |
|---|---|---|---|
| unsloth UD-Q4_K_XL (3090, 64 chunks) | 0.0078 | 96.58% | 0.9989 |
| unsloth Q4_0, 8-bit activations (3090, 64 chunks) | 0.0259 | 93.34% | 1.0220 |
| **W4A4 v2 (3090, 64 chunks)** | **0.0428** | **91.57%** | 1.0279 |
| W4A4 v2, box 1, one GPU (16 chunks) | 0.0425 | 91.44% | 1.0312 |
| W4A4 v2, box 1, TP4 (16 chunks) | 0.0420 | 91.57% | 1.0296 |
| previous W4A4, ost44-wg-rb4 (3090, 64 chunks) | 0.0513 | 90.54% | 1.0337 |
| first W4A4: plain Hadamard + GPTQ, per-token activations (16 chunks) | 0.0846 | 87.9% | 1.046 |

- **v2 sits 0.017 KLD above Q4_0.** On the previous model the error map (`accuracy/attrib27b.sh`) put 46% of the
  damage on activation quantization: with 16-bit activations our own int4 weights score 0.0265, Q4_0's level. By
  class, ffn_gate+up holds 43% of the damage, ffn_down 27%, the last 4 layers 18%, DeltaNet inputs 12% and attention
  8%.
- **The MI50 and the 3090 agree within the 16-chunk noise (±3%).** The W4A4 matmuls are bit-identical across
  backends; the other ops' float rounding differs, which moves a W4A4 model's KLD by a few percent.
- **From the first W4A4 model to v2:**
  - OSTQuant instead of a plain Hadamard: -12% KLD;
  - activation groups of 64 instead of one range per token, with per-256 weight scales: -31%;
  - Hessians from every position and 64 windows, plus weight groups of 64: -17%.
  - All of it is free at runtime: same kernels, same file size.

## The model

`models/qwen3.8-27b-ost44-v2-wg64-rb4-gptq.gguf` (17.18 GB; box 1 `~/models`), with the drafter
`models/dspark/dflash2-ost44-h4-l4-c8-g64-rb4.gguf` (2.65 GB).

**Types.** unsloth's Qwen3.8-27B-Q4_0 type map. Each of its 4-bit linears becomes ours:
- `W4A4` for the inputs of a norm: attn q/k/v and gate, DeltaNet qkv / z / alpha-beta, ffn_gate / ffn_up;
- `W4A4H` (an online Hadamard first) for attn_output and ffn_down.

`ssm_out` and the lm_head are Q8_0 (for speed and accuracy over Q5_K / Q6_K); token_embd stays Q4_0; the rest keeps
unsloth's choice.

**The format.**
- Weights: 4.5 bits per weight. Blocks of 256 hold a 16-byte header and 128 bytes of codes, with a per-row fp32
  scale and four 4-bit multipliers per block, one per group of 64 (`--wgroups 64`).
- Activations, quantized on the fly per token: asymmetric int4 over [min(x,0), max(x,0)] in groups of 64. Each group
  has its own zero point and a scale that is an integer multiple (1-255) of the token's unit, so a dot product stays
  in int32 until one fp32 epilogue.
- The online Hadamard is block-diagonal: 4 blocks per row (`--rot-blocks 4`), each H_p ⊗ H_m (Paley × Sylvester),
  e.g. 4352 = 68 × 64 for ffn_down. So one GPU and TP2/TP4 run the same file.
- Activation clip 1.0 (unset): the 27B is worse with any clip.

**Recipe** (the commands; `./run.sh` is the ruquant CLI, on the RTX 3090):
1. OSTQuant: a learned R1 rotation and smoothing scales, from the repo's recipe in `accuracy/ostquant_qwen.py`.
   The step-44 checkpoint is `models/ost-s44` (training longer scored worse).
2. A rounded model to collect Hessians on:
   `./run.sh convert Qwen3.8-27B-UD-Q8_K_XL.gguf models/qwen3.8-27b-ost44-rtn.gguf --ost models/ost-s44
   --types-from Qwen3.8-27B-Q4_0.gguf`.
3. Hessians from every position of 64 wikitext windows:
   `./run.sh hessian models/qwen3.8-27b-ost44-rtn.gguf accuracy/ref/wikitext/wiki.train.raw models/hess-ost44-v2
   --seqs 64` (6270 s, 94 GB).
4. GPTQ conversion:
   `./run.sh convert Qwen3.8-27B-UD-Q8_K_XL.gguf models/qwen3.8-27b-ost44-v2-wg64-rb4-gptq.gguf --ost models/ost-s44
   --types-from Qwen3.8-27B-Q4_0.gguf --set ssm_out=q8_0 --set output.weight=q8_0 --hessians models/hess-ost44-v2
   --wgroups 64 --rot-blocks 4` (1537 s).
5. Drafter, from incoai's DFlash2 for Qwen3.8-27B. Its own lm_head and layers become W4A4H, its `fc` stays Q8_0
   with R1 folded into the input chunks, and it gets an unrotated token embedding:
   `./run.sh drafter Qwen3.8-27B-DFlash2-Q8_0.gguf Qwen3.8-27B-Q4_0.gguf
   models/dspark/dflash2-ost44-h4-l4-c8-g64-rb4.gguf --ost models/ost-s44 --head w4a4h --linears w4a4h
   --rot-blocks 4`.

## The runtime

**Build.** `rig/build_mx.sh` on the box: mx-llama.cpp at c4d12e2 plus `patches/mx-llama.cpp-ruquant.patch` (40
files, applies cleanly; box 1's tree is exactly that). It includes ROCm 5.7 fixes, RCCL on, HIP graphs compiled in
but turned off at run time. W4A4 / W4A4H MUL_MAT is bit-identical to the CPU reference (`test-backend-ops`, zero
tolerance, 56 + 28 shapes, plus all six fused Hadamard variants with `RQ_ROT_BLOCKS=2/4/8`).

**Environment** (box 1):

```sh
export GGML_CUDA_DISABLE_GRAPHS=1     # HIP graphs are slower and flaky on ROCm 5.7
export HSA_FORCE_FINE_GRAIN_PCIE=1    # needed for the peer-write all-reduce under -sm tensor
export GGML_META_TOKEN_GRAPH=0        # token-graph capture off (HIP graphs again)
export GGML_META_SPIN=2000000         # TP lane threads spin instead of sleeping: +2%, and stable clocks
```

**Server, TP4 with speculative decoding** (the fastest configuration):

```sh
llama-server -m qwen3.8-27b-ost44-v2-wg64-rb4-gptq.gguf -md dflash2-ost44-h4-l4-c8-g64-rb4.gguf \
    -ngl 99 -ngld 99 -fa 1 -sm tensor -devd ROCm0 \
    --spec-type draft-dflash --spec-draft-n-max 7 -bs --samplers temperature
```

- `-sm tensor` splits each layer across the visible GPUs (TP). One GPU: drop `-sm tensor -devd ROCm0`. TP2: two
  GPUs visible.
- `-devd ROCm0` pins the drafter to one GPU. Replicating it on all four (mx's default) is 19% slower; splitting it
  is 24% slower.
- Draft length 7, the most DFlash2's block of 8 allows, is the best: 5 and 6 are 6% and 4% slower at TP4.
- `-bs --samplers temperature` is greedy with backend sampling. Under `-sm tensor` our argmax fast path does the
  greedy pick on the CPU.
- Long prompts: add `-b 2048 -ub 2048 --ctx-checkpoints 0`. TP4 server prompt speed goes 845 -> 1066 t/s; the
  checkpoints cost 12-14% but are what lets the server reuse a cached prefix.
- Patch defaults that are on without flags, each with an off switch:
  - prefill all-reduces travel as q4_K blocks (`GGML_NCCL_AR_WIRE=f16` turns it off; +39% TP4 prefill, a few % KLD on
    prompts only);
  - the online Hadamard and the activation quantizer fused into two launches (`RQ_FUSE_HADAMARD=0`);
  - the skinny kernel's split-K rules (`RQ_SKINNY_OLD_G=1`);
  - 24-token prefill chunks and the tiled GEMM from 64 tokens (`RQ_CHUNK`, `RQ_GEMM_MIN`).

## Methodology

- **Quality.** `llama-perplexity --kl-divergence` against UD-Q8_K_XL's logits on wikitext-2 test, context 2048:
  - 16 chunks (33k tokens), about ±3% noise; 64 chunks (131k tokens) for decisions. `accuracy/score.sh` on the 3090,
    `rig/score_box.sh` on an MI50.
  - TP runs are scored with the prefill wire in f16 (`GGML_NCCL_AR_WIRE=f16`). Otherwise perplexity quantizes every
    token's all-reduce, where serving quantizes only the prompt's.
- **Speed.** `llama-bench -fa 1`, 3 repetitions:
  - pp512 (ubatch 512), pp2048 (ubatch 2048), tg128 (plain decode);
  - pp8, the 8-token forward a verify step runs.
- **Speculative decoding.** `rig/acc_bench.py`: llama-server, 30 prompts (10 math, 10 code, 10 chat) × 256 tokens,
  greedy, draft 7.
  - t/s is the mean of the server's per-request generation speed.
  - Acceptance is accepted tokens per verify step, the bonus token included.
- **Changes.**
  - Every speed change was checked for byte-identical generations (`rig/same_tokens.sh`, `rig/spec_bench.sh TEXT=`)
    or for unchanged acceptance where float summation order moved.
  - Kernel work was timed per kernel with one-GPU rocprof. rocprof hangs on TP4 forwards, so TP4 was judged by real
    forwards.
  - A slower experiment was traced to its cause (kernel times, ISA, occupancy) before it was dropped.
- **Noise.** Box 1 drifts between sessions by up to ±3% on speculative runs. Only same-session pairs compare, and
  the final table is one session.
- **Reproduce the final table:** `rig/final_mi50.sh MODEL DRAFTER` (W4A4), and
  `NMAX=4 SCORE=0 DEVD= rig/final_mi50.sh Qwen3.8-27B-Q4_0.gguf Qwen3.8-27B-DFlash2-Q8_0.gguf` (the baseline;
  its drafter can't be pinned under TP).

## Box 2 (2× MI50 16 GB, no peer access), 2026-11-04 build

The same v2 model and drafter, `-sm tensor` on 2 GPUs. Without P2P, every all-reduce goes through host memory via
RCCL. The build predates the 11-04/05 kernel work.

| box 2 | t/s |
|---|---|
| TP2 pp512 / pp2048 | 318 / 413 |
| TP2 tg128 | 31.3 |
| **TP2 + DFlash2, draft 7** | **70.0** (acceptance 4.26, 60.9 ms per step) |
| layer split (`-sm layer`), 2 GPUs: pp2048 / tg128 | 730-775 / 28.7 |

- Here layer split prefills 1.8× faster than TP, because it needs no all-reduce. TP still decodes faster with the
  drafter.
- Box 2 needs `--cache-ram 0`: the server's 8 GiB prompt cache gets it OOM-killed in a 12 GiB container.

## What did not pay

Measured, then dropped; numbers and causes are in the reports:

| | result |
|---|---|
| HIP graphs on ROCm 5.7 (single GPU and the TP token graph) | slower and unstable (TP4 T=8: 200 ± 40 t/s vs ~285 without) |
| drafter replicated on all GPUs / split TP4 | -19% / -24% (mx's multi-GPU wrapper: ~2 ms graph build and 5.7-7.3 ms enqueue per drafter graph) |
| draft length 5 / 6 at TP4 | -6% / -4% |
| `NCCL_P2P_LEVEL=SYS` for prefill all-reduces | slower (TP4 pp512 882 -> 723) |
| bf16 TP all-reduce payload | KLD +4% and lower acceptance; f16 kept |
| TP lanes free-running (one host join per forward) | no change: the TP4 verify is GPU-bound |
| no split-K at TP4 (benchmark said faster) | 292 -> 288 t/s in the forward |
| Hadamard + quantizer as one kernel, one block per token | 4× slower per call. Redesigned as two launches (shipped): +0.8-1.9% |
| a 1024-thread activation quantizer | no change |
| W4A4 lm_head (target) | KLD 0.0846 -> 0.107; Q8_0 kept |
| DSpark drafter | lower acceptance than DFlash2 |
| quality: GPTAQ, OSTQuant retrained for groups or longer (step 64), learned weight scales (`--refit`), 128 windows, wikitext + C4, per-group activation clip search, any activation clip on the 27B | within noise or worse |

## Where the time goes now, and what would move it

A TP4 DFlash2 step takes ~39.5 ms for 4.26 tokens:
- **Drafter, ~6.2 ms**, on GPU 0 while the other three wait. Per draft: lm_head 0.97 ms, layers 1.67, top-k over 248k
  0.40, Hadamard and quantize 0.45, ~150 small glue kernels 0.8.
- **Verify, ~31 ms**: launching and waiting for the 8-token forward (27.4 ms of it on the GPUs). Per GPU that is about:
  - 9.6 ms of W4A4 matmuls (5.6 at full bandwidth: wave quantization, ~120 resident workgroups);
  - 5.7 ms in 128 all-reduces (~45 us each over PCIe 3.0 peer writes);
  - ~6 ms of small replicated ops;
  - 2.9 ms of Q8_0 `ssm_out`.
- **Injecting the verified tokens into the drafter, 1.1 ms; acceptance, 0.5 ms.**
- **Graph rebuilds**, ~1.7 ms per step on 256-token requests: ~21 ms each under TP4, about 4 per request.

What would move it, by size:
- A drafter fine-tuned on the W4A4 target: acceptance 4.26 against 4.66 with the Q6_K reference, up to +9%.
- Stream-K for the skinny W4A4 kernel: up to ~6%.
- A reduced-vocabulary drafter head with top-k on the small logits: ~3% if acceptance holds.
- A cheaper TP graph rebuild and drafter graph reuse: 1-2%.
- Faster GPU links: the all-reduces are ~20% of the verify.

Quality is the other open front: KLD 0.043 against Q4_0's 0.026. The cheap weight-side knobs are used up; the next
lever is training (block-wise reconstruction).
