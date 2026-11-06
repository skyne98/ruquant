# ruquant: final results (2026-11-05)

Qwen3.8-27B in pure W4A4: int4 weights × int4 activations, multiplied with `v_dot8_i32_i4` on gfx906. It is served by
mx-llama.cpp with our patch and DFlash2 speculative decoding, on AMD MI50s.

Everything below was measured on box 1 (4× MI50 32 GB, PCIe 3.0 through the CPU) with the final build, unless a line
says otherwise. The default runtime is ROCm 7.12 from AMD's TheRock with whole-token HIP graphs; ROCm 5.7 is the
fallback, and its numbers are kept below.

The long version, with every experiment and dead end, is in `REPORT-2026-11-02.md` (speed and TP) and
`REPORT-2026-11-03-quality.md` (quality).

## Bottom line

Box 1, one session on 2026-11-05, ROCm 7.12 with HIP graphs (`BIN=~/mx-llama.cpp/build-r712/bin
ROCM_LIB=~/therock-7.12/lib GRAPHS=1 rig/final_mi50.sh`). The W4A4 model is v2 with its W4A4H DFlash2 drafter
(draft 7); the baseline is unsloth's Q4_0 (4-bit weights, 8-bit activations) with incoai's DFlash2 (draft 4, its best)
on the same runtime.

| | 1 GPU | TP2 | TP4 |
|---|---|---|---|
| **W4A4 + DFlash2, t/s** | **84.2** | **98.4** | **113.3** |
| acceptance (tokens per step) / ms per step | 4.33 / 51.4 | 4.21 / 42.8 | 4.31 / 38.0 |
| math / code / chat, t/s | 110.3 / 87.7 / 65.5 | 132.4 / 101.8 / 77.0 | 150.8 / 115.0 / 89.1 |
| Q4_0 + DFlash2, t/s | 45.7 | 62.6 | 73.1 |
| Q4_0 acceptance / ms per step | 3.67 / 80.2 | 3.70 / 59.1 | 3.68 / 50.3 |
| **verify forward (8 tokens), ms:** W4A4 / Q4_0 | **40.6** / 91.6 | **30.7** / 54.4 | **25.5** / 37.2 |
| **prefill pp512, t/s:** W4A4 / Q4_0 | **500.3** / 302.2 | **823.1** / 529.5 | **1200.4** / 832.6 |
| **prefill pp2048, t/s:** W4A4 / Q4_0 | **503.7** / 306.8 | **881.7** / 562.7 | **1388.9** / 953.2 |
| plain decode tg128, t/s: W4A4 / Q4_0 | 31.6 / 28.9 | 46.8 / 46.5 | 60.1 / 64.2 |

- **Speculative decoding: +84% over Q4_0 on one GPU, +57% at TP2, +55% at TP4.** A W4A4 verify of 8 tokens takes 2.25×
  less time than Q4_0's on one GPU, and the drafter accepts more tokens per step (4.2-4.3 vs 3.7).
- **Prefill: +64-66% over Q4_0 on one GPU, +44-57% at TP2 / TP4.** TP4 pp2048 at 1389 t/s is above castagna-veloce's
  published 943 (4 cards, UD-Q4_K_XL, PCIe 4.0).
- **Plain decode, one token per pass, is bandwidth-bound and about level with Q4_0** (+9% on one GPU, even at TP2,
  -6% at TP4). The gain is in multi-token passes, which is why it comes with speculative decoding.
- **Q4_0's DFlash2 under TP runs with its drafter replicated on every GPU.** incoai's drafter borrows the target's
  lm_head, which `-sm tensor` splits, so pinning it to one GPU aborts at load. Our converted drafter has its own
  W4A4H head and is pinned. For our drafter, replication costs 19%, so Q4_0's TP numbers carry part of that handicap.
- **The first TP4 speculative run of a session can run slower** (41.8 ms per step seen on 5.7 with either model);
  repeated runs agree within ~1%.
- **Hardware:** box 1 is 4× MI50 32 GB (gfx906, 1 TB/s HBM2, ~826 GB/s measured), GPUs on PCIe 3.0 through the CPU
  (peer writes 5-6.5 GB/s), 12 CPU cores, Ubuntu 22.04. One session; box 1 drifts by up to ±3% between sessions.

**ROCm 7.12 against the earlier ROCm 5.7 runtime:**
- Faster everywhere: DFlash2 +5% / +10% / +3% at 1 GPU / TP2 / TP4; plain decode +8% / +13% / +14%; prefill +2-6%.
- Identical quality: the same tokens and the same KLD to the last digit (below).
- The ROCm 5.7 table, same model, earlier the same day (`rig/final_mi50.sh` without `GRAPHS`):

| ROCm 5.7 | 1 GPU | TP2 | TP4 |
|---|---|---|---|
| W4A4 + DFlash2, t/s (acceptance) | 80.2 (4.35) | 89.5 (4.27) | 109.6 (4.31) |
| Q4_0 + DFlash2, t/s | 43.5 | 60.5 | 70.5 |
| verify forward, ms: W4A4 / Q4_0 | 42.2 / 95.5 | 35.1 / 62.8 | 27.6 / 40.6 |
| prefill pp2048: W4A4 / Q4_0 | 487.8 / 290.7 | 833.1 / 529.0 | 1324.0 / 901.4 |
| plain decode tg128: W4A4 / Q4_0 | 29.2 / 27.9 | 41.6 / 39.7 | 52.7 / 51.5 |

## Quality

KL divergence and top-1 agreement against UD-Q8_K_XL's logits, wikitext-2 test, context 2048.

| | KLD | top-1 | PPL / base |
|---|---|---|---|
| unsloth UD-Q4_K_XL (3090, 64 chunks) | 0.0078 | 96.58% | 0.9989 |
| unsloth Q4_0, 8-bit activations (3090, 64 chunks) | 0.0259 | 93.34% | 1.0220 |
| **W4A4 v2 (3090, 64 chunks)** | **0.0428** | **91.57%** | 1.0279 |
| W4A4 v2, box 1, one GPU (16 chunks; ROCm 7.12 and 5.7 identical) | 0.0425 | 91.44% | 1.0312 |
| W4A4 v2, box 1, TP4 (16 chunks; ROCm 7.12 and 5.7 identical) | 0.0420 | 91.57% | 1.0296 |
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

**ROCm 7.12, the default.** AMD's TheRock nightly for the gfx90X family (`7.12.0a20260225`, the last gfx90X
nightly), installed in user space:
- Unlike AMD's packaged ROCm 7.x, it ships gfx906 code in RCCL and rocBLAS.
- It needs only glibc 2.27, so it runs on Ubuntu 22.04 without root.
- With it, whole-token HIP graphs pay off: ROCm 5.7 replays a graph as ordinary launches, 7.12 as pre-built work.

```sh
git clone -b ruquant https://github.com/skyne98/mx-llama.cpp ~/mx-llama.cpp
git clone https://github.com/skyne98/ruquant ~/ruquant
~/ruquant/rig/therock/setup.sh    # ~/therock-7.12 (4.7 GB slim install) and ~/mx-llama.cpp/build-r712
```

- Builds are faster on a workstation: `rig/therock/build_local.sh ~/mx-llama.cpp HOST` builds in an Ubuntu 22.04
  container (85 s on 32 threads, 19 s incremental) and copies `bin/` to the box.
- W4A4 / W4A4H MUL_MAT is bit-identical to the CPU reference (`test-backend-ops`, zero tolerance, 56 + 28 shapes,
  plus all six fused Hadamard variants with `RQ_ROT_BLOCKS=2/4/8`).
- The fork is mx-llama.cpp at c4d12e2 plus `patches/mx-llama.cpp-ruquant.patch` (40 files); box 1's tree is exactly
  that.

**Environment** (box 1):

```sh
export LD_LIBRARY_PATH=~/mx-llama.cpp/build-r712/bin:~/therock-7.12/lib
export HSA_FORCE_FINE_GRAIN_PCIE=1    # needed for the peer-write all-reduce under -sm tensor
export GGML_META_TOKEN_GRAPH=1        # whole-token HIP graphs (unset also means on); leave GGML_CUDA_DISABLE_GRAPHS unset
export GGML_META_SPIN=2000000         # TP lane threads spin instead of sleeping: +2%, and stable clocks
```

**Server, TP4 with speculative decoding** (the fastest configuration):

```sh
~/mx-llama.cpp/build-r712/bin/llama-server \
    -m qwen3.8-27b-ost44-v2-wg64-rb4-gptq.gguf -md dflash2-ost44-h4-l4-c8-g64-rb4.gguf \
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
  - whole-token HIP graphs reused across rebuilds by a fingerprint of the per-GPU graphs and a scratch-pool epoch
    (`GGML_META_TOKEN_GRAPH=0`). With them on, single-stage TP drops the scheduler's input ring
    (`LLAMA_TP_PIPELINE=1` keeps it);
  - the two-rank all-reduce gate at 131072 values, so TP2 verify all-reduces stay on the custom kernel instead of RCCL
    (TP2 DFlash2 +6% on 5.7, and it lets graphs record TP2);
  - prefill all-reduces travel as q4_K blocks (`GGML_NCCL_AR_WIRE=f16` turns it off; +39% TP4 prefill, a few % KLD on
    prompts only);
  - the online Hadamard and the activation quantizer fused into two launches (`RQ_FUSE_HADAMARD=0`);
  - the skinny kernel's split-K rules (`RQ_SKINNY_OLD_G=1`);
  - 24-token prefill chunks and the tiled GEMM from 64 tokens (`RQ_CHUNK`, `RQ_GEMM_MIN`).

**ROCm 5.7, the fallback** (the box's own `/opt/rocm`):
- Build with `rig/build_mx.sh` into `~/mx-llama.cpp/build`.
- Run with `GGML_CUDA_DISABLE_GRAPHS=1 GGML_META_TOKEN_GRAPH=0` and the other variables above.
- On 5.7 HIP graphs are slower and flaky, and without token graphs the scheduler keeps its input ring (dropping it
  cost 5.7 ~1.9 ms per forward on the GPU).
- Numbers: the ROCm 5.7 table in the bottom line.

**Not used:**
- AMD's packaged ROCm 7.2.4 (`rig/rocm72/setup.sh`): its RCCL has no gfx906 code, so TP prefill is 35-38% slower.
- ROCm 10.1 from mixa3607/ML-gfx906 (`rig/rocm101/build_local.sh`; Ubuntu 24.04 container build, a private glibc on
  the box): the same tokens and KLD as 7.12, but 1-2% slower. TP4 DFlash2 is 111.3 against 113.6 t/s, and TP4 pp2048
  1360 against 1386.
  - Its RCCL and rocBLAS keep their gfx906 code in `.kpack` files next to `lib`; without them TP prefill collapses.
- Details: REPORT-2026-11-02.md, "whole-token HIP graphs on ROCm 7.2.4" and "ROCm 10.1 from mixa3607's gfx906
  packages".

## Long context (box 1, 2026-11-06, local fork build)

At 64k tokens the cost is attention in the 16 full-attention layers. The DeltaNet layers cost the same at any depth,
and the drafter attends over a 4096-token window. On the published build with an f16 cache, a DFlash2 step at 64k
takes 79-80 ms on one GPU (51 ms with a short prompt), 57-59 on TP2 (42-44) and 44-48 on TP4 (36-38). A 64k prefill
takes 243 / 134 / 84 s.

**`-ctk q4_0 -ctv q4_0`** turns on the fork's own attention over a 4-bit cache (`rq-fattn.cu`). It uses the rotated
q4_0 nibbles straight into `v_dot8` for Q·K and f16 dot2 for P·V, for decode and the verify.
- On one GPU at 64k it takes the DFlash2 step to 71-72 ms (-11%), the verify forward from 64.4 to 59.6 ms, and decode
  from 44.7 to 38.5 ms.
- TP4 steps are unchanged.
- The cache shrinks from 4.35 to 1.2 GB.
- KLD stays within noise of the f16 cache: 0.0423 against 0.0416, 16 chunks at verify width.
- llama.cpp's own q4_0 / q8_0 attention is slower than f16 on gfx906 at depth (decode 3× slower with q4_0).

This is in the local fork tree and `patches/`, not in the published fork commit. Details: REPORT-2026-11-02.md, "64k
context", "quantized KV cache on our model" and "our q4 KV attention with v_dot8".

## Methodology

- **Quality.** `llama-perplexity --kl-divergence` against UD-Q8_K_XL's logits on wikitext-2 test, context 2048:
  - 16 chunks (33k tokens), about ±3% noise; 64 chunks (131k tokens) for decisions. `accuracy/score.sh` on the 3090,
    `rig/score_box.sh` on an MI50.
  - TP runs are scored with the prefill wire in f16 (`GGML_NCCL_AR_WIRE=f16`). Otherwise perplexity quantizes every
    token's all-reduce, where serving quantizes only the prompt's.
- **Speed.** `llama-bench -fa 1`, 3 repetitions: pp512 (ubatch 512), pp2048 (ubatch 2048), tg128 (plain decode).
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
- **Reproduce the final table:** `BIN=~/mx-llama.cpp/build-r712/bin ROCM_LIB=~/therock-7.12/lib GRAPHS=1
  rig/final_mi50.sh MODEL DRAFTER` (W4A4), and the same with `NMAX=4 SCORE=0 DEVD=` for
  `Qwen3.8-27B-Q4_0.gguf Qwen3.8-27B-DFlash2-Q8_0.gguf` (the baseline; its drafter can't be pinned under TP).
  Without `BIN`, `ROCM_LIB` and `GRAPHS` it measures the ROCm 5.7 build.
- **The 8-token forward** is the median of repetitions 11-30 of `llama-bench -p 8 -r 30`, after any graph is
  recorded.

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
| HIP graphs on ROCm 5.7 (single GPU and the TP token graph) | slower and unstable (TP4 T=8: 200 ± 40 t/s vs ~285 without); on ROCm 7.12 they pay (above) |
| AMD's packaged ROCm 7.2.4 | graphs pay (TP4 +6%), but its RCCL has no gfx906 code: TP prefill -35-38% |
| ROCm 10.1 (mixa3607's gfx906 TheRock packages) | same tokens and KLD as 7.12, 1-2% slower (TP4 DFlash2 111.3 vs 113.6 t/s, 1 GPU 80.8 vs 84.2) |
| drafter replicated on all GPUs / split TP4 | -19% / -24% (mx's multi-GPU wrapper: ~2 ms graph build and 5.7-7.3 ms enqueue per drafter graph) |
| draft length 5 / 6 at TP4 | -6% / -4% |
| `NCCL_P2P_LEVEL=SYS` for prefill all-reduces | slower (TP4 pp512 882 -> 723) |
| bf16 TP all-reduce payload | KLD +4% and lower acceptance; f16 kept |
| TP lanes free-running (one host join per forward) | no change: the TP4 verify is GPU-bound |
| no split-K at TP4 (benchmark said faster) | 292 -> 288 t/s in the forward |
| stream-K for the skinny W4A4 kernel | same speed as split-K at the same grid; the forward did not gain at any grid size (TP4 25.3 -> 25.3-25.9 ms) |
| Hadamard + quantizer as one kernel, one block per token | 4× slower per call. Redesigned as two launches (shipped): +0.8-1.9% |
| a 1024-thread activation quantizer | no change |
| W4A4 lm_head (target) | KLD 0.0846 -> 0.107; Q8_0 kept |
| DSpark drafter | lower acceptance than DFlash2 |
| quality: GPTAQ, OSTQuant retrained for groups or longer (step 64), learned weight scales (`--refit`), 128 windows, wikitext + C4, per-group activation clip search, any activation clip on the 27B | within noise or worse |

## Where the time goes now, and what would move it

Measured on ROCm 5.7; on 7.12 the graphs take ~2 ms off the verify and the step is 38.0 ms. A TP4 DFlash2 step
took ~39.5 ms for 4.26 tokens:
- **Drafter, ~6.2 ms**, on GPU 0 while the other three wait. Per draft: lm_head 0.97 ms, layers 1.67, top-k over 248k
  0.40, Hadamard and quantize 0.45, ~150 small glue kernels 0.8.
- **Verify, ~31 ms**: launching and waiting for the 8-token forward (27.4 ms of it on the GPUs). Per GPU that is about:
  - 9.6 ms of W4A4 matmuls (5.6 at full bandwidth: each wave keeps one step of weights in flight, and grids whose
    shares are a power-of-two number of activation groups line the streams up on the same DRAM channels);
  - 5.7 ms in 128 all-reduces (~45 us each over PCIe 3.0 peer writes);
  - ~6 ms of small replicated ops;
  - 2.9 ms of Q8_0 `ssm_out`.
- **Injecting the verified tokens into the drafter, 1.1 ms; acceptance, 0.5 ms.**
- **Graph rebuilds**, ~1.7 ms per step on 256-token requests: ~21 ms each under TP4, about 4 per request.

What would move it, by size:
- A drafter fine-tuned on the W4A4 target: acceptance 4.26 against 4.66 with the Q6_K reference, up to +9%.
- A reduced-vocabulary drafter head with top-k on the small logits: ~3% if acceptance holds.
- A cheaper TP graph rebuild and drafter graph reuse: 1-2%. Capturing the four lanes' token graphs in parallel: ~1%.
- Faster GPU links: the all-reduces are ~20% of the verify.

Quality is the other open front: KLD 0.043 against Q4_0's 0.026. The cheap weight-side knobs are used up; the next
lever is training (block-wise reconstruction).
