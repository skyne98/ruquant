# 2026-11-03: quality without giving up int4 x int4

The 4B proxy (`accuracy/proxy.sh`, Qwen3.5-4B: the 27B's hybrid layout, head dims and Paley x Sylvester Hadamard
shapes, tied embeddings that `convert` now unties) runs the whole pipeline in ~7 minutes and an ablation variant in
~20 s (`ruquant graft`, `ruquant a16`). KLD against BF16 log-probs, 16 wikitext chunks.

## Where the 27B's quality goes (KLD 0.0495 on the 3090; `accuracy/attrib27b.sh`)

| part | share of the damage | of which activation quantization |
|---|---|---|
| all activations at A16 (our own int4 weights) | 46% (0.0265 = Q4_0's 0.026) | |
| ffn_gate + ffn_up | 43% | about half |
| ffn_down | 27% | ~30% |
| last 4 layers | 18% | about half |
| DeltaNet inputs | 12% | |
| attention q/k/v/o | 8% | |

The 4B ranks the classes the same way (gate+up 43%, down 29%, DeltaNet inputs 14%).

## What moved the 4B (all free at runtime, bit-exact CPU / CUDA / HIP)

| | KLD | top-1 |
|---|---|---|
| unsloth Q4_0 | 0.0494 | 91.1% |
| W4A4: Hadamard R1, GPTQ, weight groups of 256 | 0.0725 | 88.1% |
| Hessians from every position (bug: the last layer's FFN got one token per window) | 0.0710 | 88.4% |
| + 64 calibration windows (128, or wikitext + C4: within noise) | 0.0673 | 89.0% |
| **+ weight groups of 64** (`convert --wgroups 64`) | **0.0637** | 89.0% |
| same at activation clip 0.95 (within 16-chunk noise; -1.7% on 64 chunks, below) | 0.0628 | 89.4% |

- The Hessian collector asked for the last token's logits only, and llama.cpp drops every other row before the last
  layer's FFN: ffn_gate/up/down of the last layer were GPTQ'd against 16 tokens. Every position is an output now.
- Weight groups of 64: four 4-bit multipliers per block (`sum = 1 << 30 | 1 << 29 | k0 | k1 << 4 | k2 << 8 | k3 <<
  12`, `RQ_W_MULT_G`; older files read as before). The kernels already fold per activation group of 64, so the
  multiplier is just indexed per 64: no speed change on the MI50 (dn.qkvz 81-82 us at T = 8, 2122-2123 us at 512 both
  ways). The repacked multipliers are int8 to fit the blocks' slack. Weight error alone (A16) fell 14% (0.0321 ->
  0.0275). Needs activation groups of 64 (asserted). test-backend-ops varies scales per 64 now (a per-256 index
  fails 44/56 shapes).
- Did not help: GPTAQ (asymmetric calibration, `hessian --fp`): its least-squares target shrinks the weights 8-14% per
  layer and that compounds (0.078 at a quarter of the correction, worse at full); OSTQuant retrained for the group
  quantizer (0.0758; at 10x lr the loss rises); global clips beyond 0.95 (noise).

## 64-chunk 4B reference: how much is noise

16 chunks (33k tokens) can't separate changes under ~3%: twelve activation clips between 0.93 and 1.0 scored
0.062-0.067 with no pattern. `accuracy/ref/q35-4b-bf16-64.kld` (32 GB, 131k tokens; BF16 with 16 of 32 layers on the
GPU) halves that; use `REF=accuracy/ref/q35-4b-bf16-64.kld accuracy/score.sh MODEL 64`.

| 4B, 64 chunks | KLD | top-1 |
|---|---|---|
| weight groups of 256 (s64 Hessians) | 0.0741 | 88.64% |
| weight groups of 64 | 0.0691 | 89.05% |
| + clip C 0.95 / CH 1.0 | 0.0694 | 89.14% |
| + clip C 1.0 / CH 0.95 | 0.0680 | 89.13% |
| + clip 0.97 / 0.97 | 0.0680 | 89.05% |
| + clip 0.95 / 0.95 | 0.0679 | 89.28% |
| + clip 0.93 / 0.93 | 0.0711 | 89.12% |
| + per-group clip search (4 candidates, 1.0-0.7, least squared error; experiment, removed) | 0.0689 | 89.13% |

- Weight groups of 64 hold at -6.7%.
- Clipping pays only on the Hadamard inputs (attn_output, ffn_down): CH 0.95-0.97 is -1.7%, free (stored per tensor,
  `ruquant setclip MODEL C CH`). Below 0.95 it turns.
- A per-group choice of the clip (the activation quantizer trying 4 clips per group of 64 and keeping the least
  squared error) moved 0.4%: 64 rotated values are near Gaussian and their max (~2.4 sigma) is already close to the
  MSE-optimal clip for 16 levels. Not worth its cost in the decode-time quantizer; reverted.
- Already measured before (REPORT-2026-11-02): learned weight scales (`--refit`) are within noise. The cheap GPTQ
  knobs are used up; what is left on the weight side is training (block-wise reconstruction).

## The 27B redone (3090)

`models/qwen3.8-27b-ost44-v2-wg64-rb4-gptq.gguf`: the same OST44 rotation and smoothing and `--rot-blocks 4` as
before; Hessians recollected from `models/qwen3.8-27b-ost44-rtn.gguf` with the fixed collector and 64 windows
(`ruquant hessian ... --seqs 64`, 6270 s, 94 GB in `models/hess-ost44-v2`), then `convert --wgroups 64` (1537 s).
16 chunks against UD-Q8_K_XL:

| 27B | KLD | top-1 | PPL / base |
|---|---|---|---|
| unsloth Q4_0 | 0.026 | | |
| before (ost44-wg-rb4: per-256 weight groups, 16 windows, Hessian bug) | 0.0495 | 90.63% | 1.0334 |
| **v2: fixed Hessians, 64 windows, weight groups of 64** | **0.0402** | **91.65%** | 1.0273 |
| v2 at clip 0.95 / 0.95 | 0.0432 | 91.56% | 1.0261 |
| v2 at clip 1.0 / 0.95 (Hadamard inputs only) | 0.0438 | 91.55% | 1.0358 |

On 64 chunks (`accuracy/ref/ud-q8_k_xl.kld` rebuilt: UD-Q8_K_XL, 44 of 64 layers on the 3090, 4 min):

| 27B, 64 chunks | KLD | top-1 | PPL / base |
|---|---|---|---|
| before (ost44-wg-rb4) | 0.0513 | 90.54% | 1.0337 |
| **v2** | **0.0428** | **91.57%** | 1.0279 |
| unsloth Q4_0 (8-bit activations) | 0.0259 | 93.34% | 1.0220 |
| unsloth UD-Q4_K_XL | 0.0078 | 96.58% | 0.9989 |

- -17% KLD on 64 chunks (-19% on 16), +1.0 point top-1, no runtime cost: same file size (17.18 GB), same kernels (the per-64 multipliers
  were timed on the MI50 before access ended: identical).
- Unlike the 4B, the 27B wants no activation clip: both clipped variants are worse by more than the 16-chunk noise.
  OST44's smoothing already narrows the ranges the clip would cut. The model is left at clip 1.0 (unset).
- Not run (box 1 is off limits for now): the MI50 decode / pp512 check of this file and the drafter acceptance. The
  format and kernels are unchanged from the timed per-64 build, so the speed should be identical.

## The v2 27B on box 2 (2x MI50 16 GB), 2026-11-04

Box 2: two MI50 16 GB (gfx906), i7-5820K (4 threads in the container), 12 GiB RAM, ROCm with RCCL. **No peer access
between the cards** (`amd-smi topology`: access, DMA and atomics disabled), so mx turns its custom TP all-reduce off
and every all-reduce goes through RCCL via host memory. mx-llama.cpp c4d12e2 + `patches/mx-llama.cpp-ruquant.patch`
built in `~/rq/mx-v2` (`rig/build_mx.sh`, -j4; the box has no `patch`, so `git apply` first). W4A4 MUL_MAT tests
56/56 on both cards with weight groups of 64 (and per row / 256 on GPU 0). Model
`qwen3.8-27b-ost44-v2-wg64-rb4-gptq.gguf`, drafter `dflash2-ost44-h4-l4-c8-g64-rb4.gguf` on GPU 0, `-fa 1`,
`HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_META_TOKEN_GRAPH=0 GGML_CUDA_DISABLE_GRAPHS=1`; llama-bench per-repetition
samples, warm-up outliers left out.

| box 2 | t/s | box 1 TP = 2 (P2P, earlier builds) |
|---|---|---|
| TP = 2 pp512 | 318 (305-328) | 349 (before the GEMM) |
| TP = 2 pp2048 (ubatch 512 or 1024) | 413 (405-414) | 713 |
| TP = 2 tg128 (plain decode) | 31.3 (31.0-31.5) | 39.7 |
| TP = 2 T = 8 forward (a verify step) | 180 (44.4 ms) | 237 (33.8 ms) |
| **TP = 2 + DFlash2, draft 7, 30 prompts** | **70.0** (accept 4.26, 60.9 ms/step; math 93.1, code 74.2, chat 53.8) | 94.4 (4.32, 45.8 ms) |
| layer split, 2 GPUs, pp2048 (ubatch 512 / 256) | 730 / up to 775 | |
| layer split tg128 | 28.7 | |

- Without peer writes, TP's all-reduces cost box 2 a third of the verify step (44 vs 34 ms) and half of TP prefill.
  Layer split needs no all-reduce: it prefills 1.8x faster than TP here (730 vs 413) and decodes nearly as fast.
- Acceptance 4.26 matches box 1's 4.29-4.30 (single run, +-3%): the -17% KLD does not show in drafting, as before
  (the drafter is the limit).
- The box owner's baseline (Q8_0, layer split): pp512 272, tg128 22, DSpark 33.8 t/s. Fine-grained PCIe memory makes
  no difference to decode here (32.0 without).
- llama-server's prompt cache defaults to 8 GiB of host RAM and got the server OOM-killed in the 12 GiB container
  mid-run: pass `--cache-ram 0` on box 2.
- Uploads to box 2 run at ~2.5 MB/s whatever the stream count (Hugging Face downloads on the box are no faster), and
  long transfers drop connections: 17 GB took ~2 h with a chunk-hash repair pass.
