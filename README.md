# ruquant

Quantization research on ggml kernels. Rust host on our llama.cpp fork: W4A4 (int4 weights x int4
activations) as ggml tensor types with CPU and CUDA kernels, a converter that turns a GGUF into a
rotated W4A4 GGUF, and our own CUDA kernels running as nodes inside ggml-cuda's captured CUDA graph.

## Use it

```sh
nix develop
./build-llama.sh         # once: upstream llama.cpp at a pinned commit + patches/llama.cpp-ruquant.patch,
                         # built into build-llama/ (CUDA, HIP or CPU, whichever the machine has)
cargo build --release    # bindgen + our kernels, links build-llama/build/bin
./run.sh                 # device demo
./run.sh convert IN.gguf OUT.gguf     # a Qwen3.5-family GGUF into a rotated W4A4 GGUF
build-llama/build/bin/llama-perplexity -m OUT.gguf ...   # and every other llama.cpp tool
```

After editing the fork in `build-llama/src`, `./build-llama.sh --export` writes its diff back to
the patch.

This is a git repo and the dev shell is a flake: nix only sees files git tracks, so `git add`
a new file before `nix develop` can use it.

Inside `nix develop`, run through `./run.sh`, not `cargo run`. `libggml-cuda.so`'s RUNPATH
contains the toolkit's `lib/stubs`, and the devShell overrides `LD_LIBRARY_PATH`, so the stub
`libcuda.so.1` wins ("CUDA driver is a stub library"). `run.sh` puts `/run/opengl-driver/lib`
first. Outside the devShell the host `LD_LIBRARY_PATH` already does that.

## What it runs

```
256x256x256 f32 matmul, best of 3
  CPU          1.390 ms     24.1 GFLOP/s  reference
  CUDA0        0.010 ms   3362.5 GFLOP/s  139.30x cpu  max_err 0.00e0  ok
  Vulkan0      0.042 ms    791.2 GFLOP/s  32.78x cpu  max_err 0.00e0  ok

custom kernel in-graph on CUDA0: best 8.3 us  median 8.5 us  ok

Q4_0 weights × f32 activations, K=4096 N=1024 M=1 (best of 5)
  CPU         0.078 ms   107.6 GFLOP/s    30.2 GB/s  reference
  CUDA0       0.010 ms   863.2 GFLOP/s   242.8 GB/s  rel_err 4.0e-3  ok
  Vulkan0     0.023 ms   359.0 GFLOP/s   101.0 GB/s  rel_err 3.2e-3  ok

Q4_0 weights × f32 activations, K=4096 N=1024 M=64 (best of 5)
  CPU         5.013 ms   107.1 GFLOP/s     0.5 GB/s  reference
  CUDA0       0.028 ms  19302.9 GFLOP/s    84.8 GB/s  rel_err 1.2e-2  ok
  Vulkan0     0.045 ms  11855.4 GFLOP/s    52.1 GB/s  rel_err 7.9e-4  ok
```

RTX 3090 + Ryzen 9 9950X3D, 2026-11-01.

The Q4_0 rows are 4-bit *weights* only. ggml-cuda unpacks them to int8 (dp4a at M=1, int8
`mma` at M>1); ggml-vulkan dequantises to f32. Nothing in ggml uses int4×int4 hardware.

## W4A4 in llama.cpp

Four ggml types (`patches/llama.cpp-ruquant.patch`): weights `W4A4`, and `W4A4H`, which applies an
online Hadamard to its input first; activations `A4` / `A4H`, which exist only as the CPU backend's
vec_dot types. A row of K int4 codes is K/256 blocks of a 16-byte header (the row's fp32 scale and
code sum, repeated in every block) plus 128 bytes of codes: 4.5 bits per weight, codes 16-byte
aligned. A matmul does the spec of `../mi50emu`: per-token asymmetric int4 activations over
[min(x,0), max(x,0)], int32 accumulation, `y = (fp32(acc + (8-z)*colsum) * s_a) * s_w`, except that
the output stays fp32 instead of rounding to fp16.

- CPU: the reference (ggml-quants.c, ggml-cpu/quants.c).
- CUDA sm_80+: online Hadamard, activation quantizer and mi50emu's `mma.m16n8k64 s4` GEMM reading
  the codes in place (ggml-cuda/rq-w4a4.cu). `test-backend-ops -o MUL_MAT -p type_a=w4a4` checks it
  against the CPU at zero tolerance: 62 shapes, bit-identical, Hadamard included.
- HIP (gfx906): a skinny `v_dot8_i32_i4` GEMM (ggml-cuda/rq-w4a4-skinny.cuh, from gfx906/w4a4_skinny.h).
  - The weight is repacked in place on its first use; HIP graphs stay off until then.
  - The epilogue is fused into the dot8 kernel, with a last-workgroup split-K fixup.
  - The online Hadamard has an LDS-tiled Paley stage.
  - Tokens go in chunks of 64 (any number of tokens works); weights need rows a multiple of 64.
  - Bit-exact with the CPU on an MI50 (58 shapes). `RQ_SKINNY=1` runs the same path on CUDA, dot8 emulated,
    to check the indexing.
- Activation clip: a weight tensor carries its own in its block headers (`pad[1]`), which `ruquant setclip`
  writes; `RQ_ACT_CLIP` / `RQ_ACT_CLIP_H` (W4A4 / W4A4H inputs) override it; without either it is 1.

The Hadamard is H_p kron H_m / sqrt(n), H_p a Paley matrix (p - 1 prime, = 3 mod 4), m a power of 2,
in a fixed operation order on every backend: 5120 = 20 x 256, 6144 = 12 x 512, 17408 = 68 x 256.

`ruquant convert` (src/convert.rs, ggml's own dequantizers, quantizer and Hadamard) folds a rotation
R1 = diag(random signs) H into the residual stream: the RMSNorm scales go into the weights that read
it, `attn_output`, `ssm_out` and `ffn_down` become W4A4H (their inputs can't be rotated offline: the
attention gate and DeltaNet's z-gate sit in front), the norms become ones, the embedding and lm_head
keep their types. Qwen3.8-27B from unsloth's UD-Q8_K_XL: 31.5 GB in, 18.2 GB out, 97 s.

The converter's options and the other subcommands:
- `--types-from Q4_0.gguf` takes unsloth's type map: its 4-bit linears become ours, everything else keeps
  that file's type.
- `--set NAME=TYPE` overrides the map, e.g. `--set ssm_out=q8_0 --set output.weight=q8_0`.
- `--hessians DIR` (from `ruquant hessian`, optionally `--act-clip C --act-clip-h CH` to fit against the
  quantized activations) turns rounding into GPTQ.
- `--ost DIR` folds R1 and the smoothing scales from `accuracy/ostquant_qwen.py`.
- `ruquant drafter` converts a DFlash / DSpark drafter for a rotated target: R1 folded into its `fc`, and its
  own unrotated embedding and lm_head (`--head w4a4h`, `--linears w4a4h`).
- `ruquant head` rebuilds a converted model's lm_head in another type.
- `ruquant setclip` writes the activation clips into a model.

Tonight's state, results and how to reproduce them: [REPORT-2026-11-02.md](REPORT-2026-11-02.md).

## Our kernels inside ggml's graph

[`patches/ggml-cuda-custom1-hook.patch`](patches/ggml-cuda-custom1-hook.patch) (3 hunks in
`ggml-cuda.cu`, marked `RUQUANT patch`) adds `ggml_cuda_set_custom1_hook()`. Once
`src/kernels.rs` registers a launcher, ggml-cuda claims `GGML_OP_MAP_CUSTOM1` and launches it
on its own stream from inside graph capture, so our kernel replays with the rest of the graph.

Measured on the same graph (`a → map_custom1 → ggml_scale`), C++ harness, 50 iters:

| route | how | best |
|---|---|---|
| A, patch (used) | launch inside ggml-cuda's capture | 5.7 µs |
| B, no patch | own `ggml_backend_i` + `ggml_backend_sched` | 9.2 µs |

B pays an event and a separate launch at every backend boundary, because ggml-cuda only
captures its own split of the graph.

Limits of the hook as it stands:
- One global launcher. Every `map_custom1` node on CUDA runs it, whatever its CPU function is.
  A second kernel needs dispatch on the node (e.g. the `userdata` op param).
- `supports_op` doesn't check types or layout; `rq_scale` assumes contiguous f32.
- The Rust path measures 8.3 µs for the graph the C++ harness ran in 5.7 µs. Not explained yet.

## MI50 cost model (`gfx906/`)

The plan: W4A4 with `v_dot8_i32_i4` on the MI50 frees enough compute to verify wide
speculative-decoding trees for about the price of one token. The 3090 can't measure that (its
int4 tensor cores never run out of compute), so this compiles the real kernels for gfx906,
counts their instructions, and turns the counts into a verify-step time per width T.

```sh
nix develop .#rocm -c gfx906/build.sh      # gfx906 asm: llama.cpp q4_0 kernels + ours
python3 gfx906/model.py --kernels          # instruction mix + Qwen3.8 / DFlash2 step times
nix develop -c nvcc -O2 -arch=sm_86 -std=c++17 gfx906/test_skinny.cu -o gfx906/build/test_skinny
LD_LIBRARY_PATH=/run/opengl-driver/lib gfx906/build/test_skinny   # same kernels on the 3090
```

- `w4a4_skinny.h`: W4A4 GEMM for T = 1..64. The inner loop is weight loads plus dot8:
  activations are wave-uniform and feed dot8 from SGPRs. K is split across the waves of a
  workgroup (reduced in LDS) and across workgroups (int32 atomics, still exact). The same
  source builds for CUDA with dot8 emulated. On the 3090 it matches a CPU reference exactly,
  and the epilogue matches the spec bit for bit, over 36 cases.
- `isa.py`: instruction mix per loop and resource use, read from the compiler's asm.
- `model.py`: roofline over VALU / SALU / LDS / L1 / scalar cache / HBM, plus Little's law for
  memory latency. It times a whole speculative step: drafter forward, lm_head, target forward,
  lm_head again. Shapes come from the Hugging Face configs of Qwen3.8-27B, Qwen3.8-Flash-Next
  and their DFlash2 drafters.

Steady-state K loop, VALU cycles per 512 multiply-adds (pure dot8 = 1.0):

| T | today (q4_0 × q8_1) | W4A4 dot8 | today's real limit |
|---|---|---|---|
| 1 | 12.5 (MMVQ) | 1.56 | HBM |
| 8 | 5.69 (MMVQ) | 1.08 | VALU |
| 16 | 4.66 (MMQ) | 1.04 | LDS, 5.4 cycles |
| 32 | 4.04 (MMQ) | 1.05 | LDS, 5.0 cycles |
| 64 | 3.71 (MMQ) | 1.03 | LDS, 4.8 cycles |

DFlash2 drafts a whole block in one pass, and the target verifies one path through it. So a
block of B tokens means the target runs at T = B. The model cards publish block 8 for 27B
(4.10-5.46 tokens accepted per step) and block 4 for Flash-Next (2.83-3.26; the card says
the current sparse-attention code limits it).

**Qwen3.8-27B + DFlash2**, one 32 GB MI50 (15.7 GB of q4_0 weights incl. drafter and q6_K
lm_head), 8K context, 1.725 GHz, 0.8 × 1024 GB/s, 1 µs latency:

| block T | today step | W4A4 step | speedup |
|---|---|---|---|
| 1 (no speculation) | 19.05 ms | 17.26 ms | 1.10× |
| 8 (model card) | 27.47 ms | 21.44 ms | 1.28× |
| 16 | 51.15 ms | 25.55 ms | 2.00× |
| 32 | 94.91 ms | 36.10 ms | 2.63× |

- With the card's acceptance at block 8: 149-199 tok/s today, 191-255 tok/s W4A4.
- Going to block 16 pays off once it accepts 1.19× more tokens per step than block 8 does.
  Today's path needs 1.86×. That's where the freed compute shows up: block 16 or a
  DDTree-style tree over DFlash2's top-16 candidates becomes affordable.
- Across bandwidth efficiency 0.65-0.8, latency 1-1.5 µs and scalar cache 8-16 B/clk, the
  block-16 break-even stays at 1.14-1.25×. Long context changes it: attention runs fp16 on
  both paths and grows with T × context, so break-even is 1.36× at 32K and 1.66× at 128K.
  At 128K, attention is most of the step.

**Qwen3.8-Flash-Next + DFlash2**, four MI50s (70 GB of q4_0 weights; n-gram embedding table
assumed in host memory):

| | layer split (llama.cpp default) | tensor split (15 µs all-reduce) |
|---|---|---|
| W4A4 speedup at block 4 | 1.09× | 1.06× |
| tok/s at block 4, today → W4A4 | 268-309 → 292-336 | 651-750 → 687-792 |
| acceptance needed for block 8 over block 4, today / W4A4 | 1.63× / 1.57× | 1.40× / 1.35× |

W4A4 barely matters here. Each extra token in a verify step pulls in about 10 more experts
out of 512, so a wider block costs expert weight bandwidth, not compute. With independent
routing, T=8 reads about 7.5× the expert weights of T=1. Even with half the experts shared
between draft tokens (`--expert-reuse 0.5`), block 8 needs 1.40× the acceptance.

Read the comparisons, not the milliseconds. The model assumes perfect overlap and no bank
conflicts. It leaves out launch overhead, norms, convs, router, hyper-connections and the
drafter's selector. The absolute tok/s are far above what llama.cpp gets on an MI50 today.

On Qwen3.8, W4A4 needs more online Hadamards than OSTQuant's recipe. The attention output
gate and the DeltaNet z-gate multiply elementwise in front of o_proj / out_proj, and a
rotation can't be folded through that. The model charges for all three Hadamards, these two
plus down_proj (about 0.1 ms per 27B step at block 8).

What the asm taught the kernel:
- Per-token activation addressing cost 2 SALU per token per step. Step-major activations
  (`rq_act_index`) make each step one contiguous scalar load.
- The scheduler hoists all of a step's scalar loads and spills SGPRs through
  `v_writelane`/`v_readlane`. One `sched_barrier` per token block removes the spills.
- Split-K partials through L2 cost more than the weights at T=64. Reducing inside the
  workgroup through LDS first is what keeps T=16-32 near the floor.

### Measured on gfx906

Two borrowed boxes, through `rig/`: four MI50 32 GB (ROCm 5.7) and two Radeon Pro VII 16 GB
(the same chip, ROCm 10.1). The JSON behind these numbers is in `rig/measured/`.

The kernel test passes and mi50emu's parity runner is bit-exact with the spec on all six
cards, under both toolchains.

| | MI50 box | Pro VII box | model's guess |
|---|---|---|---|
| dot8 issue rate | 0.99 | 0.97 | 1.0 |
| clock under dot8 load | 1606 MHz (178 W cap) | 1568 MHz | 1725 MHz |
| HBM read | 826 GB/s | 883 GB/s | 819 GB/s |
| loaded latency | 408 ns | 424 ns | 1000 ns |
| kernel launch, back to back | 1.2 µs | 1.4 µs | not modelled |
| all-reduce, 8 tokens × 5120 f16 | 41 µs (2 GPUs), 66 µs (4) | 101 µs (2, no P2P) | 15 µs |

Per matmul on the same card (Pro VII box), ggml's q4_0 against `w4a4_skinny`. Our vendored
ggml and the box owner's newer build agree within ±30%.

| T | W4A4 faster by | why |
|---|---|---|
| 1 | 1.3-2.0× | ggml's MMVQ reaches about half of HBM bandwidth, ours 70-83% |
| 8 | 3.3-4.8× | ggml's T=8 costs 2.5-4× its T=1; ours costs about the same as T=1 |
| 16-64 | 3-13× | MMQ |

27B + DFlash2 on one card with those matmuls (`model.py --measured rig/measured/provii-x2-rocm101`
plus the box's calibrated flags):

| block T | today step | W4A4 step | speedup |
|---|---|---|---|
| 1 (no speculation) | 32.4 ms | 21.9 ms | 1.48× |
| 8 (model card) | 95.3 ms | 37.7 ms | 2.53× |
| 16 | 108.3 ms | 38.0 ms | 2.85× |
| 32 | 175.7 ms | 63.0 ms | 2.79× |

- With the card's acceptance at block 8, that's 43-57 tok/s today and 109-145 tok/s W4A4.
- The q6_K lm_head runs on ggml's kernel in both columns. At block 8 it is a third of the
  W4A4 step, because ggml's T=8 q6_K takes 3.8× its T=1. It is the next kernel to write.
- Tensor parallel on the four-GPU box: every P2P transfer crosses the CPU on PCIe gen3,
  5.2 GB/s per pair and 2.6 GB/s per link with all four busy. RCCL was the fastest of four
  all-reduces tried. The 128 all-reduces of a 27B block-8 step cost about 8.5 ms on four
  GPUs, half the step, so four GPUs are 1.7× faster than one.

What the MI50 taught the kernel:
- At T ≥ 24 a token block is only 32 dot8s, too few to hide a scalar load issued right
  before them. Loading the next block's activations one block ahead made T=32 8-23% faster.
- One LDS slice per wave meant 24 KB per workgroup at T=32, which capped occupancy at 2
  waves per SIMD. One shared buffer that the waves `ds_add` into made T=32 1.25-1.5× faster
  and T=64 up to 1.76× faster.
- T=32 still reaches only about half the dot8 peak. `rig/probe_t32.cu` removes one input at a
  time, and the loss is spread across the DRAM weight stream, scalar-cache refills and
  occupancy. No single fix is left.

## Layout

```
src/sys.rs          bindgen output + device lookups
src/bench.rs        f32 matmul on every device, checked against CPU
src/custom_op.rs    our kernel as a ggml node, timed
src/kernels.rs      the hook: Rust launcher -> rq_scale_launch on ggml-cuda's stream
src/quant.rs        Q4_0 matmul on every device, checked against CPU
src/convert.rs      ruquant convert / drafter / head / setclip: GGUF -> rotated W4A4 GGUF and friends
src/hessian.rs      ruquant hessian: GPTQ Hessians from libllama's forward pass
src/gptq.rs         GPTQ onto the W4A4 grid, W4A4 block packing
src/gguf.rs         GGUF v3 reader / writer
kernels/*.cu        our CUDA kernels (nvcc -arch=native, via build.rs, only where CUDA is)
build.rs            bindgen, kernels, link, kernels/compile_flags.txt for clangd
build-llama.sh      fetches upstream llama.cpp at the pinned commit, applies our patch, builds it
accuracy/           W4A4 accuracy in simulation (PyTorch) and the llama.cpp reference files
gfx906/             MI50 cost model: gfx906 asm, instruction counts, step-time model
rig/                day-one kit for the MI50 box: benches, parity, calibration (rig/README.md)
patches/            our changes to llama.cpp (llama.cpp-ruquant.patch) and to mx-llama.cpp, the MI50 fork with
                    DSpark / DFlash2 speculative decoding (mx-llama.cpp-ruquant.patch, rig/build_mx.sh)
third_party/        the old vendored llama.cpp (ggml 0.26.0), still used by rig/build_ggml.sh
run.sh              dev runner (LD_LIBRARY_PATH)
```

## Dev build = this GPU only

`build-llama.sh` passes `GGML_NATIVE=ON` and `CMAKE_CUDA_ARCHITECTURES=native`, and `build.rs`
compiles our kernels with `nvcc -arch=native`: one arch (`sm_86`) instead of ggml's full list.

## NixOS gotchas already handled

| symptom | cause | fix |
|---|---|---|
| `Unable to find libclang` | not on default path | `LIBCLANG_PATH` = libclang's `lib` output |
| bindgen can't find libc headers | no glibc headers on the default path | `BINDGEN_EXTRA_CLANG_ARGS=-isystem glibc.dev/include` |
| `glslc` not found by ggml-vulkan | binaries live in shaderc's `bin` output | `vulkanSdk` symlinkJoin in `flake.nix` |
| `sccache: cannot find binary path` | ggml's `GGML_CCACHE` finds sccache and wraps nvcc | `GGML_CCACHE=OFF` |
| `CUDA driver is a stub library` | stub `libcuda.so.1` on libggml-cuda's RUNPATH | `run.sh` puts `/run/opengl-driver/lib` first |

## Related

`../mi50emu`: a W4A4 GEMM on the 3090's int4 tensor cores (`mma.s4`), bit-exact with the
MI50's `v_dot8_i32_i4` pipeline. Separate project, not wired into ruquant.

## Parked

ROCm/HIP at runtime: there is no AMD GPU here. `nix develop .#rocm` (`hipcc` 7.2.4) is
compile-only, for `gfx906/`.
