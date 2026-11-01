# ruquant

Quantization research on ggml kernels. Rust host, ggml backends for CPU / CUDA / Vulkan, and
our own CUDA kernels running as nodes inside ggml-cuda's captured CUDA graph.

## Use it

```sh
nix develop
./build-ggml.sh          # once: builds third_party/llama.cpp into build-ggml/ (~10 min)
cargo build --release    # bindgen + our kernels, links build-ggml/bin
./run.sh
```

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
| 16 | 4.83 (MMQ) | 1.04 | LDS, 5.5 cycles |
| 32 | 4.14 (MMQ) | 1.05 | LDS, 5.1 cycles |
| 64 | 3.78 (MMQ) | 1.03 | LDS, 4.9 cycles |

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
| 16 | 52.36 ms | 25.66 ms | 2.04× |
| 32 | 96.93 ms | 36.28 ms | 2.67× |

- With the card's acceptance at block 8: 149-199 tok/s today, 191-255 tok/s W4A4.
- Going to block 16 pays off once it accepts 1.20× more tokens per step than block 8 does.
  Today's path needs 1.91×. That's where the freed compute shows up: block 16 or a
  DDTree-style tree over DFlash2's top-16 candidates becomes affordable.
- Across bandwidth efficiency 0.65-0.8, latency 1-1.5 µs and scalar cache 8-16 B/clk, the
  block-16 break-even stays at 1.14-1.26×. Long context changes it: attention runs fp16 on
  both paths and grows with T × context, so break-even is 1.37× at 32K and 1.66× at 128K.
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

Measure on the MI50 before trusting the rest:
1. `v_dot8_i32_i4` issue rate. Everything assumes one per 4 cycles per wave, like any VALU op.
2. Achievable HBM bandwidth and latency under load (`--bw-eff`, `--latency`).
3. Scalar cache bandwidth. Assumed 16 B/clk per 4 CUs. At half that, T ≥ 32 turns
   scalar-cache-bound (the 27B block-32 step gets 1.35× slower); then use R=2 there, which
   halves it per dot8.
4. L2 atomic throughput for split-K (modelled as plain L2 bandwidth).

## Layout

```
src/sys.rs          bindgen output + device lookups
src/bench.rs        f32 matmul on every device, checked against CPU
src/custom_op.rs    our kernel as a ggml node, timed
src/kernels.rs      the hook: Rust launcher -> rq_scale_launch on ggml-cuda's stream
src/quant.rs        Q4_0 matmul on every device, checked against CPU
kernels/*.cu        our CUDA kernels (nvcc -arch=native, via build.rs)
build.rs            bindgen, kernels, link, kernels/compile_flags.txt for clangd
build-ggml.sh       applies the patch if missing, builds third_party/llama.cpp
gfx906/             MI50 cost model: gfx906 asm, instruction counts, step-time model
patches/            our changes to third_party/llama.cpp
third_party/        llama.cpp as vendored by llama-cpp-sys-2 0.1.160 (ggml 0.26.0), patched
run.sh              dev runner (LD_LIBRARY_PATH)
```

## Dev build = this GPU only

`build-ggml.sh` passes `GGML_NATIVE=ON` and `CMAKE_CUDA_ARCHITECTURES=native`, and `build.rs`
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
