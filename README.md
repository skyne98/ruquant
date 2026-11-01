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

ROCm/HIP: the toolchain works in a devShell (`hipcc` 7.2.4), but there is no AMD GPU here to
run it on. Removed until there is.
