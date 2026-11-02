# rig/: day one on the 2× MI50 box

Everything to run on the MI50s, in one tarball, without nix or numpy on the box. It checks
the dot8 kernel and mi50emu's bit-parity on real gfx906, measures every hardware number
`gfx906/model.py` guesses, and times both matmul paths on the Qwen3.8 / DFlash2 shapes.
Back here, one command turns the results into model flags plus measured-vs-predicted.

## Before you go (here)

```sh
rig/pack.sh                        # -> dist/ruquant-rig.tar.gz (~24 MB)
scp dist/ruquant-rig.tar.gz box:
```

The box is reached through a Cloudflare quick tunnel whose hostname changes when its owner
restarts it. `rig/box.sh HOST ...` wraps the cloudflared ProxyCommand (fetching cloudflared
from nixpkgs if it isn't installed): `push` sends and unpacks the tarball, `start CMD` runs a
command under nohup so a dropped tunnel doesn't kill it, `log` tails it, `pull` brings the
newest results tarball back.

```sh
rig/box.sh HOST push
rig/box.sh HOST start 'rig/build.sh && rig/run_all.sh'
rig/box.sh HOST log
rig/box.sh HOST pull               # -> rig/results/<dir>
```

## On the box

```sh
tar xzf ruquant-rig.tar.gz && cd ruquant
rig/probe.sh | less                # ROCm version, both gfx906, rocBLAS gfx906 files, topology, who else is on the GPUs
rig/build.sh                       # hipcc only: benches, kernel test, mi50emu parity runner
rig/build_ggml.sh                  # optional, longer: needs cmake + hipBLAS/rocBLAS; today's path + llama-bench
rig/run_all.sh                     # -> rig/results/<host>-<date>.tar.gz
```

With a Qwen3.8-27B GGUF on the box, `MODEL=/path/model.gguf rig/run_all.sh` also runs
llama-bench, on one GPU and with a layer split across both.

Nothing installs anything or needs root. It's someone else's machine: `env.txt` records
`rocm-smi --showpids`, so check nobody else is on the GPUs, or the numbers are noise.

## W4A4 end to end (mx-llama.cpp)

```sh
rig/build_mx.sh                    # mx-llama.cpp + patches/mx-llama.cpp-ruquant.patch (ROCm 5.7 works)
rig/spec_bench.sh MODEL [DRAFT] [N_MAX]    # llama-server decode t/s, acceptance, ms/step (SPEC=draft-dflash ...)
rig/score_box.sh MODEL [label]     # KLD / top-1 vs ~/ref/ud-q8_k_xl-16.kld (copy it from accuracy/ref/)
rig/final_box.sh MODEL DRAFT       # both, plus llama-bench forward times, with the model's own clips
rig/upload.sh FILE HOST DEST       # big files through the tunnel: O_DIRECT reads, resumes after a drop, checks size
ggml_bench --types w4a4,q4_0 --lm q6_K,q8_0 --widths 1,6,8 | python3 rig/bench_summary.py
```

Use `GGML_CUDA_DISABLE_GRAPHS=1` on ROCm 5.7: HIP graphs there are slower and less stable.

## Bring it back

```sh
scp box:ruquant/rig/results/<dir>.tar.gz rig/results/ && tar xzf rig/results/<dir>.tar.gz -C rig/results
python3 rig/calibrate.py rig/results/<dir>
python3 ../mi50emu/tools/parity.py compare rig/results/<dir>/parity/m333_n4096_k4096-gpu0   # and the rest
```

## What each step answers

| step | question | pass / feeds |
|---|---|---|
| `test_skinny` (both GPUs) | does the dot8 kernel get the integers right with real `v_dot8` and 64-lane waves? | must print `ALL OK` |
| parity (`w4a4_ref`, both GPUs) | does the MI50 produce the spec's bits? | `parity.py compare` says `PARITY OK` |
| `bench_valu` | is dot8 full rate? clock under load? launch cost? | `--dot8-rate`, `--clock` |
| `bench_mem` | achievable HBM bandwidth, loaded latency, cache latencies | `--bw-eff`, `--latency` |
| `bench_atomic` | real cost of split-K partial sums | `--atomic-gbs` |
| `bench_p2p` | GPU-to-GPU hand-off latency | `--handoff-us` |
| `bench_tp` (all NGPU) | peer bandwidth per pair and with every link busy; all-reduce of verify-sized activations over 2 and N GPUs: RCCL, copy engines, P2P reads, P2P writes | `--allreduce-us`, `--allreduce-gbs` |
| `bench_skinny` | our kernel on every model shape and split; R=1 vs R=2 at T ≥ 32 | measured vs predicted, w4a4 |
| `ggml_bench` | ggml's q4_0 MMVQ/MMQ and q6_K lm_head on the same shapes | measured vs predicted, today |
| `llama-bench` | real end-to-end decode tok/s (`MODEL=... LLAMA_BENCH=...`, a build already on the box) | sanity check on the absolute numbers |

Kernel timings are batched (10 launches between one event pair), because on ROCm an event
pair around a single small launch adds about 10 µs. `ggml_bench` times a graph of several
mul_mats and divides, for the same reason.

The R=1 vs R=2 pair was meant as the scalar-cache test. On the MI50, R=2 lost at T ≥ 32
(more VGPRs, less occupancy). `probe_t32.cu` is the finer tool: it removes one input path at
a time from the T=32 loop.

With both `skinny-gpu0.json` and `ggml-gpu0.json` from the same card, the model can use the
measured matmuls instead of its roofline:

```sh
python3 gfx906/model.py <flags from calibrate.py> --measured rig/results/<dir>
```

## Checked before packing

- Every harness ran on the 3090 through `gfx906/gpu.h`, the same source with a CUDA spelling.
  The kernel test passed and every JSON parsed into `calibrate.py`.
- Everything compiled for gfx906 from the unpacked tarball. The dot8 and dot4 benchmark loops
  are 64 `v_dot8` / `v_dot4` per iteration and nothing else. The pointer chase compiles to
  vector loads, the path weights take.
- The RUQUANT hook in `ggml-cuda.cu` compiles as HIP, both the host and device passes.
- Not checked, because only the box can: anything at runtime on gfx906.

## If something breaks

- **No gfx906 in `rocminfo`, or "no binary for GPU":** the container's ROCm runtime lacks
  gfx906. `probe.sh` shows the version.
- **`build_ggml.sh` fails on hipBLAS/rocBLAS:** skip it. The kernel test, parity and benches
  need only hipcc and the HIP runtime. ggml's q4_0 matmuls use its own kernels, so a rocBLAS
  without gfx906 kernels breaks only fp16/fp32 matmuls, which `ggml_bench` doesn't run.
- **P2P is off in the container:** `bench_p2p` stages through pinned host memory instead and
  reports `"p2p": false`; `bench_tp` skips the P2P read/write all-reduces.
- **No `rocm-smi` (ROCm 7+):** `probe.sh` and `run_all.sh` fall back to `amd-smi`.
- **One GPU only, or one is busy:** `NGPU=1 rig/run_all.sh`.
- **A step crashed:** `run.log` lists exit codes; each step's stderr is next to its output.
