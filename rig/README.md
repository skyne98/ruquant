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
rig/spec_bench.sh MODEL [DRAFT] [N_MAX]    # llama-server decode t/s, acceptance, ms/step (SPEC=draft-dflash ...);
                                           # TEXT=file saves the generations, to check a build changes no output
rig/prof_spec.sh MODEL DRAFT [N_MAX] [OUT] # the same under rocprof: kernel ms per verify step (PROF=--hip-trace adds host API calls)
python3 rig/phases.py OUT/results.csv      # that trace split into head / host gap / drafter / target phases
python3 rig/add_mtp.py MODEL SRC UNROT OUT # a rotated model's MTP layer made usable (--spec-type draft-mtp), see the script
python3 rig/acc_bench.py MODEL DRAFT 7     # draft acceptance per category on rig/acc_prompts.json (30 prompts)
rig/score_box.sh MODEL [label]     # KLD / top-1 vs ~/ref/ud-q8_k_xl-16.kld (copy it from accuracy/ref/)
rig/final_box.sh MODEL DRAFT       # both, plus llama-bench forward times, with the model's own clips
rig/upload.sh FILE HOST DEST       # big files through the tunnel: O_DIRECT reads, resumes after a drop, checks size
rig/final_mi50.sh MODEL DRAFTER    # the final table (RESULTS.md): 1 GPU / TP2 / TP4 prefill, decode, verify, DFlash2, KLD
rig/same_tokens.sh MODEL GPUS EXTRA  # hash of a greedy completion: a change must keep it byte for byte
python3 rig/tp_breakdown.py ONE.csv TP2.csv TP4.csv   # prefill kernel time per category, 1 GPU vs TP (rocprof traces)
python3 rig/draft_breakdown.py results.csv            # a one-GPU DFlash2 trace split into target / draft / inject graphs
rig/therock/setup.sh                # on the box: AMD TheRock ROCm 7.12 (gfx90X nightly, RCCL with gfx906) in ~/therock-7.12 + build-r712
rig/therock/build_local.sh MX HOST  # the same build on a workstation in an Ubuntu 22.04 container (85 s on 32 threads), shipped to the box
rig/rocm72/setup.sh                 # AMD's packaged ROCm 7.2.4 in user space + build-r72 (no RCCL: its rccl has no gfx906 code)
rig/rocm101/build_local.sh MX HOST  # ROCm 10.1 (mixa3607 gfx906 apt) in an Ubuntu 24.04 container, private glibc 2.39 on the box; 1-2% slower than 7.12
rig/prof.sh LABEL [VAR=v ...] -- CMD # on the box: one rocprofv3 profile into ~/profiles/<time>-LABEL with meta.txt (never overwritten)
rig/prof_pull.sh HOST               # finished box profiles into profiles/box1 (one .tar.gz each); profiles/INDEX.md says what each is
rig/prof_summary.py PROFILE [RE]    # kernel-trace and counter medians per kernel and grid, from a profile dir or .tar.gz
rig/membench/build.sh HOST          # stream_bench: MI50 read bandwidth vs waves, loads in flight, stride, rotation, launch size
rig/depth_bench.sh MODEL            # pp512 / tg128 / pp8 at context depths (default 0-64k), one GPU, TP2, TP4, with buffer sizes
rig/long_ctx_bench.py M D N DEPTH   # DFlash2 through llama-server after a DEPTH-token document, against the questions alone
rig/kvquant_bench.sh MODEL          # stock KV cache types: KLD at 2k, KLD at 32k against the f16 cache, speed at depth 0 / 64k
rig/passkey_bench.py M D N DEPTH    # pass key retrieval at 5 positions of a DEPTH-token document, through llama-server with DFlash2
rig/kv_box_bench.sh MODEL DRAFT     # q4_0 cache with our dot8 attention: KLD at 32k vs UD-Q8_K_XL, pass keys, speed, what context fits
ggml_bench --types w4a4,q4_0 --lm q6_K,q8_0 --widths 1,6,8 | python3 rig/bench_summary.py
```

Use `GGML_CUDA_DISABLE_GRAPHS=1` on ROCm 5.7: HIP graphs there are slower and less stable.

Activations are quantized in groups of 64 (`RQ_ACT_GROUP=128|256` for experiments); models from `ruquant convert
--wgroups` also carry a scale multiplier per 256 weights. Clips stored for the older per-token scales (0.8 / 0.7) are
too tight for groups: `ruquant setclip FILE 1.0 1.0`, or leave them unset (1.0). Compare speeds only within one
session: box 1 drifted ~10% between runs an hour apart. `acc_bench` runs on four GPUs at once give valid acceptance
but +-2 ms/step timing; time with one GPU alone or with `llama-bench -p 8 -ub 8`.

## Tensor parallelism (4x MI50)

```sh
HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_META_TOKEN_GRAPH=0 GGML_CUDA_DISABLE_GRAPHS=1 GGML_META_SPIN=2000000 \
  llama-server -m MODEL-rb4.gguf -md DRAFTER-rb4.gguf -ngl 99 -ngld 99 -fa 1 -lm dio -sm tensor -devd ROCm0 \
  --spec-type draft-dflash --spec-draft-n-max 7 --samplers temperature
```

- Models need `ruquant convert/drafter --rot-blocks 4`: a whole-row online Hadamard can't be split along K, and
  ggml aborts on such a model under `-sm tensor`.
- `HSA_FORCE_FINE_GRAIN_PCIE=1` enables the peer-write all-reduce (PCIe P2P; `rig/p2p_check.cpp` checks every pair).
  Without it peer writes fault and mx falls back to a slower staged all-reduce.
- `GGML_META_TOKEN_GRAPH=0`: HIP graphs on ROCm 5.7 are slower and noisy. `GGML_TP_AR_WIRE=f16|bf16|f32` picks the
  all-reduce payload (f16 default; bf16 costs W4A4 quality). The drafter goes on one GPU (`-devd ROCm0`).
- `GGML_CUDA_HOST_PROF=1` prints host time per op at exit; `GGML_META_PROFILE=1` the TP dispatcher's split.

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
