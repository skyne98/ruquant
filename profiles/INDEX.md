# Profiles

Recorded GPU profiles, kept so any of them can be studied again later.

**Layout**
- `box1/<UTC time>-<label>.tar.gz` holds one profile: rocprofv3 CSVs or results, the command's log, and the scripts
  that made it.
- `<name>.meta.txt` next to each archive holds its metadata, readable without unpacking: command, settings,
  environment, binary and `libggml-hip.so` md5, exit code.

**Tools**
- `rig/prof.sh` records a profile on the box and never overwrites one.
- `rig/prof_pull.sh HOST` copies the finished ones here.
- `python3 rig/prof_summary.py ARCHIVE [KERNEL_REGEX]` prints kernel-trace medians and counter medians per kernel
  and grid, from a directory or straight from an archive.

**Builds named below** (box 1, `~/mx-llama.cpp/build-*`, all TheRock 7.12):
- `build-r712`: the published default.
- `build-sk2`: the refactored skinny kernel with stream-K behind `RQ_SKINNY_SK`.
- `build-ablN`: timing-only ablations of `build-sk2`, with wrong results.
- `build-rot`: `build-sk2` plus `RQ_SKINNY_ROT`.

Write-ups of all of these are in REPORT-2026-11-02.md ("stream-K for the skinny kernel", "what limits the skinny kernel's
bandwidth").

## 2026-11-05: skinny W4A4 kernel, stream-K and memory bandwidth (box 1, GPU 1 unless noted)

| profiles | what | result |
|---|---|---|
| `skinny-rocprof57-1gpu` | ROCm 5.7 rocprof v1 kernel trace of one-GPU T=8 forwards (build/, 5.7) | per-launch skinny times in a forward; the TP4 trace of the same run was lost (rocprof v1 timed out writing it) |
| `skinny-perf-fused` | rocprofv3 kernel trace, test-backend-ops perf, W4A4 shapes, stream-K off / on | perf mode fuses three repeats per launch: grids 3× the single matmul |
| `skinny-per-shape` | single launches (`GGML_CUDA_DISABLE_FUSION=1`), split-K vs stream-K (120) | stream-K no faster; TP4 qkv+z 30.6 vs 34.1 us |
| `skinny-grid-sweep` | time vs grid size (split-K G, stream-K slots) | power-of-two shares (32 groups per workgroup) run 2× slower |
| `skinny-counters` | rocprofv3 counters, six groups × six configurations | nothing saturated: VALUBusy 13-34%, MemUnitStalled < 0.4%, FETCH_SIZE = weight bytes |
| `tccscan-e36` … `-e60` | gfx906 TCC event scan with custom definitions (`rpdefs`, `ROCPROFILER_METRICS_PATH`) | event 49 is the EA read level: at 64 workgroups, 770-cycle (450 ns) DRAM latency, ~2200 reads in flight |
| `membench-sweep` | `rig/membench/stream_bench`: waves × loads in flight × cached / nontemporal | plain streaming reaches 800-814 GB/s with 240 waves × 2 KB in flight |
| `membench-stride-rot` | GB/s against each wave's run length; per-wave rotation | power-of-two runs lose 25-45%; rotation recovers 6-21% in the skinny-like pattern. Part 1 used single launches (idle-clock bias) |
| `membench-size`, `membench-size-batched` | GB/s against launch size, single launches vs 50 back to back | single launches mismeasure small sizes (clocks); back to back, 10 MB at 64 workgroups streams 546 GB/s |
| `abl-<build>-<M>-<K>` (20) | kernel traces of `build-sk2` and ablations abl1 (no colsum/wk loads), abl2 (no activation loads), abl4 (no split-K fixup), abl7 (all three) | the three together are 4-5 us of 30 (TP4) and 17 of 175 (one-GPU gate+up); the stream itself is the main loss |
| `rot<N>-<M>-<K>` (30) | kernel traces of `build-rot` at `RQ_SKINNY_ROT` 0 / 1 / 3 / 5 / 7, six shapes | rotation 7: one-GPU gate+up 175 → 153 us, attention output 52 → 39; TP4 shapes ±2%; 5 is worse on some shapes |
| `fwd-rot` | T=8 forwards (llama-bench pp8), `RQ_SKINNY_ROT` 0 / 1 / 7, one GPU / TP2 / TP4 | one GPU 40.4 → 39.2 ms, TP2 30.2 → 29.6, TP4 unchanged |

## 2026-11-05: 64k context (box 1, all four GPUs, published build `build-r712`)

| profiles | what | result |
|---|---|---|
| `depth-64k` | `rig/depth_bench.sh` (pp512 / tg128 / pp8 at depths 0-64k, one GPU / TP2 / TP4) and `rig/long_ctx_bench.py` (DFlash2 after a 64k-token wikitext document, server logs included) | 64k adds 25 / 16 / 8 ms to the T=8 forward (one GPU / TP2 / TP4); DFlash2 steps 80 / 58 / 46 ms against 51 / 43 / 37 short; 4.35 GB of f16 KV cache |
| `kvquant` | `rig/kvquant_bench.sh`: KV cache f16 / q8_0 / q8_0+q4_0 / q4_0 (rotation on): KLD at 2k against UD-Q8_K_XL, KLD at 32k against our f16-cache run, speed at depth 0 and 64k (one GPU, TP4) | 2k KLD 0.0425 / 0.0420 / 0.0431 / 0.0440 (unrotated q4_0 0.0450); at 64k every quantized cache is slower (one-GPU decode 45 → 64 / 64 / 133 ms) |
| `kvquant-controls` | the 32k KLD's controls: f16 cache on one GPU, f16 with q8_0 all-reduces, q8_0 without rotation; the UD-Q6_K part failed (model no longer on box 1) | any numerics change gives the same 0.035 as a quantized cache: the 32k KLD measures the model's sensitivity, not cache error |

## 2026-11-06: q4 KV attention with v_dot8 (`rq-fattn.cu`, box 1)

Builds `build-fa` (v1), `build-fa2` (v2), `build-fa3` (v3, kept), `build-fa4` (v4, dropped).

| profiles | what | result |
|---|---|---|
| `fa-dot8-tests` | test-backend-ops FLASH_ATTN_EXT at the model's shapes, v1, RQ_FA_DOT8 0 / 1 / 2 | int8 Q passes all 8; int4 Q at NMSE 5.1-5.4e-4 (limit 5e-4) |
| `fa-dot8-speed` | v1 speed at depth (one GPU complete; stopped during TP4) | decode 64k 40.0 ms (f16 44.7); verify 64k 71.4 ms, slower than f16 (64.4) |
| `fa-dot8-v2`, `fa-dot8-v3`, `fa-dot8-v4` | one-GPU tg128 / pp8 at depth 0 / 16k / 64k per version, plus tests | verify 64k 60.3 / 59.6 / 59.1 ms; decode 38.7 / 38.5 / 40.9 |
| `fa-fa3-pmc0` … `-pmc4`, `fa-fa3-trace` | rocprofv3 counters and kernel trace of v3 at 64k (llama-bench pp8) | 1.22 ms per call, VALU busy 50%, waves waiting 54%, 64 GB/s: latency-bound at 2 workgroups per CU |
| `fa-dot8-quality` | KLD with 8-token ubatches (f16, stock q4_0, dot8 int4 Q, int8 Q) and DFlash2 after a 64k document (one GPU, TP4) | KLD 0.0416 / 0.0442 / 0.0423 / 0.0425; one-GPU DFlash2 step at 64k 71.5 ms against 79-80 with f16 |
| `fa-dot8-final` | final build (`build-fa7`, on by default, Q rounded as q4_0): all FLASH_ATTN_EXT tests, KLD at 8-token ubatches, one-GPU speed at 64k | 4001/4001 tests pass; KLD 0.0434; verify 59.6 ms, decode 38.5 ms at 64k |
