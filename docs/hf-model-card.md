---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - w4a4
  - int4
  - quantized
  - amd
  - mi50
  - gfx906
  - speculative-decoding
  - dflash
---

# Qwen3.8-27B W4A4 GGUF (int4 weights × int4 activations)

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) quantized by [ruquant](https://github.com/skyne98/ruquant)
to pure W4A4. Both the weights and the activations are int4, and the matmuls run as int4 × int4 dot products
(`v_dot8_i32_i4` on AMD gfx906 / MI50). A W4A4H version of incoai's DFlash2 drafter comes with it, for speculative
decoding.

**These files need the ruquant fork of mx-llama.cpp:**
[skyne98/mx-llama.cpp, branch `ruquant`](https://github.com/skyne98/mx-llama.cpp/tree/ruquant). Stock llama.cpp does
not know the W4A4 / W4A4H tensor types. The fork was built and measured on MI50s (gfx906) with ROCm 5.7.

| file | size | what |
|---|---|---|
| `Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf` | 17.19 GB | the target model |
| `Qwen3.8-27B-DFlash2-W4A4H-drafter-rb4.gguf` | 2.65 GB | the DFlash2 drafter converted for it |
| `SHA256SUMS` | | checksums |

## Results (4× AMD MI50 32 GB, ROCm 5.7, PCIe 3.0)

Measured in one session on 2026-11-05. The comparison is unsloth's Q4_0 (4-bit weights, 8-bit activations) with
incoai's DFlash2. Speculative runs: 30 prompts × 256 tokens, greedy, draft 7 (Q4_0: 4, its best).

| | 1 GPU | TP2 | TP4 |
|---|---|---|---|
| **W4A4 + DFlash2, t/s** | **80.2** | **89.5** | **109.6** |
| Q4_0 + DFlash2, t/s | 43.5 | 60.5 | 70.5 |
| W4A4 acceptance (tokens per step) | 4.35 | 4.27 | 4.31 |
| verify forward of 8 tokens, ms: W4A4 / Q4_0 | 42.2 / 95.5 | 35.1 / 62.8 | 27.6 / 40.6 |
| prefill pp2048, t/s: W4A4 / Q4_0 | 487.8 / 290.7 | 833.1 / 529.0 | 1324.0 / 901.4 |
| plain decode, t/s: W4A4 / Q4_0 | 29.2 / 27.9 | 41.6 / 39.7 | 52.7 / 51.5 |

Quality, KL divergence against UD-Q8_K_XL's logits (wikitext-2, 64 chunks of 2048):

| | KLD | top-1 | PPL / base |
|---|---|---|---|
| unsloth UD-Q4_K_XL | 0.0078 | 96.58% | 0.999 |
| unsloth Q4_0 | 0.0259 | 93.34% | 1.022 |
| **this model (W4A4)** | **0.0428** | **91.57%** | 1.028 |

## Reproduce the results

The speed numbers above ran exactly as below, on 4× MI50 32 GB with ROCm 5.7.1. The 64-chunk quality numbers come
from the same `llama-perplexity` commands on an RTX 3090, built from the ruquant repo's llama.cpp patch.

**1. Build the fork** (gfx906; needs ROCm, cmake and ninja; the flags of `rig/build_mx.sh` in the ruquant repo):

```sh
git clone -b ruquant https://github.com/skyne98/mx-llama.cpp && cd mx-llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_RCCL=ON \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j --target llama-server llama-bench llama-perplexity test-backend-ops
HIP_VISIBLE_DEVICES=0 build/bin/test-backend-ops -o MUL_MAT -p type_a=w4a4    # 56/56 passed: bit-exact
```

**2. Environment, for every command below:**

```sh
export GGML_CUDA_DISABLE_GRAPHS=1     # HIP graphs are slower and flaky on ROCm 5.7
export HSA_FORCE_FINE_GRAIN_PCIE=1    # the peer-write all-reduce under -sm tensor needs it
export GGML_META_TOKEN_GRAPH=0        # no token-graph capture (HIP graphs again)
export GGML_META_SPIN=2000000         # tensor-parallel lane threads spin instead of sleeping (+2%)
```

`HIP_VISIBLE_DEVICES` picks the GPUs: `0` for one GPU, `0,1` for TP2, `0,1,2,3` for TP4.

**3. The server behind the W4A4 + DFlash2 numbers (the optimal configuration), TP4:**

```sh
HIP_VISIBLE_DEVICES=0,1,2,3 build/bin/llama-server \
    -m Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf -md Qwen3.8-27B-DFlash2-W4A4H-drafter-rb4.gguf \
    -ngl 99 -ngld 99 -fa 1 -lm dio -c 8192 -np 1 -sm tensor -devd ROCm0 \
    --spec-type draft-dflash --spec-draft-n-max 7 -bs --samplers temperature
```

- TP2: the same line with `HIP_VISIBLE_DEVICES=0,1`.
- One GPU: `HIP_VISIBLE_DEVICES=0` and no `-sm tensor -devd ROCm0`.
- What each setting does, and what was measured against it:
  - `-sm tensor` splits every layer across the GPUs (tensor parallelism).
  - `-devd ROCm0` pins the drafter to one GPU. Replicating it on every GPU (mx's default) is 19% slower; splitting it
    is 24% slower.
  - `--spec-draft-n-max 7` is the most DFlash2's block of 8 allows, and the best: 5 and 6 are 6% and 4% slower.
  - `-bs --samplers temperature` with temperature 0 is greedy decoding.
  - `-lm dio` only loads the file with direct I/O.

The requests: `POST /v1/chat/completions` with `"temperature": 0, "max_tokens": 256, "cache_prompt": false`, over the
30 prompts in the ruquant repo's `rig/acc_prompts.json` (10 math, 10 code, 10 chat). t/s is the mean of the server's
`timings.predicted_per_second`, and acceptance is accepted tokens per verify step. `rig/acc_bench.py` does both. It
starts the server with the line above and sends the 30 requests; run it from the mx-llama.cpp directory, with
[skyne98/ruquant](https://github.com/skyne98/ruquant) cloned next to it:

```sh
GPUS=0,1,2,3 EXTRA="-sm tensor -devd ROCm0" BIN=build/bin python3 ../ruquant/rig/acc_bench.py \
    Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf Qwen3.8-27B-DFlash2-W4A4H-drafter-rb4.gguf 7
```

For prompt-heavy serving, also add `-b 2048 -ub 2048 --ctx-checkpoints 0`. TP4 prompt processing in the server goes
from 845 to 1066 t/s, but without checkpoints the server can't reuse a cached prefix. The tables above don't use it.

**4. Prefill, the 8-token verify forward and plain decode** (drop `-sm tensor` on one GPU):

```sh
build/bin/llama-bench -m Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf -ngl 99 -fa 1 -lm dio -sm tensor -p 512,8 -n 128 -r 3
build/bin/llama-bench -m Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf -ngl 99 -fa 1 -lm dio -sm tensor \
    -p 2048 -n 0 -b 2048 -ub 2048 -r 3
```

`pp8` is the verify forward (ms = 8000 / t/s), and `tg128` plain decode. Prefill all-reduces travel as q4_K blocks by
default; `GGML_NCCL_AR_WIRE=f16` sends them in f16, which cost TP4 pp2048 about 30% (947 vs 1325 t/s, 2026-11-04).

**5. Quality.** First, a reference from unsloth's Qwen3.8-27B-UD-Q8_K_XL on the wikitext-2 raw test set. It is
33 GB, so offload as many layers as fit (`-ngl 44` on a 24 GB card). Any llama.cpp can write it:

```sh
build/bin/llama-perplexity -m Qwen3.8-27B-UD-Q8_K_XL.gguf -ngl 44 -f wiki.test.raw -c 2048 -b 2048 --chunks 64 \
    --kl-divergence-base ud-q8_k_xl.kld
```

Then score the model against it:

```sh
build/bin/llama-perplexity -m Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf -ngl 99 -fa on -lm dio -c 2048 -b 2048 \
    --kl-divergence-base ud-q8_k_xl.kld --kl-divergence
```

Under `-sm tensor`, add `GGML_NCCL_AR_WIRE=f16`. Otherwise perplexity quantizes every token's prefill all-reduce,
where serving quantizes only prompts. On the MI50s against a 16-chunk reference (`--chunks 16`): KLD 0.0425 / top-1
91.44% on one GPU, 0.0420 / 91.57% at TP4.

**6. All of it in one go**, from the ruquant checkout (`git clone https://github.com/skyne98/ruquant`). It runs 1 GPU,
TP2 and TP4, then scores the model against a 16-chunk reference in `~/ref/ud-q8_k_xl-16.kld`:

```sh
export BIN=/path/to/mx-llama.cpp/build/bin
rig/final_mi50.sh Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf Qwen3.8-27B-DFlash2-W4A4H-drafter-rb4.gguf
NMAX=4 SCORE=0 DEVD= rig/final_mi50.sh Qwen3.8-27B-Q4_0.gguf Qwen3.8-27B-DFlash2-Q8_0.gguf   # the Q4_0 baseline
```

The baseline's drafter is replicated (`DEVD=`): incoai's drafter borrows the target's lm_head, which `-sm tensor`
splits, so pinning it aborts at load. Our converted drafter has its own head.

Measurement notes:
- The first speculative TP4 run of a session ran slower (41.8 ms per step, ~102-103 t/s) with either model. The
  TP4 109.6 t/s is the mean of the two runs after it (109.2, 110.0).
- The box drifts by up to ±3% between sessions, so the tables are one session.

## How it was made

- **Types**: unsloth's Qwen3.8-27B-Q4_0 type map. Its 4-bit linears become `W4A4`, and attn_output / ffn_down
  become `W4A4H` (an online Hadamard first). `ssm_out` and the lm_head are Q8_0.
- **Rotation**: OSTQuant, a learned R1 rotation plus smoothing scales, folded into the weights. The online Hadamard
  is block-diagonal, 4 blocks per row (Paley × Sylvester), so one file serves 1, 2 and 4 GPUs.
- **Weights**: GPTQ against Hessians collected on the rounded model (64 wikitext windows, every position). 4.5 bits
  per weight: a per-row scale and a 4-bit multiplier per group of 64.
- **Activations**: quantized on the fly per token, asymmetric int4 in groups of 64. Each group's scale is an integer
  multiple of a per-token unit, so the dot product stays in int32 until one fp32 epilogue. No activation clip.
- **Drafter**: incoai's [Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). Its `fc` takes the
  target's R1 and stays Q8_0; its layers and lm_head become W4A4H.

Every kernel is bit-exact with the CPU reference implementation. The full recipe, methodology, measurements and what
did not work: [RESULTS.md](https://github.com/skyne98/ruquant/blob/main/RESULTS.md).

## Limits

- Quality is below unsloth's Q4_0, which keeps 8-bit activations: KLD 0.043 against 0.026, top-1 91.6% against 93.3%. That is the price of 4-bit
  activations, and it buys the int4 × int4 speed.
- The kernels are written for gfx906. The fork's CUDA path emulates dot8 and is only a correctness check. The ruquant
  repo has a separate llama.cpp patch with a sm_80+ CUDA path, used for quality scoring on an RTX 3090.

## Credits and license

Apache-2.0, like its sources:
- Qwen3.8-27B by Qwen;
- the GGUF source and type map by unsloth;
- the DFlash2 drafter by Inco AI;
- the runtime builds on mx-llama.cpp (MIT) and llama.cpp (MIT).

Methods: OSTQuant (Hu et al., 2025), GPTQ (Frantar et al., 2022), DFlash speculative decoding.
