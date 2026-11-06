# Distilling W4A4 with a donated MI50 fleet: design (2026-11-05)

The idea: community-donated MI50s run the teacher (unsloth's Q8 or Q6_K Qwen3.8-27B) over a large corpus and stream
randomly sampled teacher logits, live, to one or two rented RTX 5090s. The 5090s train our W4A4 student against them.
No logits touch disk.

This document covers the method, the numbers, the system and the donor experience. Measured numbers are marked as
measured; everything else is an estimate to check in the first experiments (end of document).

## TL;DR

- **The method exists and fits.** Random Sampling KD (RS-KD, ACL 2025) samples about 12 unique tokens per position from
  the teacher's distribution and sends their ids and counts. It gives an unbiased KD gradient and matched full-logit
  distillation in the paper. Top-K is biased: it drops the tail and inflates the kept probabilities.
  - Per token that is ~36-48 bytes, so a donor at 250 tokens/s uploads ~10 KB/s.
  - Streaming live to the trainer is easy, and fresh samples every pass amortize the sampling over tokens and epochs.
- **The 5090 is the bottleneck, not the fleet.**
  - Training the 27B student end to end costs ~164 GFLOP per token, so one 5090 does ~400-600 tokens/s.
  - A 32 GB MI50 running unsloth's UD-Q6_K teacher does **261 tokens/s** (measured on box 1, pp2048), and 248 tokens/s
    with every position's logits copied to the host.
  - So **one 5090 is fed by ~2 MI50s**, and two 5090s by ~3-5.
  - A 100M-token distillation then takes ~1-2 days on two 5090s (~$20-55 of rental at $0.35-0.99/h).
- **A big donor fleet is worth more than teaching one or two 5090s:**
  - more rented trainers, scaled linearly;
  - decentralized training on the MI50s themselves (DiLoCo-style, only the small quantization parameters synced);
  - generating drafter-training text (decode-heavy, where many cards help);
  - distillation campaigns for more models, and checkpoint evaluation.
- **Donors run one `docker run` line.**
  - Everything is outbound HTTPS; the teacher is downloaded once from Hugging Face; one worker runs per GPU.
  - Work units are small, leased and retried.
  - Duplicate-assigned work units with deterministic seeds catch bad results, because the same container on the same
    gfx906 should give bit-identical teacher logits (to verify).
- **First, prove the training recipe on the 4B proxy.** It runs on a single GPU in minutes. If distillation can't move
  W4A4 KLD there, the fleet won't help.

## 1. Method

### What gets trained

The student is our W4A4 format:
- int4 codes, a per-row fp32 scale, four 4-bit multipliers per 256-block;
- per-token int4 activations in groups of 64;
- OSTQuant's R1 and smoothing folded into the weights, and an online Hadamard on attn_output and ffn_down.

Two phases, after EfficientQAT, which trained a 2-bit Llama-2-70B on one A100-80GB in 41 h:

1. **Block-wise (Block-AP), on the trainer alone, no fleet.**
   - Per transformer block, train the latent weights and quantization parameters with fake W4A4 quantization, to match
     the BF16 block's outputs on calibration activations.
   - The BF16 block (0.8 GB) runs on the trainer.
   - 4096 × 2048 calibration tokens (EfficientQAT found 128 samples overfit and 4096 fixed it).
2. **End-to-end (E2E-QP + KD), fed by the fleet.**
   - Freeze the int4 codes and train only the continuous quantization parameters end to end: per-row scales, activation
     clips, optionally smoothing.
   - The loss is RS-KD against the teacher's sampled distribution, optionally plus a small cross-entropy term on the
     text.
   - This is the phase where the token count, and so the fleet, matters.

The discrete parts of the format, the 4-bit codes and multipliers, change only in phase 1, through a straight-through
estimator on the latent weights. Phase 2 is light: weights stay frozen, gradients flow only to scales and clips, and
optimizer state is small.

### Random Sampling KD ([Anshumann et al., ACL 2025](https://arxiv.org/abs/2503.16870))

How it works:
- Per position, draw N tokens with replacement from the teacher distribution q = p^τ (τ = 1 by default; 0.8-1.2
  behave the same, and uniform sampling diverges). Keep the unique ids and their counts c_i, so the sparse target is
  c_i / N.
- The student minimizes forward KL to that sparse target. With τ = 1 this is cross-entropy on the sampled tokens,
  weighted by count.
- It is unbiased and preserves the gradient in expectation.
- The paper's main setting is 12 unique tokens per position, at 17-bit id + 7-bit probability = 3 bytes each.

Results:
- 8B → 3B over 100B tokens, all at 12 entries: Top-K reached LM loss 2.499 with ECE 4.66%; RS-KD 2.350 / 0.20%; full KD
  2.344 / 0.20%. Speculative-decoding acceptance was 73.19% for RS-KD against 73.36% for full KD.
- Training overhead was under 10% over plain cross-entropy.

For us, Qwen's 248,320-token vocabulary needs 18 bits per id, so an entry is 4 bytes and a position ~48 bytes at 12
unique tokens. Because the trainer consumes samples live, every epoch draws fresh samples (seed by campaign, work unit
and position). That amortizes the sampling noise over both tokens and passes, which is the "basically perfect distill"
effect. Nothing needs storing.

### Teacher

- **unsloth Q8 (UD-Q8_K_XL, 33 GB) or Q6_K (22 GB):** both are near-lossless; their error is small next to the student's
  gap (W4A4 KLD 0.043, Q4_0 0.026).
  - Measured on box 1, 16 chunks: UD-Q6_K against UD-Q8_K_XL has KLD 0.0024, top-1 98.0% and PPL +0.2%. That is 17×
    below our W4A4 student's KLD, so Q6_K is a good teacher.
- **Fitting it in memory:**
  - UD-Q8_K_XL doesn't fit a 32 GB MI50 alone, so it needs 2 cards.
  - A plain Q8_0 (~28.7 GB) does fit one 32 GB card for 2048-token prefill: the KV cache of the hybrid model is tiny.
  - Q6_K fits one 32 GB card easily, or two 16 GB cards.
- **One teacher file per campaign,** pinned by sha256. Mixing Q8 and Q6_K donors is acceptable (both near-lossless), but
  results are only compared within one teacher class.

The teacher's job is prefill with logits at every position, which is what `llama-perplexity` does.
- 248,320 floats are ~1 MB per token, ~250 MB/s over PCIe at the teacher's rate. It is cheaper than it sounds:
  `llama-perplexity`, which copies all of them to the host, ran at 248 tokens/s against 261 for prefill alone (-5%).
- So the first worker can sample on the CPU: log-softmax, N draws with replacement, unique ids + counts. A sampling
  kernel after the lm_head (~48 bytes per token off the GPU) is a later optimization worth ~5%.

## 2. Throughput and sizing

### Teacher on an MI50

- **Cost:** ~2 × 27.3B ≈ 55 GFLOP per token, plus the lm_head at every position (2.5 GFLOP).
- **Measured on box 1, one 32 GB MI50, ROCm 7.12:** prefill pp2048 is 504 t/s for our W4A4 and 307 t/s for unsloth's
  Q4_0.
- **Measured for the UD-Q6_K teacher (21 GiB), box 1, ROCm 7.12, pp2048:**

  | | ubatch 512 | ubatch 2048 |
  |---|---|---|
  | one 32 GB MI50 | 252 t/s | 261 t/s |
  | two MI50s, layer split (as 2× 16 GB cards would run it) | 389 t/s | 260 t/s |
  | one MI50, all logits copied to the host (`llama-perplexity`) | | 248 t/s |

  - Two cards in layer split pipeline the four 512-token ubatches, so a pair of 16 GB cards should give ~390 t/s.
    That is ~195 per card, against 261 for a card that holds the whole teacher.
  - Not measured: UD-Q8_K_XL on two cards (it isn't on box 1, and the disk has 17 GB free). It reads 1.5× the bytes
    of Q6_K, so expect it to be somewhat slower.

### Student on an RTX 5090

- **5090 specs:** 32 GB GDDR7 at 1.8 TB/s; ~209.5 TFLOPS dense BF16 with fp32 accumulate (419 with fp16 accumulate,
  838 FP8).
- **Cost of phase 2:** forward 2N + recompute 2N (gradient checkpointing) + backward to inputs 2N ≈ 6N = 164 GFLOP per
  token for N = 27.3B. There are no weight gradients, only scales and clips.
- **Throughput:** at 35-50% of peak that's **~400-600 tokens/s per 5090**, RS-KD overhead included.
  - Cross-check against EfficientQAT's Table 7: its end-to-end phase on Llama-2-70B took ~14.3 h on one A100-80GB for
    4096 × 4096 tokens, i.e. ~330 tokens/s.
  - Scaled to 27B and to the 5090's lower BF16 peak (209.5 against 312 TFLOPS), that is ~570 tokens/s.
- **Memory:** int4 student ~15 GB + checkpoints of 64 layers at seq 2048 (~1.3 GB) + one layer's recompute + full-vocab
  student logits computed in chunks ≈ 20-24 GB. It fits.
- **Phase 1 (Block-AP):**
  - ~6 × 0.42B × 16.8M tokens (4096 × 2048, 2 epochs) ≈ 42 PFLOP per block, about 9 minutes at 80 TFLOPS, so ~10 h for
    all 64 blocks on one 5090.
  - Scaling EfficientQAT's 26.6 h for the 70B on an A100 gives ~15 h instead, so plan for 10-15 h. No fleet needed.
  - The block inputs and the BF16 targets are 4096 × 2048 × 5120 × 2 bytes = 86 GB each. They live in the rented box's
    RAM or on its NVMe, which is local scratch rather than stored logits.
  - Phase 1 is optional: phase 2 can start from today's GPTQ model.

### Putting it together

| | 1× 5090 | 2× 5090 | 10× 5090 |
|---|---|---|---|
| student tokens/s (estimate) | 400-600 | 800-1200 | 4000-6000 |
| teacher MI50s needed (32 GB, 261 t/s each, measured) | ~2 | ~3-5 | ~15-23 |
| 100M tokens of phase 2 | ~2.3 days | ~1.2 days | ~5 hours |
| rental for that run ($0.35-0.99 per GPU-hour) | ~$20-55 | ~$20-55 | ~$20-55 |
| upload from the whole fleet (48 B/token + framing) | ~30 KB/s | ~60 KB/s | ~300 KB/s |

Two notes on the table:
- **Cost doesn't grow with the number of trainers,** because the GPU-hours are the same. More 5090s only shorten the
  wall time.
- **The token budget is a guess to tune on the 4B proxy.** EfficientQAT's end-to-end phase used 4096 × 4096 ≈ 17M
  tokens in one epoch. W4A4 with a KD loss likely wants more, so 50-200M.

The answer to "will a 5090 be fast enough": yes, one 5090 trains the 27B student, and the whole run fits in days. But
it consumes only ~2 MI50s of teacher output. A fleet of dozens of MI50s needs more trainers, or more jobs, to be used.

### Using a large fleet

- **More trainers.** Data-parallel 5090s share only the small set of quantization parameters (well under 1 GB), so
  they can sync over the internet DiLoCo-style, every 100-500 steps. Prime Intellect's
  [INTELLECT-1](https://arxiv.org/abs/2412.01152) trained across 3 continents this way with ~400× less communication.
- **Training on the donors' MI50s.**
  - mixa3607/ML-gfx906 ships PyTorch wheels for gfx906 on ROCm 7.14 / 10.x.
  - A 32 GB MI50 at ~10 TFLOPS of real fp16 trains the 27B student at roughly 60 tokens/s, with a second card or a
    neighbour as its teacher. So 50 such pairs would train ~5-7× faster than one 5090.
  - It needs robust aggregation against bad donors (outlier-robust averaging, reputation); more research, more upside.
- **Drafter data.**
  - Fine-tuning DFlash2 on our W4A4 target (worth up to +9% speed) needs the target's own continuations: decode at
    ~30-80 tokens/s per MI50, so many cards help.
  - Only the text needs to travel. The trainer recomputes the five tapped hidden states with one prefill, instead of
    receiving 25-50 KB per token.
- **Evaluation and more campaigns:** KLD / acceptance scoring of checkpoints, and other models (Qwen3.8-Flash-Next
  is 70 GB even at 4 bits, so it needs several trainer GPUs).

## 3. System

```
 donors (docker, one worker per GPU)        coordinator (public HTTPS)              trainer (rented 5090s)
 ┌───────────────────────────┐   lease WU   ┌───────────────────────────────┐ stream ┌──────────────────────┐
 │ teacher GGUF (HF, sha256) │ ───────────▶ │ campaigns, WU leases (SQLite) │ ─────▶ │ bounded RAM queue    │
 │ prefill + sampler (CPU)   │ ◀─────────── │ validation, reputation, credit│ ◀───── │ RS-KD loss, AdamW on │
 │ results: ids+counts, zstd │  results     │ backpressure from the trainer │ depth  │ quant params, ckpt   │
 └───────────────────────────┘              └───────────────────────────────┘        └──────────────────────┘
        corpus shards: Hugging Face dataset streaming, by shard id (donors and trainer read the same text)
```

**Work unit (WU):**
- (campaign, corpus shard id, sequence range, seed, teacher sha256, sampler settings).
- 16 sequences × 2048 tokens ≈ 33k tokens, about 2 minutes on one MI50, so a lost WU costs little.
- The text itself never crosses the network: donors and trainer stream the same shard from Hugging Face and
  tokenize it with the same tokenizer.

**Leases:**
- A WU is leased for ~3× its expected time, with a heartbeat every 30 s.
- An expired lease is reissued; results are idempotent by WU id, and duplicates are dropped.
- A worker that gets SIGTERM returns its lease.

**Streaming and backpressure:**
- Results arrive as zstd-compressed binary frames over HTTPS or WebSocket: per sequence, per position, (count, id) pairs.
- The coordinator forwards accepted results into the trainer's bounded in-RAM queue, a few minutes deep.
- When the queue is full, the coordinator stops leasing, and donors idle or move to an evaluation or drafter-data
  campaign.
- Nothing is written to disk but WU metadata and the trainer's small checkpoints.

**Trainer:**
- It pulls mini-batches from the queue in any order (a shuffle buffer), aligns each with the token positions of its
  shard, and steps.
- It checkpoints the trainable parameters (small) every ~10 minutes. A restart resumes, and in-flight WUs are simply
  reissued.
- Rented marketplace hosts can vanish (a known caveat of spot/community tiers), so checkpoints also go off-box.

**Epochs:** the same shards are reissued with a new seed, so each pass gets fresh samples.

## 4. Trust and verification

Donors are untrusted, and a few bad results stream straight into training. The defences, borrowed mostly from BOINC
and Prime Intellect:

- **Deterministic duplicates.**
  - Seeds derive from (campaign, WU, sequence, position). The same container on the same architecture should produce
    bit-identical logits, so two donors' results for the same WU can be compared exactly.
  - Our W4A4 integer matmuls are bit-exact, and our split-K sums are integer atomics, but every teacher kernel must be
    checked for this (first experiments).
  - Only compare within one platform class, BOINC's "homogeneous redundancy": the same image, gfx906, the same teacher
    file.
- **A trust ramp.**
  - A new donor's first ~20 WUs are all duplicated, then 5-10% at random, plus canary WUs with known answers.
  - Each donor gets a running error rate, as MilkyWay@home does.
  - On a mismatch, the donor is quarantined and their in-flight WUs reissued.
- **Bounded damage.** Online training can't un-see a bad batch. The coordinator can hold each result a few minutes,
  until its duplicate (if any) arrives, before forwarding. Or accept the small noise of a caught-late bad donor; with a
  5-10% check rate a cheater is caught within a few dozen WUs.
- **TOPLOC, if duplication gets expensive.** Prime Intellect's [TOPLOC](https://primeintellect.ai/blog/toploc) commits
  to the top-k values of the last hidden state. A trusted validator re-runs only sampled positions, which catches
  model, prompt or precision swaps. INTELLECT-2 used it to verify untrusted inference workers.
- **Pinned teacher.** Workers report the teacher file's sha256 and the image version; anything else is refused.

## 5. Donor experience

One command, after getting a token from a Discord bot (`/donate`):

```sh
docker run -d --name ruquant-kd --restart unless-stopped \
    --device /dev/kfd --device /dev/dri --group-add video \
    --cap-drop ALL --security-opt no-new-privileges \
    -v ruquant-kd:/data -e RQ_TOKEN=<token> -e RQ_GPUS=all \
    ghcr.io/skyne98/ruquant-kd-worker:latest
```

**The image:**
- Ubuntu 24.04 with a slim ROCm userland for gfx906, either mixa3607's ROCm 10.1 packages or AMD's TheRock 7.12. Inside
  Docker we pick the base, so box 1's glibc problem doesn't exist.
- Our patched llama.cpp, with a `kd-worker` mode: prefill + the RS-KD sampler.
- The worker client: one static binary (Rust fits the project) or a small Python app.
- ~2-3 GB in total.
- The host needs only the amdgpu kernel driver that any MI50 llama.cpp box already has. Box 1 runs TheRock 7.12's
  userland on its ROCm 5.7-era driver.

**First start:**
1. Detect GPUs: VRAM 16 or 32 GB, gfx906.
2. Choose the teacher: Q8_0 on one 32 GB card, Q6_K on 2× 16 GB, or as the campaign says.
3. Download it once into the volume from Hugging Face and check its sha256.
4. Run a 30-second self-benchmark, register (GPU model, VRAM, measured tokens/s), and start leasing.
5. One worker per GPU (or per pair of 16 GB cards).

**Updates:** each lease carries the required image version. An outdated worker exits with a known code, and Docker's
restart pulls the new tag (`--pull always` at start, or Watchtower).

**Safety for donors:**
- No inbound ports; outbound HTTPS only to the coordinator and Hugging Face.
- Non-root inside the container, `--cap-drop ALL`, only the GPU device nodes.
- Optional `--cpus` / memory limits, and a "pause when the GPU is busy" check for shared machines.

**Feedback:** per-donor credit (tokens taught, validated) and a leaderboard posted by the Discord bot. BOINC's
experience is that visible credit keeps volunteers.

## 5b. Existing projects: what to borrow

| project | what it is | what we take |
|---|---|---|
| [BOINC](https://github.com/BOINC/boinc/wiki/JobReplication) | volunteer-computing middleware | replication/quorum, homogeneous redundancy, adaptive per-host validation, credit. Not the server itself: its file-based batch flow doesn't suit online streaming |
| [Prime Intellect TOPLOC / INTELLECT-2](https://arxiv.org/html/2505.07291) | verifiable inference on untrusted workers; decentralized RL | TOPLOC-style commitments if duplication is too costly; prime-rl's HTTP trainer / inference split |
| [OpenDiLoCo / INTELLECT-1](https://arxiv.org/abs/2412.01152) | low-communication training across the internet | the option of training on donors or many rented trainers |
| [Petals](https://research.yandex.com/blog/petals-decentralized-inference-and-finetuning-of-large-language-models) / Hivemind | splitting one model across volunteers (DHT) | not needed: a whole teacher fits one donor box |
| [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906) | ROCm, PyTorch and llama.cpp builds and images for gfx906 | the base ROCm layer the community already trusts; PyTorch for on-donor training |
| [RS-KD](https://arxiv.org/abs/2503.16870) | unbiased sparse logit distillation | the loss and the wire format |
| [EfficientQAT](https://arxiv.org/abs/2407.11062) | Block-AP then E2E-QP | the two-phase training plan |
| Hugging Face datasets / Hub | corpus streaming, model downloads | donors fetch text and teacher directly; only ids travel through us |

## 6. First experiments, in order

1. **Prove the recipe on the 4B proxy.** It's a single GPU (the 3090 or one 5090), with a BF16 4B teacher on the same
   machine. Run phase 1 + phase 2 with RS-KD against the current pipeline (64-chunk KLD 0.0691). Find out how far KLD
   moves and how many tokens it takes. This decides everything else.
2. **The teacher on an MI50** (done for UD-Q6_K, section 2): 261 t/s, 248 with every logit copied to the host.
   - Left to do: UD-Q8_K_XL on two cards, and a CPU sampler at full rate (~65M logits/s per GPU through log-softmax
     and sampling).
3. **Determinism:** the same WU on two different MI50s in the same container must give identical sampled ids. If not,
   find the non-deterministic kernels (float atomics) or fall back to tolerance and TOPLOC.
4. **Prototype:** the worker, a coordinator on a small VPS, and the trainer on one rented 5090. Close-beta with 3-5
   donors, then open it on the Discord.
5. **Then the 27B campaign:** phase 1 on the 5090 (~10 h), phase 2 fed by the fleet, KLD and DFlash2 acceptance scored
   on box 1.
