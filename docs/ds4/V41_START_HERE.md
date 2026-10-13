# Strata-DSV4.1 — START HERE

**A new session reads THIS file and nothing else before starting work.**  Everything below is current and measured.
Open the long files only for a specific question (see "Where the details live").  Last updated 2026-10-13 evening (session 6: DSpark D1 in the engine DONE; D2 designed, not coded).

## 1. What this is
Our own engine (`tools/v41/engine/`) running **DeepSeek-V4.1-Flash** (40 layers x 384 routed experts, top-6, Engram,
CSA/indexer attention, mHC) from **our own GSQ mixed-tier quant** (experts 42.8% ternary PTQ1_0 / 39.2% Q2_0 / 18.0%
Q3_K = 2.25 bpw; dense Q8_0) on **one RTX 4090 24 GB + Ryzen 7 5700X + 90 GB DDR4 + NVMe**.  Public: GitHub
`Indras-Mirror/Strata-DSV4.1` (local remote `ds4`, `git push ds4 deepseek4:main`, pushing is OK'd), HF
`IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored` (upload complete; card = `tools/v41/hf/README.md`).

**Mal's goal now: 50+ tok/s decode with DSpark speculative decoding + n-gram drafting + plain-decode work, plus vision.**

## 2. Numbers today
| | |
|---|---|
| Prefill 8K / 16K / 20K | 244-248 / 288 / 266 tok/s, exact (MMQ default) |
| Decode p256 x 256, forced tokens (identical work) | **26.4-27.5 tok/s = 37.5 ms/token** (28-29.5 on luckier token paths) |
| Quality | 4096-tok code ppl 2.3790 exact routing; decode biases (defaults) cost code 2.38 -> 2.87 |
| Head gate vs FP8 teacher (held-out) | ppl 5.645 vs 4.915, top-1 0.674 vs 0.701 |
| DSpark (offline harness, fixed) | **OUR engine states: 3.27 accepted drafts/pass = 4.27 tokens/pass** (378 steps; 43% full 5-blocks; first draft 88.6%; confidence calibrated 23/61/91/99%); teacher states 3.24 / 4.24 - our 2-bit drift costs nothing. By verify depth 1..5: 1.89 / 2.65 / 3.29 / 3.83 / 4.27 tokens/pass |
| Our hidden states vs teacher at layers 37-39 (exact routing, 639 tokens) | cosine **0.876-0.879** (large drift) |
| Vision / DSpark in the engine | **NOT implemented** (weights exported in the sidecar only) |

## 3. Per-token decode budget (37.5 ms; `STRATA_DEC_PROF=1` phase timer + nsys node trace)
- `attn_router` **17.8 ms** = the dense CUDA graph per layer (15.2) + router graph (1.1) + readback.  Inside: dense
  Q8_0 mat-vecs **8.94 ms at ~850 GB/s = 84% of peak** (7.6 GB/token incl. the LM head and the grouped
  `attn_output_a`, nrows 1024 x 8 channels) - at the bandwidth wall; **elementwise "zoo" 5.45 ms** = 3288 kernels/token
  of mean < 5 us (`k_bin_bcast`, `scale_f32`, `quantize_q8_1`, `rms_norm`, `concat`, `cpy`, `dsv4_hc_*`, sigmoid...);
  attention itself ~0.74 ms.
- `tier` **16.6 ms** = expert service: VRAM hits (~96%) + PCIe misses (~5.6/token, ~0.6 ms each) + NVMe file-tier
  misses (~2.6/token, **6.0 ms/token of reads**, on the critical path) + CPU pool (~1.8 ms) + device sum.  Expert
  kernels on the device ~3.7 ms (ternary `native_gu<143>` runs at only ~136 GB/s: ALU-bound trit decode).
- `finish` 1.9, `begin` (Engram) 0.9, `predict` 0.26.
- Structure: per layer `attn_router(l)` -> host reads top-6 ids -> `tier(l)` -> k-ordered device sum -> `attn_router(l+1)`.
  Strictly serial per layer by data dependency (the sum feeds the next layer); prefetch (`--pf-b 1.43`) is the only
  overlap and is tuned.

## 4. Dead ends - measured, do not retry
Slice C (selection on-device: the k-ordered sum needs host-served miss rows every layer) · C' (GPUDirect: no
nvidia-fs, gguf on ntfs3; resident file tier = 83 GB, impossible) · removing ggml-stream syncs (S4: the wait moves) ·
tier metadata packing / tier CUDA graph `STRATA_TIER_GRAPH` / demand-copy overlap `STRATA_DMA_OVERLAP` (all
bit-identical, all neutral: host calls under a busy device) · `--pf-b` other than 1.43 · bigger arena (62/65 GiB:
neutral) · speculative NVMe reads as implemented (`STRATA_SPEC_TOP`: tier -1.4 ms but its extra predict graph +1.4 and
decode-thread issuing +1.2 ate it) · attention-level levers (KV quant, window, MLA absorb, split-K: <= 0.7 ms total) ·
quarter-block ternary dot / `STRATA_EXP_MODE_T*` (~1%) · hard pruning (soft penalties only - Mal).

## 5. State at shutdown (2026-10-13 ~06:00, session 5) - nothing is running; resume from here
- **DSpark teacher test: DONE.**  Teacher (FP8) states 0.437 accepted/pass vs ours 0.405 (`~/AI/v41-dev/dspark/
  accept_teacher.log`): our 2-bit drift is NOT the cause.  Production DSpark: SGLang accept length ~5 at bs=1 on
  V4-Pro, ~100% at bs=1 on V4-Flash (lmsys blog 2026-07-06; r/LocalLLaMA 1uqpers) => **our harness feeds DSpark
  wrongly**.  model.py's own DSpark loop is an untested smoke test ("nothing calls forward_spec").  Already ruled out
  in our harness: FP8 block size (32, honoured), RoPE/YaRN (compress ratio 0 = plain rope).  Teacher workdir
  `/media/Crucial1TB/v41-teacher`: `h/L00..L40.bin` (4 HC streams ENTERING each layer, 639 tokens = p256 prompt +
  383-token exact-routing generation), `pre/`, `tokens.npy`, `stats/head_argmax_L40.npy` (teacher greedy next token;
  teacher top-1 vs the text 0.857) - any main_hidden variant can be built on the CPU.
- **DSpark audit: SOLVED** (report `docs/ds4/V41_DSPARK_AUDIT.md`).  The harness drove DSpark exactly like
  SGLang / vLLM / model.py (main_hidden = HC-mean of the streams ENTERING 37/38/39; slot 0 = bonus token at p+1 + 4
  noise slots, drafts p+2..p+6; window = each block's own kv of main_x for accepted positions, non-causal block) - the
  bug was `load()`: checkpoint `mtp.N.attn.wo_a` is FP8 + 32x32 E8M0 scales but model.py builds it bf16, so the scale
  was dropped (~1000x too large).  Fixed in `tools/v41/dspark_accept.py` (default on; `--no-fp8-dequant` = old).
  Teacher states, 96 steps: **2.58 accepted (3.58 tokens/pass)**, accepted-length dist 0:23% 1:18% 2:10% 3:8% 4:8%
  5:32%, first draft 77% (78% vs teacher top-1).  The engine/GGUF never had the bug (our export dequantizes wo_a).
  **DONE 2026-10-13 afternoon: ours 3.27 accepted (4.27 tokens/pass) = teacher 3.24 -> no retraining; go to engine
  draft+verify.**  (Logs `~/AI/v41-dev/dspark/fixed_ours.log`, `fixed_teacher.log`; how they were run:)
  `python3 tools/v41/dspark_accept.py ~/AI/v41-dev/dspark/main_exact.bin ~/AI/v41-dev/dspark/gen_exact.txt
  --teacher-top1 /media/Crucial1TB/v41-teacher/stats/head_argmax_L40.npy` (OUR states - the number that sizes the
  engine work; our 37-39 states have cosine 0.876 to the teacher's) and the same with `--teacher
  /media/Crucial1TB/v41-teacher` for the full 378-step teacher number.  Also check the SIDECAR's `mtp.N.attn.wo_a`
  was exported with its block scale (`tools/v41/v41_sidecar.py`; its export check reported rel RMS 6e-3, so likely
  fine - verify before the engine uses it).
- **Vision slice 1 (encoder): DONE, committed** - `tools/v41/vision/ds41_vision.py` + `test_ds41_vision.py`; report
  `~/.quetza-data/conductor/reports/v41-vision-enc.md` (worker's gate PASS: per-row cosine >= 0.999 vs model.py on the
  original weights, 0.0 vs same-weights, protocol smoke as Strata's server drives it; CPU ~11 s/image quiet).
  **Conductor re-run of `python3 tools/v41/vision/test_ds41_vision.py` still pending** (needs a quiet CPU, ~8 min).
  Facts for slice 2: output file = int32 header {'SVE1', n, nx, ny, n_embd} + f32 [n][5120], `n` = the WHOLE span
  incl. delimiters, `nx=n ny=1`; the Strata engine keys image rows on Qwen's `kImagePad 248056` - V4.1 uses
  `image_token_id 129264`; `bias_vl` IS in our GGUF as `blk.N.exp_probs_b_vl.bias`; image tokens get no Engram and no
  attention-mask change; an image span must be prefilled in one chunk.
- **Dense requant (q4_K/q5_K): MEASURED, NOT ADOPTED (Mal: no code/agent quality drop; benchmark first).**  Forced-token
  speed (1 run each): `attn_router` q8 17.11 / q5_K 14.87 / q4_K 13.52 ms -> 26.64 / 27.73 / 29.75 tok/s.  Quality
  (`~/AI/v41-dev/dq/results.txt`, 4096-tok ppl + KL vs the arena-60 q8 logits): a q8 CONTROL at arena 40 already differs
  (code KL 0.0095 / agent KL 0.25, top-1 agreement 0.98 / 0.85 - routing flips on near-ties, accuracy unchanged) - so
  judge candidates against a same-arena control.  q4_K: code ppl +1.7% (2.393 -> 2.434), KL 0.026 (2.8x floor); agent
  ppl 14.87 -> 13.43 (more confident) with KL 0.48 (2x floor), accuracy 0.561 -> 0.559.  q4_K reason + the head-kept-Q8
  variant were cut off by the shutdown (`~/AI/v41-dev/dq/q4qual.sh`, resumable by editing the loop).
- **Folder cleanup done**: 12 old `~/AI/Strata-DS4-*` worker worktrees removed (branches kept in the repo; the wire
  worktree's 2 modified files = `git stash@{0}`), `~/AI/Strata-architectds` deleted (public clone).  Kept: Strata (its
  server is what serve_ds41.py runs), Strata-data, Strata-DS4-private, Strata-MiMo, Strata-GLM* (GLM's; GLM-rebase has
  382 uncommitted changes - Mal to decide), PolyStrata (GLM files in it).
- Repo/GitHub/HF: everything committed + pushed at shutdown (see `git log -5`); HF card has the 4-step quick start and
  the Engram placeholders `model-0004{7,8}-of-00048.safetensors.NEED-TO-DOWNLOAD-ENGRAM.txt`.

## 5b. Session 6 (2026-10-13 afternoon/evening) - READ THIS, it supersedes §5 where they differ
- **DSpark on OUR states (fixed harness, 378 steps): 3.27 accepted/pass = 4.27 tokens/pass** (= FP8 teacher 3.24) ->
  no retraining.  Logs `~/AI/v41-dev/dspark/fixed_ours.log`, `fixed_teacher.log`.
- **D1 DONE + committed `c0fe0e06`: `tools/v41/engine/ds41_dspark.{hpp,cpp}` (class Ds41DSpark) + `test_ds41_dspark.cpp`**.
  The draft stack on CUDA: sidecar weights (its `wo_a` verified correctly scaled, rel err 5.5e-3) + trunk
  `output.weight` / `token_embd` loaded itself (standalone: 8.61 GB VRAM incl. a 0.7 GB duplicate head).  API:
  `step(main_hidden[k*15360], pos[k], k, bonus, ids[5], conf[5])` appends k accepted rows to each block's 128-ring and,
  if bonus >= 0, drafts p+2..p+6 (cached graph per (k, draft); 4.9 ms/draft).  **Gate PASS** on the harness's own
  records (96 steps): engine 2.719 accepted vs harness 2.760; first draft identical 99.0% (slots 99.0/97.9/92.7/91.7/
  89.6%).  Run: `flock ~/.quetza-data/conductor/ds4-gpu.lock build-ds4-dspark/test_ds41_dspark --sidecar
  /media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.gguf -m "$G" --main ~/AI/v41-dev/dspark/main_exact.bin --gen
  ~/AI/v41-dev/dspark/gen_exact.txt --max 96 --dump F` (compare `dspark_accept.py ... --max 96 --dump-drafts F2`).
  Built in a SEPARATE tree `build-ds4-dspark` (same options as build-ds4-gpu minus `STRATA_DSV41_HC_MIX`).
- **Drafter precision cannot hurt output quality** (every draft is verified by the trunk) - only acceptance.  So the
  6.8 GB of MXFP4 DSpark experts can be compressed hard (Q4/Q2/hot-subset) or tiered; measure acceptance with
  test_ds41_dspark only.  Also share the trunk's head instead of loading a second copy (-0.7 GB).
- **D2 DESIGN (not coded yet) - verify/accept/rollback in `ds41_generate.cpp` one-shot path, greedy (temp 0) v1,
  `STRATA_DSPARK=<sidecar>`, `STRATA_DSPARK_DEPTH` (<= 4, see ring):**
  1. *Verify pass* = the existing `pass_m` lambda (ds41_generate.cpp ~513: one dense pass of m tokens, tier per token
     via `run_to_dev` into the routed rows), copied as `pass_v` that also captures main_hidden: after
     `dense.finish(l)` for l in {36,37,38} call `dense.read_state(x, pre)` (token-major [m][HC][D]) and accumulate the
     HC-mean into rows [m][3*D] (same math as the STRATA_DUMP_MAIN block ~490).  Logits: `dense.logits(row0, nrows, out)`.
     Dense `max_pass` = S (512 with the default layer-major prompt batch) so m <= 6 fits.
  2. *Loop*: state P (last fed position), t (trunk token for P+1, not yet fed), drafts d[0..k).  Emit t; pass tokens
     [t, d0..d(k-1)] at P+1..P+1+k; g_i = argmax(row of position P+1+i); a = #leading d_i == g_i; emit d0..d(a-1);
     t' = g_a; DSpark.step(main_hidden rows of P+1..P+1+a, a+1 rows, bonus t') -> new drafts; P += 1+a.
  3. *Rollback is free except two rings*: every cache is position-indexed and simply overwritten when a position is
     fed again (compressed rows, indexer keys, Engram history, sel/cand per column).  (a) the window K ring has
     RAW = SWA+3 = 131 slots (`ds41_dense.cpp` init `im.RAW = im.SWA + 3`, slot = pos % RAW): a rejected position q
     evicts q-131, which a later query at q' needs iff q >= q'+4 -> **safe for depth <= 4**; depth 5 needs RAW >= SWA+1+5
     (one-line change in ds41_dense.cpp, bit-identical by construction - gather by position; do it when the fusion
     slice is done with that file).  (b) the **ratio-2 compressor state ring is 2 slots by parity** (`i_sidx`/`i_sslot`,
     ds41_dense.cpp ~1112): a pass writes only its LAST 2 tokens' states, so after a partial rejection the slot of the
     last accepted position may hold a rejected token's state.  FIX (no engine change): if the pass would start at an
     ODD position and the previous pass rejected anything, start it ONE position earlier, re-feeding the last accepted
     token (its group's states then come from inside the pass, `row(q) = 2 + q - pos0`).
  4. *Seeding the DSpark window*: the layer-major prefill gives no per-position states, so feed the prompt minus its
     last min(128, L-1) tokens normally, open the VRAM cache, then run that tail through `pass_v` in chunks of <= 16
     (set `Ds41DSparkConfig::max_append = 16`), appending the main_hidden rows (bonus -1), the last chunk drafting.
  5. *Gate*: spec output vs plain greedy (`STRATA_DSPARK` unset) - identical up to batch-shape near-ties (report the
     first divergence index + ppl); tokens/pass in-engine vs 4.27 offline; tok/s.
  6. *Known speed ceiling of v1*: `pass_m` runs the tier PER TOKEN (~16 ms each) and `Ds41MoeTier::run_multi` falls back
     token-by-token for per-expert-geometry models (ours), so a 5-token verify costs ~5x tier.  **D3 = a multi-token
     tier path** (union of the pass's experts, each expert's weights read once for all its tokens - `native_expert_grouped`
     already takes `ent_tok`; the prefill `run_chunk_dev` groups by expert but streams whole layers): that is what
     decides 50 tok/s.  Union of 5 tokens' experts also raises misses: keep route bias on (it pulls the union into VRAM).
- **Kernel fusion slice DONE + committed `86724531`** (worker closed): fused HC-mix op `GGML_OP_DSV41_HC_MIXES`
  (`tools/v41/ggml/dsv41-fuse.patch`, APPLIED in the local third_party tree; opt-in `-DSTRATA_DSV41_HC_MIX=ON`, which
  `build-ds4-gpu` is configured with).  Bit-exact (test_ds41_hcmix, forced tokens 0-diff, logits md5 equal); -546
  kernels/token (-14.9% of the dense graph), ~0.5 ms/token - timed A/B within noise (26.42/25.89 vs 25.91).  Next-best
  fusions it listed: router tail softplus/sqrt/+bias (~0.3 ms), the shared expert's clamps block ggml's
  MUL_MAT+MUL_MAT+GLU fusion, F16 window ring to drop the f32->f16 cpy.  Prefill md5 with the op: not re-checked.
- **Dense requant (forced A/B, 1 run each): q8 26.64 / q5_K 27.73 / q4_K 29.75 tok/s** (attn_router 17.11 / 14.87 /
  13.52).  q4_K quality vs a same-arena q8 control: code ppl +1.7% (KL 0.026 vs floor 0.0095), agent KL 0.48 vs floor
  0.25.  NOT adopted - Mal: quality benchmark first (§6 item 3).
- **llama.cpp fixes to pull (queued slice, after fusion)**: #30103 (flash-attn tile race, our D=512 prompt attention;
  maybe the cause of the prefill md5 drift) + #29953 (MMQ OOB reads) - our vendored ggml predates both.
- Strata 0.1.41-0.1.43.1 scanned: its decode-collapse bug does not apply (no pinned thread / host callbacks in our
  engine); useful later: `tools/strata_bench.py` speed suite, pinned ring for pageable H2D, CPU share for short prompts.
- Cleanup done: 12 old worker worktrees removed (branches kept; wire leftovers = `git stash@{0}`), Strata-architectds
  deleted.  Swap is not a lever (8+32 GB swap files on the SATA root SSD ~0.5 GB/s; zram = compressed RAM; the arena is
  pinned).  Mal's mode: **the conductor writes the core code itself; the one flash worker is for research/legwork.**

## 6. Work queue (Mal's order; ONE flash worker at a time)
1. **DSpark on OUR states: DONE - 4.27 tokens/pass, no retraining.**  => item 4 (engine draft+verify) is the top
   priority after the in-flight fusion slice.  50 tok/s needs a verify pass <= ~85 ms at depth 5 (or <= ~53 ms at depth 2).
2. **Kernel fusion of the elementwise zoo** (packet ready: `~/.quetza-data/conductor/packets/v41-zoo-fusion-task.md`):
   ~3,300 tiny kernels/token = 5.45 ms; bit-exact fused ops (forced-token 0-diff is the gate); expect -1.5..-3 ms (+4-8%).
2b. **ggml upstream fixes** (after fusion; it owns the ggml tree): our vendored ggml predates llama.cpp #30103 (race in
   `flash_attn_ext_f16` tile combine: `fattn-mma-f16.cuh:1747` `if (np > 1)` -> `if (np > 1 || nbatch_combine != DKQ/2)`;
   our prompt attention is D=512) and #29953 (MMQ out-of-bounds reads, `mmq.cu:129` fallback).  Ship as
   `tools/v41/ggml/upstream-fixes.patch`; gate = forced decode 0-diff + does the 8K prefill `--dump-last` md5 become
   reproducible across runs with different free VRAM (the FA race may be the real cause of today's md5 drift).
3. **Quality benchmark baseline** (Mal: decide every lossy change against it): (a) DwarfStar's official V4.1 fixtures
   `~/AI/ds4-ref/gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router/` (112 prompts with official API
   continuations + top-20 logprobs; teacher-forced NLL / API top-1 agreement / top-20 recall; comparable to DwarfStar's
   Q2 = 90.1% top-1) - score exact routing AND the default decode biases (`--bias-prompt` decode-routing proxy);
   (b) Mal's agentic harness bench `~/AI/quetza-workspace/model-gauntlet/harness-bench/` (real QuetzaCodetl runs,
   hidden-graded Scheme/JSON tasks; add a `strata-ds41` backend); (c) speed side: Strata 0.1.43's
   `tools/strata_bench.py` suite (short chat, 4K/16K prompts, five-turn agent session) against our server (port 8150).  Note: the decode route bias (code ppl 2.38 -> 2.87)
   is a bigger quality lever than any dense quant - benchmark the configs Mal would actually choose between.
4. **DSpark engine draft + verify** (after 1): DSpark stack as a ggml graph (resident dense, experts resident/tiered),
   main_hidden in-graph, verify pass of 1+k tokens (dense `pass_m` + tier `run_multi`), KV rollback, depth adaptive by
   the confidence head; port the shape from the V4 engine's `--mtp/--verify` (`tools/ds4`).  Then **n-gram /
   prompt-lookup drafting** on the same verify path.  50 tok/s = 20 ms/token -> needs ~2+ accepted per pass.
5. **Vision engine + serve** (slice 2): `GENI` for V4.1 (facts in §5), `VRAM <mib>` + `--vram-elastic` in `--serve` so
   Strata's server can run the encoder LAZILY (Mal's wish: not resident; start on the first uncached image, the engine
   lends ~1-2 GB of expert slots while it encodes, stop after `idle_s`, cache grows back - the server side exists:
   `start_lazy_vision`, `Engine.vram()` in `~/AI/Strata/serve/server.py`); config `"vision": {..., "lazy": true}`;
   CPU-encode fallback; gate = a synthetic image with a known word + coloured shape answered correctly.  Then make
   `serve_ds41.py` portable (it hard-codes `~/AI/Strata`).
6. Later: NVMe-miss work (reads from the existing top-16 prediction on an I/O thread; frequency admission), dense requant
   decision after the benchmark, faster ternary expert kernel.

## 7. How to measure (mandatory)
- **Correctness bar = forced-token identity**: `STRATA_FORCE_IDS=~/AI/v41-dev/dec7/force256.tok` makes every run
  decode the same 256 tokens; diff the emitted ids vs the file (skip the non-numeric memguard banner line) = 0 lines.
  Numerics gate: `build-ds4-gpu/test_ds41_moe_gpu "$G" --layers 0,20,32,39 --gpu --tokens 8` (worst rel ~1.44e-2).
  Anything that moves bits: 4096-token ppl + KL vs `~/AI/v41-dev/dq/*_q8.logits` (code/agent/reason).
- **Speed A/B**: `~/AI/v41-dev/k143/ab.sh TAG [ENV=VAL ...]` (p256 x 256, `--slots 1500`, phase profile; appends to
  `k143/results.txt`; `BIN=` selects a binary).  Noise floor ~3-5% (worse on a busy box): >= 2 runs per arm, quiet box,
  never beside builds / torch jobs / the HF upload.
- **nsys**: count calls and use `~/AI/v41-dev/nsys/calls.py` / `idle_attrib.py` (they include
  `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` now - without it the dense half is invisible); `--cuda-graph-trace=node` to see
  inside the dense graph (slows decode ~50%: ratios only).  Tokens in the 4.5 s window = graph launches, not "96".
- Prefill exactness md5 (`--dump-last`) is NOT reproducible right now (MMQ grouping depends on free VRAM); decode is.

## 8. Rules (Mal)
- Kill by PID only.  Never `pkill -f`/`pgrep -f` a pattern that can match your own command line - that includes a
  `case *x*` glob over `/proc/*/cmdline` (matches your own shell; match the EXACT cmdline); never `tmux kill-server`.
- Never run a full-model engine beside a worker's ~16 GB CPU torch job: RAM runs out and memguard kills the engine.  Never edit a running bash script in place (write a copy, `mv`).
- GPU runs: `STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 86 72 -- <cmd>` (env vars INSIDE as `env X=1 cmd`
  or before `bash memguard.sh` as ab.sh does); wait for MemAvailable >= 75 GiB; one GPU job at a time; it takes
  `~/.quetza-data/conductor/ds4-gpu.lock`.  Never a CPU-only full-model load (froze the box once).
- Don't rebuild `build-ds4-gpu` while a run or the wrapper uses it.  `third_party/llama.cpp` is untracked: ggml
  changes ship as patch files in `tools/v41/ggml/` (fake-quant + rank-argsort patches are APPLIED in the local tree;
  `/usr/bin/patch`, anaconda's is broken).
- Measure before claiming; report ppl/KL for anything lossy; **Mal decides quality trades**; soft routing only
  (penalties, never masks).  Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Conductor mode: **ONE flash worker at a time** (Mal), no Agent-tool subagents; **the conductor writes the core code
  itself - the worker is for research/legwork** (Mal 2026-10-13); packets in
  `~/.quetza-data/conductor/packets/<slice>-task.md`, dispatch with `qc-dispatch <slice> --model flash --prompt "Read
  the full task packet at ...-task.md and follow it exactly ..."` (`--packet` stalls; `--prompt` overwrites
  `packets/<slice>.md`), arm `qc-await <slice>`.
- NSFW content is headless-only on this PC (never display images); vision tests use synthetic images.
- "SSD NVME" does not auto-mount after a reboot: `udisksctl mount -b /dev/nvme0n1p3` (must be ntfs3).

## 9. Paths
| what | where |
|---|---|
| model | `/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf` (+ `rank.bin`, `tokenizer/`, `V41-GSQ.gguf.experts.json`) |
| Engram tables | `--engram "/media/Crucial1TB/V41-Engram:/media/mal/SSD NVME/AI/V41-Engram"` (L14 : L1) |
| DSpark + vision sidecar | `/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.gguf` (mtp.0-2 MXFP4 + Q8_0; vision.*, aligner.*, image_*) |
| FP8 source checkpoint | `/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/` (shards 1-46; 47/48 = Engram on NVME1TB) |
| DeepSeek reference code | `~/AI/gsq-rco/dsv41-inference/` (model.py incl. DSpark + vision, vision.py, image_processor.py) |
| DwarfStar (antirez ds4, C reference) | `~/AI/ds4-ref` |
| build | `build-ds4-gpu/` (configure line in README "Running the V4.1 engine" + `-DSTRATA_DSV41_FQ_OP=ON`) |
| bench / scratch | `~/AI/v41-dev/` (`k143/` A/B, `dec7/force256.tok`, `r1ab-1237/p256.i32`, `nsys/`, `dspark/`, `dq/`, `eval/*.i32`) |
| serve | `~/.local/bin/strata-ds41-quetza` (port 8150, 512K) -> `tools/v41/serve/{serve_ds41.py,ds41-engine.sh,strata-ds41.json}`; logs `/tmp/strata-ds41.log`, text `/tmp/strata-ds41-text.log` |

Engine files: `ds41_dense.{hpp,cpp}` (dense half: attention, indexer, router, HC, Engram add, snapshot, `pass_m`),
`ds41_moe.{hpp,cpp}` (expert tier: VRAM LRU / arena / file tier / CPU pool / prefetch / grouped launches),
`ds41_generate.cpp` (CLI + `--serve` protocol + decode loop), `ds41_engram.*`, `ds41_ops.hpp` (act-quant rounding),
`ds41_geometry.hpp` (tensor map), `ds41_transcode.cu` (PTQ1_0 -> Q2_0 for MMQ).  Kernels: `src/kernels/cuda/fork/*`,
`src/kernels/cuda/iq_kernels.cu` (`native_expert_grouped`, `decode_rows_sum`).

## 10. Where the details live (open only for a specific question)
- `docs/ds4/V41_RESUME_PROMPT.md` - the full engineering log, newest first (~150 KB: history, every A/B).
- `docs/ds4/V41_DECODE_OVERLAP_PLAN.md` §5-§8 - the decode attribution and why each host-side lever failed.
- `docs/ds4/V41_PROGRESS.md` - the public step-by-step speed history.  `README.md` - public results.
- `~/.quetza-data/conductor/reports/*.md` - worker reports (`attn-node-profile`, `moe-miss-research` = literature on
  miss reduction, `dec-scout`, `gds-scout`, ...).

## 11. Paste-able resume prompt
> Continue the DeepSeek-V4.1-Flash Strata-DSV4.1 work for Mal in `~/AI/Strata-DS4` (branch `deepseek4`).  Read ONLY
> `docs/ds4/V41_START_HERE.md` first - it is complete; open other docs only for a specific question.  Goal: 50+ tok/s
> decode (DSpark retrain + engine draft/verify + n-gram drafting + plain-decode kernel work) and vision.  Check §5
> (in flight) first, then work §6 in order, measuring per §7 and following §8.  Conductor mode, one flash worker.
> Update §2/§5/§6 of START_HERE when something lands (keep it short; history goes in V41_RESUME_PROMPT.md).
