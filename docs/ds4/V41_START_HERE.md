# Strata-DSV4.1 — START HERE

**A new session reads THIS file and nothing else before starting work.**  Everything below is current and measured.
Open the long files only for a specific question (see "Where the details live").  Last updated 2026-10-13 ~06:00 (end of session 5).

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
| DSpark (offline harness, FIXED 2026-10-13) | **2.58 accepted drafts/pass = 3.58 tokens/pass on teacher states** (96 steps; first draft 77%, confidence head calibrated 12/40/78/99%). Old 0.4 was a HARNESS weight-loading bug (`wo_a` FP8 block scale dropped). On OUR engine states: not yet measured with the fix |
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
  **First thing next session (CPU, ~15 min each, never beside an engine run):**
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

## 6. Work queue (Mal's order; ONE flash worker at a time)
1. **DSpark on OUR states** (the two runs in §5, CPU).  If ours is close to the teacher's 2.58 -> go straight to item 4
   (engine draft+verify), no retraining.  Only if ours << teacher: retrain `main_proj` (15360->5120) on our states.
2. **Kernel fusion of the elementwise zoo** (packet ready: `~/.quetza-data/conductor/packets/v41-zoo-fusion-task.md`):
   ~3,300 tiny kernels/token = 5.45 ms; bit-exact fused ops (forced-token 0-diff is the gate); expect -1.5..-3 ms (+4-8%).
3. **Quality benchmark baseline** (Mal: decide every lossy change against it): (a) DwarfStar's official V4.1 fixtures
   `~/AI/ds4-ref/gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router/` (112 prompts with official API
   continuations + top-20 logprobs; teacher-forced NLL / API top-1 agreement / top-20 recall; comparable to DwarfStar's
   Q2 = 90.1% top-1) - score exact routing AND the default decode biases (`--bias-prompt` decode-routing proxy);
   (b) Mal's agentic harness bench `~/AI/quetza-workspace/model-gauntlet/harness-bench/` (real QuetzaCodetl runs,
   hidden-graded Scheme/JSON tasks; add a `strata-ds41` backend).  Note: the decode route bias (code ppl 2.38 -> 2.87)
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
- Conductor mode: **ONE flash worker at a time** (Mal), no Agent-tool subagents; packets in
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
