# Strata-DSV4.1 — START HERE

**A new session reads THIS file and nothing else before starting work.**  Everything below is current and measured.
Open the long files only for a specific question (see "Where the details live").  Last updated 2026-10-13 ~05:20.

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
| DSpark (offline, our trunk states) | 0.73 accepted drafts/pass (biased routing), 0.41 (exact) |
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
quarter-block ternary dot / `STRATA_EXP_MODE_T*` (~1%) · dense requant q5_K/q6_K as a SPEED lever (unforced runs
showed no gain - unexplained, see §6 item 3c; it does free VRAM) · hard pruning (soft penalties only - Mal).

## 5. In flight (2026-10-13 morning)
- **DSpark teacher test**: `~/AI/v41-dev/dspark/teacher.sh` + `followup.sh` (log `teacher.log`).  FP8 teacher states
  for the same 639 tokens in `/media/Crucial1TB/v41-teacher/h/L37-39.bin`; results `accept_teacher.log` (teacher
  states), `accept_exact_t1.log` / `accept_teacher_t1.log` (+ first draft vs the teacher's own argmax, via new flags
  `v41_calib.py head --save-argmax` and `dspark_accept.py --teacher-top1`, uncommitted until this run proves them).
- **Vision slice 1** (conductor worker `v41-vision-enc`, packet `~/.quetza-data/conductor/packets/v41-vision-enc-task.md`,
  report `~/.quetza-data/conductor/reports/v41-vision-enc.md`): `tools/v41/vision/ds41_vision.py`, a torch encoder
  process that is a drop-in for Strata server's `strata-vision` protocol, sidecar weights, gated vs model.py.

## 6. Work queue (in order unless Mal says otherwise)
1. **DSpark**: (a) read the teacher result - teacher states ~2-3 accepted => our state drift is the cause => (b)
   **retrain** DSpark's input projection `main_proj` (15360->5120) + `main_norm` (+ light adapters if needed) on OUR
   engine's states/tokens (self-distillation; data = engine prefill dumps of layer-37-39 HC-mean states + the trunk's
   top-k next tokens; MoE experts frozen; harness = `tools/v41/dspark_accept.py`, which runs DeepSeek's DSpark modules
   on our torch kernels); (c) **engine draft + verify**: DSpark stack as a ggml graph (resident dense, experts resident
   or tiered), main_hidden captured in-graph, verify pass of 1+k tokens (dense passes `pass_m` + tier `run_multi`),
   KV rollback, draft depth adaptive by DSpark's confidence head; port the shape from the V4 engine's `--mtp/--verify`
   (`tools/ds4`).  Teacher states also ~0.5 => DSpark is weak here; lean on (d) instead.
   (d) **n-gram / prompt-lookup drafting** on the same verify path (free drafts; agent sessions re-emit context).
2. **Vision**: (a) encoder process [in flight]; (b) engine `GENI`: image-span embeddings overwrite the prompt rows
   (`[IMAGE_START] + ([IMAGE]*w + [NEWLINE])*h + [IMAGE_END]`, all positions carry `image_token_id` 129264), router
   `bias_vl` on image positions, image tokens excluded from Engram n-grams, anything else model.py does with
   `image_mask`; gate = a synthetic image with a known word + coloured shape answered correctly; (c) serve shim image
   input + template; (d) `serve_ds41.py` hard-codes `~/AI/Strata` - make it portable for outside users.
3. **Plain decode** (kernel work), each A/B'd with forced tokens: (a) **fuse the elementwise zoo** (5.45 ms; bit-exact
   fused ops as opt-in patches under `tools/v41/ggml/`, like the fake-quant op that gave +5%); (b) **NVMe misses**:
   issue reads from the top-16 `finish_predict` already computes, on a dedicated I/O thread (no extra graph);
   frequency/utility cache admission instead of pure LRU; smaller blobs for the rarely-used file tier; (c) re-measure
   q5_K/q4_K dense with forced tokens + the phase timer (did `attn_router` drop?); (d) faster ternary expert kernel.

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
- Kill by PID only.  Never `pkill -f`/`pgrep -f` a pattern that can match your own command line; never
  `tmux kill-server`.  Never edit a running bash script in place (write a copy, `mv`).
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
