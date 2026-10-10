# DeepSeek-V4.1-Flash: GSQ-RCO quant + Strata-DS4 port — HANDOFF / RESUME

**Last updated:** 2026-10-11 ~07:20 by `strata-ds4-gpu` (Claude).  **READ §HANDOFF 2026-10-11 FIRST** - everything
below it is history (still accurate unless this section says otherwise).

## HANDOFF 2026-10-11 07:20

### 1. State right now
- **GSQ quant pass running** (step 4/4 of `v41allday`, chain pid in /media/Crucial1TB/v41-calib/allday.pid): layers 0-14
  written into `/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf` (152 GiB, single file, kernel ntfs3); layer 15 started
  07:07; ~12.5 min/layer -> **ETA ~12:20**, then its own head gate on the quantized chain
  (`/media/Crucial1TB/v41-quant-mixed/stats/head_L40.json`).  Check: `v41allday status`, `tail
  /media/Crucial1TB/v41-quant-mixed/quant.log`.  Pause: `v41allday stop` (after the current layer); resume
  `v41allday start` (resumes at the first unfinished layer).  The model is unusable until all 40 layers are written
  (unwritten layers are zero experts).
- **Calibration DONE** 01:29.  Teacher head gate (`/media/Crucial1TB/v41-calib/stats/head_L40.json`, held-out
  49,146 tokens): **ALL/eval ppl 4.915, top-1 0.701**; per domain eval: code 1.739, agent 2.121, reason 5.832,
  zh 6.831, agentic_long 9.793.  This is the bar for the GSQ file.
- **Tier map** `/media/Crucial1TB/v41-calib/alloc_full.json` (v41_alloc --per-proj): budget = all-Q2_0 bytes (142.4 GiB
  experts), proxy error 1.882 vs all-Q2_0 2.269 (-17%).  Early layers get Q3_K (L4-8 ~2.45-2.52 bpw), L33-39 ~2.0.
  9 (gu,down) format classes model-wide, up to 7-8 per layer.
- **Night fixes (all committed, all in the running chain):** L7 OOM (1.88 GiB fp32 torch.cat of the propagated output
  -> chunked bf16, `bf23f296`); L13 crash in `q3k_split` (all-zero super-block whose fp16 d did not round-trip ->
  search/re-round + log, `2989e535`); L14 slow (Engram table on Team CX2: slow drive + ntfs-3g FUSE, 1h40).
- **Engram shards copied to NVMe** (Mal OK'd 2026-10-11): `/media/mal/NVME1TB/Models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/`
  shard 48 (L14) DONE + verified (size + 302 sampled MiB) + `.ok` + `model.safetensors.index.json`; **shard 47 (L1)
  copy started 07:15** (`/home/mal/AI/v41-dev/copy47.log`, `.partial`): when dd says "records out", verify the same
  way (size + sampled blocks vs Crucible), `mv .partial` -> final name, `touch .ok`.  Engine config
  (`tools/v41/serve/strata-ds41.json`), `gpu_bringup.sh` and `v41_calib.DIRS` list NVME1TB FIRST.
- **PLAN for the running chain (agreed with Mal 07:15): do NOT stop/restart it for the Engram move.**  Only layers 1 and
  14 read Engram and both are already written; layers 15-39 and the final head gate never read it, so a restart would
  only redo a layer.  The NVMe Engram copies are for the ENGINE (decode reads 48 rows/token).  Let the chain finish
  (~12:20), then: (a) finish/verify the shard 47 copy, (b) Mal remounts the ntfs-3g drives with ntfs3 (commands below),
  (c) GPU bring-up.  Only if a layer must be RE-quantized later (e.g. a re-export) does the order matter: the calib/quant
  code already reads Engram from NVME1TB first (v41_calib.DIRS).
- **Drives:** Crucial1TB, Crucible, Team CX2 are `fuseblk` (ntfs-3g, slow random reads).  Suggested to Mal (not done,
  needs sudo): Crucible/Team CX2 -> `ntfs3` read-only (zero risk), Crucial1TB -> ntfs3 rw after the chain (or keep).
  Never `force`-mount a dirty volume.  With both Engram shards on NVME1TB this matters less for the engine.
- **Housekeeping done 07:05:** deleted `/media/mal/NVME1TB/syv-qwen` (vLLM Qwen3.8 repo + 23 GB models) and the wrappers
  `qwen3.8-quetza-vllm`, `qwen3.8-swift-quetza-vllm` (+ .bak) - Mal: moved to NInfer.  Gateway profiles untouched.

### 2. The engine (tools/v41/engine/, all committed; GPU NEVER RUN YET)
Build trees: `build-ds4` (CPU, ggml from FetchContent pin) and `build-ds4-gpu` (CUDA, STRATA_GGML_DIR=third_party/llama.cpp
- gitignored, NOT tracked: never add ops there; ds41_ops.hpp builds the rounding from stock ops for that reason).
CMake: `tools/v41/engine/v41_engine.cmake` (included at the end of tools/ds4/cmake/fork.cmake).
- `ds41_geometry.hpp` - keys + full tensor map + per-expert tier maps; full-length GGUF arrays (strata::GgufFile keeps
  64 items).  `test_ds41_geometry FILE` -> PASS on the real file.
- `ds41_dense.{hpp,cpp}` - dense half.  Pass = n tokens at pos0 (decode n=1 cached graphs w/ CUDA-graph reuse; prefill
  n=chunk built per pass).  Single-pass HC (pre_state carries the next attention's pre-mix); window ring [ring|pass];
  compressed KV on kv sources (2,8,14,20: ratio 2 pooled / 1 plain) read by later same-ratio layers; indexer (fused
  ggml_lightning_indexer) on index sources (2,8,14,20,24,28,32,36), top-k reused until the next; L20 candidate blocks
  (pool_2d max, newest block pinned) masking L24+; attention = flash-attn with the QUERY AS BATCH (each query its own
  128 window + top-k rows; pad keys to 256 off-CPU), sinks; Engram rows via Ds41Engram at L1/L14; act-quant rounding;
  KV BF16 (lossless with act-quant); snapshot/restore (rings, ~11 MB) for multi-turn; predict() router hint; read_sel tap.
- `ds41_ops.hpp` - FP8/pow2, FP4/E4M3, FP4/pow2 fake-quant from stock ggml ops (Veltkamp + magic-number RNE).
- `ds41_engram.{hpp,cpp}` - hash from GGUF arrays, parallel preads (fork-join pool) of 256-B rows + 8-B scales.
- `ds41_moe.{hpp,cpp}` - FORK of tools/ds4/ds4_moe with per-expert format CLASSES (Ds41MoeGeom::eclass, cls_gu, cls_d):
  per-class CPU pool runs (Q2_0 classes need ActQ: im.actq), per-class grouped GPU launches, VRAM LRU swaps within
  (layer, class), chunk path = one pass per class + one weighted sum, run_multi token-by-token for per-expert geoms,
  `skip_file` option.  `Ds41FileBlobs` = an expert's blob from the 3 per-type tensors (preads).  `ds41_read_ranked`.
- `ds41_generate.cpp` - CLI (`--ids/--ids-file -n --ppl --dump-logits`), `--serve` (Strata engine line protocol: GEN/
  STOP/QUIT -> READY/RESUME/PP/T/DONE/ERR; snapshot reuse), flags `--backend cuda|cpu --experts gpu|cpu --engram D:D
  --rank r.bin --slots auto|N --vram-margin --arena-gib --vram-lru --arena-adapt --pcie --pf-b --prefill-chunk N
  --chunk-mmq --skip-miss --skip-file --ctx --temp --stop --no-actquant`.
- Tests: `test_ds41_dense` (vs golden), `test_ds41_ops [--cuda]`, `test_ds41_moe[_gpu] FILE --layers .. [--gpu]`,
  `ds41_golden.py` (model.py on the GGUF's weights, `--no-actquant --zero-experts`, env DS41_TEST_TOPK /
  DS41_TEST_CAND_BLOCKS on both sides), `compare_ds41.py`.
- Serving: `tools/v41/serve/serve_ds41.py` (+40 tests), `ds41-engine.sh`, `strata-ds41.json` (port 8150, ctx 524288,
  chunk 1024, arena 64, vram-lru, rank file), `strata-ds41-quetza.wrapper` (NOT installed in ~/.local/bin until the
  engine runs).  Tokenizer: `/media/mal/SSD NVME/AI/V41-GSQ/tokenizer`.  Rank: `/media/mal/SSD NVME/AI/V41-GSQ/rank.bin`
  (v41_rank.py: calibration count per byte).
- Shared-file fork edits (one-liners, documented): gguf_reader.hpp (PTQ1_0 143), native_expert.hpp (kNativeActBytes
  5840, kNativeHBytes 2628, kNativeFFMax 2304), expert.hpp (ActQ kActMax 5120).

### 3. Gates passed (CPU)
- 3-layer file: block outputs cos >= 0.99996 vs model.py (21 tok), >= 0.99983 (1688 tok, top-512 of 844); decode ==
  chunked prefill; act-quant rounding bit-identical to torch (2M values); BF16 KV bit-identical to F32.
- 40-layer dense-only file (`/home/mal/AI/v41-dev/v41-all-h.gguf`, zero experts, 300 tok): real top-k: cos >= 0.996
  every layer, logits mean 0.99995, top-1 295/300; forced top-32/8 blocks: selected sets identical 272/300 at L2, mean
  Jaccard 0.98-0.994 at all index layers (boundary ties from ggml F16 activations); decode == prefill 300/300.
- Tier on the REAL L0 (all 7 classes mixed per token): rel 1.2-1.5e-2 vs dequant+F32 (the Q8 activation rounding) PASS.
- Dropping act-quant moves model.py's own outputs cos 1.6e-3 / 16% routing sets -> kept on (measure ppl on/off on GPU).

### 4. NEXT (in order) - when the GSQ pass is DONE (~12:20) and Mal OKs the GPU
1. Finish/verify the shard 47 copy (above).  Compare the GSQ head gate (v41-quant-mixed/stats/head_L40.json) with the
   teacher's 4.915 - report per domain to Mal plainly.
2. `bash tools/v41/engine/gpu_bringup.sh` (RANK=/media/mal/SSD\ NVME/AI/V41-GSQ/rank.bin optional): CUDA rounding
   bit-exact -> tier on real L0/20/39 with --gpu -> ds41_generate --ppl on 2048 held-out code tokens (chunk 512).
   Expect issues on first GPU contact (graph reuse, FA with ne3 batch, top_k sizes, lightning-indexer shapes): fix with
   the CPU tree as oracle (`--backend cpu`).
3. Engine ppl on the 6 held-out eval sequences (`/home/mal/AI/v41-dev/eval/seq*.i32`, 8192 tok each) vs the GSQ head
   gate: the engine must reproduce it (same weights; engine has Q8_0 dense vs FP8 + Q8 activations - small diff).
   Also act-quant on/off ppl.
4. Speed (Mal's target: beat GLM's 25 tok/s decode / 400 tok/s prefill, 512K ctx).  Routing is flat: rank shows 16 GiB
   of experts = 42% of lookups, 64 GiB = 78%, 80 GiB = 86% -> expect ~8-10 tok/s at first.  Levers, measure each:
   arena size (84 GiB memguard cap), vram-lru, arena-adapt, pf-b prefetch (Ds41Dense::predict), per-class launch
   count (up to ~4 per token-layer: fuse into one launch), merge the 3 readbacks per layer, Engram read time/token,
   prefill chunk 2048-8192 + chunk-mmq (Q3_K only), **soft-prune** = route-bias penalty on the coldest experts
   (GLM's recipe, not yet in ds41: port set_route_bias / --prune-penalty from tools/ds4), skip-file.
5. Size (Mal asked: Maya-size ~90 GiB?).  `/home/mal/AI/v41-dev/size_sweep.py`: prune+mixed proxy error 0.54 at
   89 GiB vs current mix 1.88 / all-Q2 2.17 - BUT pruning cost is only measured on the calibration corpus (OOD
   risk).  Recommendation given: keep the 152 file, soft-prune at runtime, decide on a smaller re-export after real
   ppl/tok/s (new tier map = minutes on CPU; a new GSQ pass = ~8 h GPU).
6. Serving: install `strata-ds41-quetza` (copy wrapper to ~/.local/bin) once `--serve` works; test via serve_ds41
   (OpenAI + Anthropic, tool calls), coherence, agentic tasks, loop test, DwarfStar's official V4.1 fixtures.
7. Later: DSpark/MTP (mtp.* not exported yet), 512K long-context run, upstream of the Engram/indexer work.

### 5. Rules (Mal)
Mal decides GPU use (strata-glm shares it: lock ~/.quetza-data/conductor/ds4-gpu.lock).  Never push without Mal's OK
(nothing pushed).  Commit each verified step with the Co-Authored-By trailer.  Never `pkill -f` a pattern that can match
your own command line (it killed this session's shell once): use PIDs.  Never edit a running script in place
(.new + mv).  CPU work nice 15; big RAM runs via tools/ds4/memguard.sh (it takes the GPU lock - not while the chain
runs).  Keep HF shards 47/48 on Crucible/Team CX2 as backups.  NSFW is headless-only (CLAUDE.md).  Explain plainly.


**Last updated:** 2026-10-10 ~09:05 by `strata-ds4-gpu`.

**2026-10-10 NIGHT (engine port, by strata-ds4-gpu; Mal asleep, chain running): READ §Engine (night of 10-10) FIRST.**
Chain: calibration resumed 22:28 at layer 25 (`v41allday start`), then tier map -> layout -> GSQ pass (~07:00 est).
Engine (tools/v41/engine/, all committed): geometry/loader, Ds41Dense (CPU-gated vs model.py), act-quant ops,
Ds41MoeTier (per-expert format classes), Ds41FileBlobs, ds41_generate (CLI/--ppl/--serve).  GPU runs NOT done yet
(GPU owned by the chain).  Next: §Engine next steps.

**09:04 UPDATE (supersedes 09:01):** the WHOLE chain runs unattended: `v41allday` (tools/v41/all_day.sh + alldayctl.sh,
start|status|stop|log): calibration 7-39 + head -> v41_alloc tier map -> GGUF layout on SSD NVME/AI/V41-GSQ -> GSQ quant pass
(mixed, in place) + its head gate.  All-Q2_0 control SKIPPED (Mal).  Resumable: rerun `v41allday start` after a failure/stop.
Log /media/Crucial1TB/v41-calib/allday.log.

(old) **09:01:** calibration is ARMED, not stopped: `v41calib start-when-idle` (driver waits until the GPU lock is free +
VRAM <3 GB for 15 min straight, i.e. strata-glm finished, then runs layers 7-39 + head unattended; CUDA OOM retries a layer
2x). Check `v41calib status` / calib.log before doing anything; do NOT `start` again while it shows RUNNING.
**Read order:** this file → [V41_PLAN.md](V41_PLAN.md) (design, V4→V4.1 deltas, all research) →
[V41_FEASIBILITY.md](V41_FEASIBILITY.md) (sizes, speed estimates) → [RESEARCH_2026-10-09.md](RESEARCH_2026-10-09.md)
(community ideas) → `../../V41_PLAN`-adjacent [../../RESEARCH](../RESEARCH_2026-10-09.md).
**Repo:** `~/AI/Strata-DS4`, branch `deepseek4`. Local only — ahead of the public `ds4/main` at `924516a`.
**Since 2026-10-10 `deepseek4` contains upstream Strata (merge `cc55d4c9`, upstream `fb58e0db`)**; pulling upstream is a
plain `git merge upstream/main` — read [../UPSTREAM_SYNC.md](../UPSTREAM_SYNC.md) (process, gates, fork-isolation rules)
before touching any file under `src/` or `include/`.
**Nothing from this work is pushed;** push only when Mal says.

---

## Paste-able resume prompt

> You are continuing the **DeepSeek-V4.1-Flash GSQ-RCO mixed-tier quant + Strata-DS4 port** for Mal, in
> `~/AI/Strata-DS4` (branch `deepseek4`; second worktree `~/AI/Strata-DS4-up`, branch `ds4-upstream`).
> Read, in order: `docs/ds4/V41_RESUME_PROMPT.md` (this file, ALL of it - especially §In flight and §Session log
> 2026-10-10), `docs/UPSTREAM_SYNC.md` (before touching `src/` or `include/`), then `docs/ds4/V41_PLAN.md` as needed.
>
> **0. Ground rules (Mal).** Work solo (no conductor workers). Measure before concluding; commit each verified step
> with the Co-Authored-By trailer; **never push without Mal's explicit OK** (`deepseek4` is public as `ds4`).
> **Mal decides when the GPU is used**: don't start GPU work he hasn't OK'd, and give strata-glm GPU windows when
> asked. Never kill processes by a `pgrep -f` pattern that can match your own command line, and never wait on one
> (`until ! pgrep -f X` never ends - it matches itself): use PIDs, pid files, or `grep "[x]..."`. Never edit a running
> bash script in place (write a copy, `mv` over it). CPU work at `nice -n 15`, 4 threads. Big RAM jobs only under
> `tools/ds4/memguard.sh`. Never a CPU-only full-model load. Explain results plainly to Mal (he reads the summary,
> not the tool output).
>
> **1. Relay.** `relay_rename strata-ds4-gpu`, then `relay_peers`. `strata-glm` (~/AI/Strata-GLM) shares the RTX 4090
> and the shared Strata core; replies to its asks often fail with `peer_gone` - answer with a fresh `relay_ask` instead.
>
> **2. State check (2 min).**
> - Calibration: `tools/v41/calibctl.sh status` (alias `v41calib`, in ~/.bashrc). Resume with `v41calib start`
>   only if Mal OKs the GPU. When all 40 layers are done, run `v41calib start` once more: it skips every layer and runs
>   the `head` gate (the driver running on 2026-10-10 was started before the head step existed).
> - Download: count `*.ok` (`ls /media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/*.ok | grep -c model-`,
>   plus 47 on Crucible and 48 on Team CX2 = 48 total). `tail .../sequence2.log`. Re-run
>   `setsid nohup bash tools/v41/fetch_sequence2.sh &` if no `fetch_v41`/`curl` is running and shards are missing.
> - Worktree `~/AI/Strata-DS4-up`: `git status` - the PTQ1_0 GPU work may still be uncommitted there (§In flight).
>
> **2b. FIRST: the calibration is STOPPED (7/40 layers) - the GPU was handed to strata-glm at Mal's request.**
> Restart with `v41calib start` ONLY when strata-glm relays "GPU free" or Mal says so (it resumes at layer 7;
> layer 7's first try OOM'd on a hot expert, fixed in this commit by chunking the expert forward - watch layer 7
> finish: `v41calib status`, `tail /media/Crucial1TB/v41-calib/logs/L07.log`).
>
> **3. Continue** with §Next steps.  Built 2026-10-10: PTQ1_0 GPU (all 9 tier pairs in the grouped kernel), GGUF
> writer, quantize pass + driver (§Export pipeline).  When calibration layers 0-39 + head are done: `v41_alloc.py
> --layers 0-39 --per-proj --out /media/Crucial1TB/v41-calib/alloc_full.json`, report the tier map to Mal in plain
> terms, then run the export (§Export pipeline, with Mal's GPU OK).  Meanwhile: the V4.1 engine port (§Next steps 5).

---

## Status snapshot (2026-10-10 ~09:00)

| item | state |
|---|---|
| Download | **complete**: 48/48 verified (1-46 on Crucial1TB, 47 Crucible, 48 Team CX2) |
| Calibration | **STOPPED at 7/40 for strata-glm** (Mal 09:00); layer 7 OOM'd 08:55 (hot expert, 88K tokens at L6) - fixed (chunked expert forward); ~6.5 h of GPU left. `v41calib start` when the GPU is handed back |
| Calibration data | agentic-first corpus, 32 x 8192 tokens (§Calibration pipeline); old Pile run archived in `v41-calib/old-pile2048/` (biased stats: damp 0.01) |
| GPTQ damping | **1.0** (layer-8 sweep; 0.01 overfit) |
| Layer-0 tier map (new corpus) | per-proj: 383 ptq1 / 140 q2_0 / 245 q3_k slots @ 2.25 bpw, proxy -39% vs all-Q2_0; per-expert 190/114/80, -24%; down proj = 68% of an expert's Q2 error; 3 rarely-routed experts floored at q2_0 |
| Engine | `deepseek4` = upstream Strata `fb58e0db` merged (`cc55d4c9`) + fork isolation (`20618210`): p600 ppl 10.47, decode 19.55 tok/s (pre-merge 16.8-17.3) |
| `head` gate | built (`5d99832d`), wired into `calib_layers.sh` after layer 39; CPU smoke passed (`e567a476`) |
| GGUF writer | **built + tested** (`70f81b03`, `d020b910`): `v41_gguf.py`, in-place expert mode, independent checker |
| Quantize pass | **built, CPU-smoked** (`d020b910`): `v41_quant.py` + `quant_layers.sh`; not run on the GPU yet |
| PTQ1_0 ternary | CPU done (`50e9428d`); GPU grouped-expert kernels done + tested (`4a403b15`); all 9 tier pairs through `native_expert_grouped` PASS (`cc26cb5d`) |
| GSQ refinement / V4.1 engine port | not started (GSQ: optional, plugs in where v41_quant calls GPTQ) |
| GitHub | nothing pushed today; `Indras-Mirror/Strata-MiMo` **deleted** at Mal's request (local `~/AI/Strata-MiMo` kept, 10 unpushed commits) |

---

## In flight (2026-10-10 ~09:00) - check each of these first

1. **Calibration driver** `calib_layers.sh 0 39` (pid in `/media/Crucial1TB/v41-calib/driver.pid`), detached via
   `calibctl`. Logs: `v41-calib/calib.log`, `v41-calib/logs/LNN.log`. It was started from the copy of
   `calib_layers.sh` BEFORE the head step was added, so after layer 39 run `v41calib start` again (head only).
2. **PTQ1_0 GPU: DONE + PASSED (`4a403b15`, on `deepseek4`).** `ptq1_0_cuda_test`: dequant f16 bit-exact
   (16384/16384); `iq_mmvq(143)` worst 9.9e-8 relative vs the exact reference (the first run's ~2e-4 "fail" was the
   test's fp32 activation scale; the kernel reads fp16).  Pieces: `strata_block_ptq1_0` + `vec_dot_ptq1_0_q8_1`
   (fork_vecdot.cuh, llama-tq @52a37aec MIT), `Fmt<143>` in the FORK lists (fork_fmt.cuh), `dq_ptq1_0` (fork_dq.cuh),
   3 hooks in iq_kernels.cu.  Rerun: `flock ~/.quetza-data/conductor/ds4-gpu.lock ~/AI/Strata-DS4-up/build-gpu/ptq1_0_cuda_test`.
   **Still open:** `native_expert_grouped` with a 143 layout (compiles, untested); a 4-part (ipb 4) dot for the
   2304-wide down rows (ipb 1 leaves lanes idle: 18 blocks per row vs 32 lanes).
   - Reference sources kept in `docs/ds4/calib-refs/llama-tq/` (`mmvq-ptq1_0.cuh` planar kernel, `vecdotq.cuh`).
3. **`head` CPU smoke: PASSED 2026-10-10 08:08** (4 x 256 Pile workdir, layer 0 on CPU, `head --layer 1`; ppl ~1e7 is
   expected on layer 1's output - code path only).  The first try died on `assert N % B == 0` (default batch 16 vs
   N=4); fixed in `e567a476` (batch default = largest divisor of N <= 4096/S; the real 32 x 8192 run is unchanged).
   Redo: `V41_WORK=<tmp> HF_DATASETS_OFFLINE=1 python3 tools/v41/v41_calib.py prep --n 4 --seq 256 && ... init &&
   OMP_NUM_THREADS=4 ... layer 0 --device cpu --gptq-experts 0 && ... head --layer 1 --device cpu`.
4. **Download: complete** (48/48 verified 2026-10-10 ~08:20).  **Quant-pass CPU smoke** (scratch, gone after a
   reboot): `v41_quant.py` layers 0 (q2_0) and 1 (ptq1, Engram layer) + head on a 4x256 workdir; layer 0 passed
   (in-place fill + checker PASS).  Redo recipe = the commit message of `d020b910` + §Export pipeline.
5. **strata-glm** is merging upstream into GLM (branch `glm-upstream`, worktree `~/AI/Strata-GLM-um`), taking DS4's
   shared-file resolution (`cc55d4c9`/`20618210`) and moving its LoRA kernels into `src/kernels/cuda/fork/`.  GLM keeps
   its own `native_expert.cpp` (no PTQ1_0 - Mal's call), so that one shared file will differ.

---

## Session log 2026-10-10 (what changed and why)

- **Afternoon session (08:05-09:05), all committed:** PTQ1_0 GPU test passed (the earlier "fail" was the test's fp32
  scale); grouped-kernel test over all 9 tier pairs; `v41_formats.py` (bit-exact encoders); GGUF writer (DwarfStar
  names, per-expert tiers, split Engram shard); quantize pass + driver; writer `--experts-in-place` (no QDIR copy:
  peak disk = the final file); independent checker.  Mal's decisions: **no Engram copy** (engine reads it from HF
  shards 47/48 - keep them), the control pass writes no weights, final file on SSD NVME (163 GiB free; 152 GiB
  needed), serve V4.1 exactly like `~/.local/bin/strata-glm-quetza` (§Serving).  **Bug found + fixed: tokenizer
  types** - 54 added-but-not-special tokens (<｜System｜>, <｜User｜>, <｜Assistant｜>, ...) must be type 4
  (USER_DEFINED), not 1, or they tokenize as text (389/394 texts wrong -> 0/394 after; also fixed in the GGUF meta).
  Mal asked (answered): RCO/GSQ from the native-FP4 experts IS viable (FP4 is the trained model; 4.25 -> 2.25 bpw;
  GPTQ beats RTN 0.58/0.70/0.74; the head gate on the quantized chain vs the all-Q2_0 control decides).

- **GPTQ overfit found and fixed.**  Layers 0-7 on the old Pile corpus: from layer 1 on, experts with fewer fit tokens
  than the median were WORSE than RTN on held-out data (q3 ratio up to 1.43): most experts see ~2.5K fit tokens but
  the gate/up Hessian is 5120 wide (rank-deficient), and damp 0.01 was too weak.  Layer-8 sweep (48 experts, damps
  0.01-3): **damp 1.0** beats RTN in every group (GPTQ/RTN 0.58 ptq1 / 0.70 q2 / 0.74 q3), 3.0 is worse.  New flags:
  `--damp a,b,c` (sweep; extra damps stored as `e_<tier>@<damp>`), `--expert-stride K`, `--screen-only TAG`.
- **Calibration corpus rebuilt agentic-first** (Mal: calibrate for what we keep - agentic tool use, long-form,
  coding, English + Chinese).  Researched other V4.1 quants: nobody published a calibrated V4.1 tier map
  (pfeifferj's RCO "kept the initial allocation"; apetersson / pyrodog / INC AutoRound are uncalibrated).  Borrowed:
  antirez/ds4's 4690-prompt imatrix set (V4 DSML -> re-rendered for V4.1) and DeepSeek's official V4.1 encoder.  Mal's
  own session transcripts are the long-agentic slice (Mal OK'd; scrubbed, never leaves the box).  Mal chose 256K
  tokens (not 512K).  smalinin publishes a 262K-token V4.1 imatrix (`imatrix-dsv41-262144.gguf`, 789 MB, not fetched):
  plan to cross-check per-expert coverage against it before the final map ("borrow other calibrations and match").
- **Overnight stall (lesson):** the "wait for layer 8" watcher used `pgrep -f` on a pattern in its own command line
  and never fired; the GPU sat idle 01:51-05:37.  Same bug recurred once for a build wait (caught).  Use log/pid-file
  watches only.
- **8K sequences OOM'd the 24 GB card** at 2 x 8192 per attention batch: `--batch` now defaults to ~4K tokens
  (1 x 8192) and the driver sets `expandable_segments`.
- **Stoppable/resumable run** (Mal will work over several sessions): `calibctl.sh` + STOP file + done = stats exist.
- **Upstream Strata merged** (Mal: keep tracking upstream, isolate fork edits).  Histories were unrelated (upstream
  rewrote its history; our main had the identical tree) -> one-time graft merge; future syncs are `git merge
  upstream/main`.  Then the isolation refactor (fork code into `src/kernels/cuda/fork/*.cuh`, `tools/ds4/cmake/
  {fork,ggml_cuda}.cmake`, `include/strata/artifact/ds4_guard.hpp`): footprint in upstream files +727 -> +327 lines.
  Upstream's Foresight swap / router look-ahead prefetch / prefill CPU share / MMVQ_IL rows table /
  --expert-cache-per-layer live in the Qwen engine path - port them to the DS4 path by hand later (they did NOT come
  in through the merge).
- **SwiGLU clamp stays:** V4.1's own config has `swiglu_limit: 10.0` and DeepSeek's `model.py:845` applies it.
- **FP8/FP4 question (Mal):** the routed experts we quantize are native FP4 (the trained values - no better source
  exists); "FP8" is the dense weights, which stay >= Q8.  Errors are measured against the FP4 teacher.
- **Engine name:** undecided (Mal is thinking: Terra / Firma / ...; suggestions given: Tezca, Xibalba, Fathom, Mantle).
  Strata is MIT - renaming is allowed; keep Niko1221's copyright + license, credit "built on Strata".

---

## Calibration + GPTQ screening pipeline (built 2026-10-09 23:00-23:40)

- `tools/v41/v41_calib.py` — DeepSeek's `model.py` run **one layer at a time** straight from the HF shards.
  **Use DeepSeek's inference `config.json` (`~/AI/gsq-rco/dsv41-inference/config.json`)**: the HF one in the model dir
  is the transformers schema and silently leaves ModelArgs at its TINY defaults (caught; now cross-checked by assert).
  - `tools/v41/build_corpus.py` (CPU, ~1 min) -> `v41-calib/corpus/*.jsonl`: session transcripts (Claude Code /
    Quetza / GLM jsonl; longest prefix <= 8191 tokens; secrets scrubbed value-by-value; tool results clipped) and
    antirez/ds4's 4690 imatrix prompts (V4 DSML parsed back into calls), all rendered by `dsv41_encoding.py`
    (DeepSeek's V4.1 encoder, vendored: DSML tags have a LEADING SPACE in V4.1).  References + their provenance:
    `docs/ds4/calib-refs/README.md`.
  - `prep --n 32 --seq 8192 --corpus v41-calib/corpus` (CPU): packs the agentic-first mix (old Pile mix without
    `--corpus`), BOS per sequence, stratified ~1/4 held out per domain; engram hash ids -> `hashes.npy`.
  - `init` (CPU): embedding (shard 2) -> `h/L00.bin` (10 GiB bf16, [128,2048,4,5120]).
  - `layer L` (GPU, ~20 GB VRAM, ~20-25 GiB RAM): attention half per batch of ~4K tokens (1 x 8192; 2 x 8192 ran
    out of the 24 GB card), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` -> gate -> **expert-major**
    MoE (each FP4 expert's reference `Expert.forward` once over all its tokens) -> **per-expert GPTQ** at
    ptq1/q2_0/q3_k (+RTN control) fitted on the 98 fit sequences, **error measured on the 30 held-out ones**:
    `e_<tier>` whole expert, `egu_<tier>` gate+up only, `edn_<tier>` down only, `rtn_<tier>`; plus REAP saliency
    (`reap` = mean ||g f(x)||), counts, `y_energy`.  -> `stats/LNN.json`, `route/LNN.npz`, `h/L(N+1).bin`,
    `pre/`, `shared/` (SharedAttentionRuntime per batch: source-layer compressed KV / index keys / top-k / candidates).
  - Engram: `MemmapEngramEmbedding` replaces the 98 GB table with memmap row reads (MADV_RANDOM).
- `tools/v41/v41_gptq.py` — exact-format grids: PTQ1_0 `{-1,0,1}·d`/128, Q2_0 `{-1,0,1,2}·d`/64 (d fp16, sign
  free), Q3_K `(q-4)·d·(sc-32)`/16 (fp16 d per 256, 6-bit sc).  Static diag(H)-weighted scale search (11 clip ratios x
  both signs), then stacked multi-tier GPTQ (no act-order).  Everything it outputs is exactly representable.
- `tools/v41/calibctl.sh` (alias `v41calib`) drives `calib_layers.sh`: start/stop/stop-now/status/log; a layer is
  done when `stats/LNN.json` exists; `STOP` file checked between layers; first start runs `init`.  `--damp` takes a
  comma list for sweeps (`--screen-only TAG --expert-stride K` writes `stats/LNN_TAG.json` without touching the chain).
- `tools/v41/calib_layers.sh FROM TO` — one memguard'd GPU block per layer behind `ds4-gpu.lock` (strata-glm slots in
  between), shard-gated (layer L = shard L+3; 47 for L=1, 48 for L=14), resumable, `--gc` keeps only 2 h files.
  Log `/media/Crucial1TB/v41-calib/calib.log`, per layer `logs/LNN.log`.  Restart = re-run the same command.
- `tools/v41/v41_alloc.py [--layers 0-7] [--per-proj] [--out map.json]` — tier map: cost = held-out error / layer's
  held-out routed energy (layers 0-1 and 37-39 x2), budget = all-Q2_0 bytes, Lagrangian bisection + greedy fill;
  ternary <= 50%/layer; experts with < 8 held-out tokens floored at q2_0.  Prints per-layer tier counts, bpw,
  GPTQ/RTN error ratios, and the proxy error of the mix vs all-Q2_0.
- Tests: `tests/test_v41_gptq.py` (17/17), `tests/test_calib_equiv.py L` (split forward == reference; needs a smoke
  workdir: `V41_WORK=<dir> v41_calib.py prep --n 4 --seq 256 && init && layer 0 --device cpu --gptq-experts 3`).
- Known limits: the screening uses GPTQ init only (GSQ refinement is the next step, at the chosen tier); the
  allocation proxy treats layers as equally important (validate end-to-end: ppl/KL needs shard 43 = head).

---

## Export pipeline (built 2026-10-10; run after the full tier map)

```
48 HF shards (read only) --quantize pass, layer by layer (GPU)--> experts written IN PLACE into shard 1 of the GGUF
                                                                   (holes laid out first by the writer)
```
1. Tier map: `python3 tools/v41/v41_alloc.py --layers 0-39 --per-proj --out /media/Crucial1TB/v41-calib/alloc_full.json`.
2. Lay out the final file (CPU, ~10 min, ~10 GB written; the rest is holes until filled):
   `python3 tools/v41/v41_gguf.py --out "/media/mal/SSD NVME/<dir>/V41-GSQ.gguf" --rtn-map .../alloc_full.json
   --experts-in-place --no-engram` -> `V41-GSQ.gguf` + `V41-GSQ.gguf.experts.json` (with --no-engram it is one plain
   file, no split).  Size at the all-Q2_0 budget: 151.8 GiB; SSD NVME had 163 GiB free (2026-10-10) - nothing else
   goes there first.
3. Quantize pass (GPU, ~4-5 h estimated, measure): `tools/v41/quant_layers.sh /media/Crucial1TB/v41-quant-mixed 0 39
   -- --map .../alloc_full.json --gguf "<the file>"`.  Per-layer lock (strata-glm slots in), resumable, STOP file
   (`touch <qwork>/STOP`), ends with the head gate on the QUANTIZED chain (stats/head_L40.json in the qwork).
4. All-Q2_0 control: `quant_layers.sh /media/Crucial1TB/v41-quant-q2 0 39 -- --tier q2_0 --no-write` (head
   numbers only, writes no weights).  Mal's rule: the mixed file ships only if its held-out per-domain ppl beats
   this control's (and compare both with the teacher's head_L40.json from the calibration run).  If it loses:
   lay out an all-Q2_0 file (`--rtn q2_0 --experts-in-place`) and rerun step 3 with `--tier q2_0 --gguf`.
5. Check: `python3 tools/v41/tests/check_v41_gguf.py <file> --fp4 8 --layers 0-39` (layout, dense vs source,
   every layer's sampled experts decoded vs FP4 - a hole decodes to rel ~1) and `v41_gguf.py --verify <file>
   --rtn-map ... --experts-in-place`.
6. Engram: **NOT copied** (Mal 2026-10-10: no 203 GB copy).  The engine port reads the Engram rows straight from
   the HF shards 47 (layer 1, on Crucible = USB SATA SSD) and 48 (layer 14, Team CX2): row = 256 B at
   `embed.weight` + r*256, scale = 8 B at `embed.scale` + r*8 (2 reads per row, 96 per token, prefetched at token
   start).  KEEP shards 47/48.  Measure Crucible's latency at layer 1 (early in the token: little time to hide
   it); if slow, move that one table (~94 GiB) to a faster SSD.  `v41_gguf.py` without --no-engram still writes a
   264-B-row Engram shard 2 (DwarfStar's format) if one is ever wanted.

Formats (`v41_formats.py`, bit-exact vs ggml / ptq1_0.cpp): Q8_0 34 B/32, Q2_0 18 B/64, Q3_K 110 B/256, PTQ1_0
28 B/128.  Expert layout in the GGUF: a role whose 384 experts share a type is the usual `blk.L.ffn_<role>_exps.weight`;
mixed -> `blk.L.ffn_<role>_exps.<q2_0|q3_k|ptq1_0>.weight` (experts of that type, id order) + I32
`blk.L.ffn_exps_{gu,down}_type[384]` (slot = rank among the experts of the same type).  Gate+up share a type.
Dense: DwarfStar's names/types (`deepseek41_quantize.py` @fc80bd6): Q8_0 attention / shared / head (rel 5.4e-3 vs
FP8), F16 embedding / compressor / indexer / Engram wkv, F32 norms / router / mHC.  mtp.* (DSpark) not exported yet.

## Serving (Mal 2026-10-10: "we'll be running deepseek like `~/.local/bin/strata-glm-quetza`")

Copy GLM's shape exactly (`~/AI/Strata-GLM/tools/glm/serve/`, its RESUME_PROMPT.md "Serving plumbing" + "Metrics"):
engine `ds41_generate --serve` (Strata's engine line protocol, docs/BATCHING.md §Engine protocol + GEN/GENI) ->
launcher `tools/v41/serve/ds41-engine.sh` (copy glm-engine.sh: ComfyUI refusal, `flock` the shared GPU lock for the
server's life, `systemd-run --scope MemoryMax=84G MemorySwapMax=0`) -> shim `tools/v41/serve/serve_ds41.py` that
imports **`~/AI/Strata/serve`** (that checkout has the llama.cpp-style metrics: `/metrics`, `/slots`,
`?format=prometheus` -> `llamacpp:*` + `strata:*`; commit 9bb510c8 is LOCAL - its remote is upstream Niko1221,
NEVER push it) and runs under `/home/mal/AI/Strata/.venv/bin/python` -> configs `tools/v41/serve/*.json` (NO
`--slots` arg: server.py dies on it) -> wrapper `~/.local/bin/strata-ds41-quetza` (copy strata-glm-quetza: port,
model name, context, QuetzaCodetl env).
**Done:** `tools/v41/serve/export_tokenizer.py` (HF tokenizer.json -> vocab.json / merges.txt / token_type.json):
Strata's BPE with pre `joyai-llm` == HF tokenizers on 394 corpus texts / 465,645 tokens, 0 mismatches.
**Next (designed, not written):** `serve_ds41.py`:
- tokenizer: patch `ST.Tokenizer` to pre="joyai-llm" (as serve_glm.py does for glm4);
- end of turn: `S.IM_END = "<｜end▁of▁sentence｜>"`;
- template: replace `F.ChatTemplate` / `S.ChatTemplate` with a class whose `render(messages, tools,
  add_generation_prompt, **kw)` calls `dsv41_encoding.encode_messages` (pure Python; tools go on the first system
  message as OpenAI-format `{"type":"function","function":...}`, tool results via role "tool" + merge; thinking_mode
  "chat" when `enable_thinking is False` else "thinking"; reasoning_effort low/high/max (map medium->high); the
  generation suffix is `<｜Assistant｜>` + `<think>` (thinking) or `</think>` (chat) - strip it for
  add_generation_prompt=False); set `.caps` by hand (tools/thinking true);
- tool calls: a streaming rewriter in front of `F.OutputParser.feed` that turns DSML (`<｜DSML｜ calls>`,
  `<｜DSML｜ invoke name="X">`, `<｜DSML｜ parameter name="K" string="true|false">V</｜DSML｜ parameter>`; note the
  LEADING SPACE in the tag names) into Qwen text (`<tool_call>\n<function=X>\n<parameter=K>\nV\n</parameter>\n
  </function>\n</tool_call>`, one per invoke; drop the calls wrapper), holding back a partial `<｜DSML｜` / `</｜DSML｜`
  tag; check how parse_tool_call trims the value newlines and string="false" -> JSON;
- tests (`tools/v41/serve/test_serve_ds41.py`): tokenizer == HF; template == encode_messages on chats with tools /
  tool results / thinking; DSML split at every byte boundary -> the same calls as `parse_message_from_completion_text`;
  end to end with the server's `--engine mock` + a DSML script over HTTP (OpenAI + Anthropic tool calls).
Tokenizer dir for the real config: next to the final GGUF (SSD NVME).

## Goal (Mal's decisions — all made)

1. **Our own quant of DeepSeek-V4.1-Flash, uncensored**, from `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`
   (510.3 GB). The abliteration (MPOA) touches **dense weights only** — routed experts, engram, router, norms are
   byte-identical to the base. Reported: MMLU 86.96 → 82.74, HarmBench 100% at both effort-off and max.
2. **GSQ (IST-DASLab) quantization of the routed experts, mixed tiers, byte-budget-neutral at all-Q2 size**
   (~153 GB of experts, 2.25 bpw average):
   - sensitive experts → **Q3_K** (3.44 bpw)
   - default → **Q2_0** (2.25 bpw = GSQ's native 2-bit grid)
   - weakest → **PTQ1_0 ternary** (1.75 bpw)
   - Every 1% of experts moved to ternary pays for 0.42% moved to Q3_K.
   - Mal: *"don't lobotomize it"* — **all-Q2 is the fallback**; the mixed tier only ships if it beats an all-Q2 control
     on held-out **per-domain** ppl, **including languages never used in calibration** (FINDINGS s37 pattern).
   - Dense / shared expert / router / engram stay **≥ Q8_0**.
3. **Hadamard rotation for the ternary tier** (and measure it on Q2 too): rotate the weight's input dim by a
   block-128 Hadamard with random signs before GSQ (calibration activations rotated the same way); runtime FWHT on the
   activation. Per-tensor flag in the GGUF.
4. **Engram (203 GB = 189 GiB) never in RAM.** Rows are read from SSD — 48 rows / ~12 KB per token — prefetched at
   token start (the hash depends only on token ids, so it can start before the forward pass). Mal will move the engram
   to NVMe after GSQ (needs ~145 GiB freed on NVME1TB; SSD NVME's 163 GiB is reserved for the expert pack).
5. **Run it in Strata-DS4** (port V4.1 into our engine; reuse soft-prune + arena-adapt + VRAM LRU + RAM arena).
   Expected **6–10 tok/s** at first (NVMe-bound: ~55–60% of the pack fits in ~85–90 GB fast memory), then DSpark.

---

## Where things are on disk (free space as of 2026-10-09 22:51)

| what | path | free |
|---|---|---|
| source shards (all but engram) | `/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/` | 368 G |
| engram layer 1 (shard 47, 101.5 GB) ✓ | `/media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/` | 57 G |
| engram layer 14 (shard 48, 101.5 GB) | `/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/` | 152 G |
| planned GSQ output (~151 GiB) | `/media/mal/SSD NVME/…` (NTFS, PCIe4; mount `udisksctl mount -b /dev/nvme0n1p3`) | 163 G |
| engram final home (later) | `/media/mal/NVME1TB/…` (needs ~145 GiB freed) | 49 G |

**Repos / references**
| what | path |
|---|---|
| GSQ + RCO repos | `~/AI/gsq-rco/GSQ`, `~/AI/gsq-rco/RCO` (cloned, not installed; reuse the quantizer modules) |
| DeepSeek V4.1 reference code (MIT) | `~/AI/gsq-rco/dsv41-inference/` (`model.py`, `engram.py`, `kernel.py`=tilelang, `convert.py`, `vision.py`, `config.json`) |
| DwarfStar = antirez/ds4 V4.1 C reference | `~/AI/ds4-ref` @ `fc80bd6`: `ds4_engram.c`, `ds4_deepseek41_cuda.cuh`, `gguf-tools/deepseek41_quantize.py` |
| msuiche GLP control vector (alt uncensor, base only) | `~/AI/Strata-DS4-private/v41/glp.gguf` (39 rank-1 dirs, alpha 0.5) |
| video transcript (Local Explained, V4.1 HW) | `~/AI/Strata-DS4-private/v41/video-UMrKQWelWp0-transcript.txt` |
| pfeifferj V4.1 GSQ-RCO GGUF (3.0 bpw) = 2nd oracle | HF `pfeifferj/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF`, its `runtime/` has a patched llama.cpp |

**Checkpoint layout** (one layer per shard): 1 vision · 2 embed · **3..42 = layers 0..39** (dense + 384 experts each)
· 43 head+norm · 44–46 DSpark/MTP · **47 = engram layer 1** · **48 = engram layer 14**.
**Mapping:** `shard = layer + 3` (layer L → shard L+3). GSQ's calibration forward adds the engram at layers **1 and 14**
(`config.json` `engram_layer_ids: [1,14]`, applied at model.py:1262). So layers **0–13 need only shard 47**; layer 14+
needs **shard 48**. Layer 0 needs no engram at all.

---

## Download (detached, survives the session)

- **Driver:** `tools/v41/fetch_sequence2.sh` (launched with `setsid nohup … &`), log
  `/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/sequence2.log`; each file's own progress in the
  `fetch.log` of its directory. (`fetch_sequence.sh` / `sequence.log` are **superseded**.)
- **Order (Engram-matched to GSQ, set 2026-10-09 20:00):**
  `47 → backbone layers 2–13 (shards 5–16) → 48 → the rest of the backbone (17–46)`.
  Rationale: shard 47 is the only Engram shard GSQ needs for layers 0–13, so those layers' backbone shards are fetched
  first so GSQ can run **while** the rest downloads; 48 is packed in well before layer 14.
  The script is **self-healing**: if it starts and no 47 fetch is running (e.g. after a reboot), it fetches 47 itself.
- **Fetcher:** `tools/v41/fetch_v41.sh DEST [REPO]` — resumable (`curl -C -`), one `.ok` marker per verified file,
  sha256 via **O_DIRECT** read (the page cache lies on this box). Env: `JOBS=1 FIT=1 MARGIN_GB=10 ONLY=regex SKIP=regex
  ORDER=name|size`. Re-run the same command any time: verified files are skipped, partials resume.
- **Manual restart after a reboot** (or just `setsid nohup bash tools/v41/fetch_sequence2.sh &`, which does all of this):
  ```
  cd /media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 && JOBS=1 FIT=1 ONLY='model-00047-of' ~/AI/Strata-DS4/tools/v41/fetch_v41.sh "$PWD"
  cd /media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 && JOBS=1 FIT=1 ONLY='model-000(0[5-9]|1[0-6])-of' ~/AI/Strata-DS4/tools/v41/fetch_v41.sh "$PWD"
  cd "/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8" && JOBS=1 FIT=1 ONLY='model-00048-of' ~/AI/Strata-DS4/tools/v41/fetch_v41.sh "$PWD"
  cd /media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 && JOBS=1 FIT=1 SKIP='model-0004[78]-of' ~/AI/Strata-DS4/tools/v41/fetch_v41.sh "$PWD"
  ```
- **Speed:** ~8–14 MB/s (occasional 1–3 MB/s dips): ~9 min per 7.4 GB shard, ~2 h per engram shard, ~11–12 h total.
- **Cautions:** never edit a running bash script in place (bash reads incrementally) — write a copy and `mv` over it.
  The HF token is visible in `ps` (`curl -H`) — don't paste process listings into anything shared.

---

## GPU coordination (single RTX 4090, shared with GLM)

- Shared with **`strata-glm`** over quetza-relay. **Lock:** `~/.quetza-data/conductor/ds4-gpu.lock`.
  Take it for **≤10–12 min blocks**, `relay_ask strata-glm` before you do.
- **Every full-model / big-RAM run** goes through `bench/ds4-2026-10-08/q.sh <log> [extra args]`:
  it waits for the lock **and** MemAvailable ≥ 70 GiB, refuses if ComfyUI (:8188) is up, then runs under
  `tools/ds4/memguard.sh 80 78 -- …`.
- `tools/ds4/memguard.sh <max_gib> <need_avail_gib> -- <cmd>` runs the process in a systemd **user scope** with
  `MemoryMax=<max_gib>G` and `MemorySwapMax=0` (kernel OOM-kills *this* process instead of swapping the desktop),
  plus a watchdog that kills the scope if MemAvailable drops below `MEMGUARD_FLOOR_GIB` (default 4), and holds the
  GPU lock for the whole run. **Why:** on 2026-10-06 a CPU-only full-model oracle (~81 GB anon RAM) + ComfyUI
  start-up thrashed swap into a **hard freeze** (no OOM kill). So: never a CPU-only full-model load.
- While GLM benchmarks: run CPU work at `nice -n 15`, 4 threads; **don't** rebuild `build-ds4-gpu` while a batch is queued.
- The calibration takes the lock **per layer** (memguard holds it for that layer only), so other jobs slot in between
  layers.  A short GPU test: `flock ~/.quetza-data/conductor/ds4-gpu.lock <cmd>` (waits for the next gap).  A longer
  GPU job (real-model p600 gate etc.): `v41calib stop` -> wait for the driver to exit -> run -> `v41calib start`
  (template: `~/AI/Strata-DS4-up/bench/upstream-merge-2026-10-10/gate.sh`; `gate-isolated-pause.sh` = the variant that
  leaves the calibration paused for a strata-glm window).
- The merged-tree GPU build lives in the worktree: `~/AI/Strata-DS4-up/build-gpu` (Ninja, sm_89, CUDA, tests ON,
  `STRATA_GGML_DIR=/home/mal/AI/Strata-DS4/third_party/llama.cpp`); the main checkout's `build-ds4-gpu` predates the
  merge.  Configure line: `docs/UPSTREAM_SYNC.md` / commit `cc55d4c9` message.

---

## Built and verified (commits on `deepseek4`)

| commit | what | verification |
|---|---|---|
| `28dd12dc`,`6ce93ed3`,`626dbac0` | `docs/ds4/V41_PLAN.md` — plan, V4→V4.1 deltas, mixed tiers, research rounds 1–2 | — |
| `28dd12dc`,`c67b47ef`,`383fd8b7` | `tools/v41/fetch_v41.sh`, `fetch_sequence.sh`, `fetch_sequence2.sh` | tiny-repo test incl. corrupted-file recovery |
| `f3bc404d` | `tools/v41/torch_kernels/kernel.py` — torch stand-in for DeepSeek's tilelang `kernel.py` (`act_quant` / `fp4_act_quant` / `fp8_gemm` / `fp4_gemm` / `sparse_attn` w/ sink / `hc_split_sinkhorn`; rounding copied from the kernels) | `tools/v41/tests/test_torch_kernels.py` 15/15; `tests/selftest_model_cpu.py` runs DeepSeek's `model.py` (prefill + decode + DSpark) on it |
| `e1076fc3` | `tools/v41/shard_check.py` — per-shard validation + per-expert weight-space tier errors → `bench/v41/shard-check/` | layer 0: 2,334 tensors, 0 issues |
| `50e9428d` | **PTQ1_0 ternary on CPU**: `include/strata/kernels/cpu/ptq1_0.hpp`, `src/kernels/cpu/ptq1_0.cpp` (pack / unpack / quantize_ref / dequant; Q8_0 dot scalar + AVX2), wired into `native_expert.cpp` via a type-trait wrapper | `tests/cpu/ptq1_0_test.cpp` **16/16** (target `ptq1_0_test` in `build-ds4`): bytes identical to llama-tq's reference quantizer at n = 5120 / 2304 / 128; AVX2 == scalar bitwise; == float reference; mutation caught |
| `42c77429` | ds4_dense: stale-graph unbind compares buffer pointers (GLM's gallocr audit) | ASan CPU gate `swa16`/`tame` (shared + per-graph allocator): PASS |
| `c3ca0410` | `docs/ds4/V41_FEASIBILITY.md` | — |
| `9221882a` | `docs/ds4/RESEARCH_2026-10-09.md` (Infernix, Strata 0.1.41, community ideas) | — |

| `2e442b20`,`317ef99a` | v41_gptq.py (exact-format multi-tier GPTQ), v41_calib.py (layer-streaming calibration), v41_alloc.py, calib_layers.sh | test_v41_gptq 17/17; test_calib_equiv: split forward == DeepSeek Block.forward bit-exact (layers 0/1) |
| `80d7cccd` | GPTQ damp sweep options; default damp 1.0 | layer-8 sweep (§Session log) |
| `d2890cc2` | `build_corpus.py`, vendored `dsv41_encoding.py`, `prep --corpus`, `docs/ds4/calib-refs/` | 0 render failures (4690 + 40 sessions); tags balanced; secrets scan clean; per-sequence BOS/role-token check |
| `baa0b23b`,`0099974a`,`7effced9`,`122c7c3e` | `calibctl.sh` (v41calib), resumable driver + STOP, ~4K-token attention batches, ETA | dry-run start/stop/resume/stop-now in a scratch dir |
| `cc55d4c9`,`497dbd47` | **upstream Strata merged** (graft merge, 8 conflicts) | CPU parity (ptq1_0_test, pool_tasks_test 168 bitwise, q8k identical, iq_avx2_parity 0 fail, ds4_geometry, test_ds4_moe 15/15); dense gate bit-identical 63/63; p600 ppl 10.41 / 19.29 tok/s |
| `20618210` | fork isolation (fork/*.cuh, tools/ds4/cmake/fork.cmake, ds4_guard.hpp) | same CPU gates + dense gate bit-identical; p600 ppl 10.47 / 19.55 tok/s |
| `2a4254d8`,`d564238a` | `docs/UPSTREAM_SYNC.md` (process, gates, isolation rules) | — |
| `5d99832d`,`e567a476` | `v41_calib.py head` + end-of-chain head gate in `calib_layers.sh`; batch default divides N | CPU smoke (4x256, head on L1) runs end to end |
| `4a403b15`,`cc26cb5d` | PTQ1_0 in the grouped-expert CUDA kernels (dot, Fmt<143>, dequant, row bytes) | `ptq1_0_cuda_test`: dequant bit-exact, mmvq 9.9e-8, `native_expert_grouped` all 9 Q2_0/Q3_K/PTQ1_0 pairs rel <= 1.6e-5 |
| `d52f57ca` | `v41_formats.py` block encoders/decoders | `tests/test_v41_formats.py` 6/6 bit-exact vs ggml + ptq1_0.cpp (builds `tests/fmt_ref.cpp`) |
| `70f81b03` | `v41_gguf.py` writer + `v41_gguf_meta.py` (DwarfStar metadata port) | full dry run claims every source tensor; 2-layer real file, synthetic mixed tiers: --verify 68/68 + `check_v41_gguf.py` PASS |
| `d020b910` | `v41_quant.py`, `quant_layers.sh`, writer `--experts-in-place`, cmd_layer hooks | CPU smoke quant L0 (q2_0); in-place file + fill: check PASS, Q2_0 rel 0.413 vs FP4 |
### Facts established (measured, or read from source)
- **V4.1 has NO per-head q RMS norm** (V4 does). JigSawPT's port erroneously kept it → fluent text but layer-39
  correlation 0.78 vs reference. Do **not** add it.
- FP8 dense: **32×32** E8M0 block scales. FP4 experts: int8-packed, **low nibble = even index**; E2M1 codes
  {0, .5, 1, 1.5, 2, 3, 4, 6 + negatives}; one E8M0 scale per 32 along K.
- Tensor names: `layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}`,
  `layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}` (I8 `[2304,2560]` / `[5120,1152]`), `ffn.shared_experts.*`,
  `ffn.gate.{weight,bias,bias_vl}`, `hc_{attn,ffn}_{fn,base,scale}`, `attn.attn_sink`.
- **Weight-only priors cannot rank experts.** All 384 experts of layer 0 have the same RTN error to 3 decimals
  (ternary 0.443 / Q2 0.378 / Q3 0.183 relative); FP4 training gives every expert the same per-32 distribution.
  ⇒ Allocation must come from **calibration**: activation-weighted (GPTQ/GSQ) error × REAP saliency.
- `transformers` has no `deepseek_v41` (only `v4`) → the reference is DeepSeek's own `model.py` + our torch kernels.
- Strata's native expert kernels run Q2_0, Q2_K, Q3_K, IQ2_XXS, IQ1_M, and (2026-10-10) PTQ1_0 on CPU and GPU.
- **Engine port must teach `include/strata/artifact/gguf_reader.hpp` type 143** (`ggml_type_name` / `block_geometry`:
  128 elems, 28 B, "PTQ1_0") - today it would size PTQ1_0 tensors as 0 bytes.  Shared upstream file: keep the edit
  isolated per docs/UPSTREAM_SYNC.md.  Its tensor-name check (`check_ds4_model`) is V4's; V4.1 needs its own.
- A mixed-tier layer = up to 9 `native_expert_grouped` calls (one per gu/down type pair present), each with its
  own `NativeExpertLayout`; the engine's per-layer single expert stride (`Ds4GgufBlobs`) must become per-expert.
- The vendored ggml (`third_party/llama.cpp`) has `GGML_TYPE_COUNT = 43`: **type 143 must never reach ggml**
  (it would read past the traits table). CPU path wraps it via a local traits object in `native_expert.cpp`.
- PTQ1_0 = id 143 in llama-tq and ik_llama.cpp (#2550). Block = 28 B: `qs[0..15]` 5 trits/byte (values m+16n),
  `qs[16..23]` 5 trits/byte (80+m+8n), `qh[0..1]` 4 trits/byte (120+h+2n, shifted to the top trit), fp16 `d`.
- DwarfStar's CUDA SSD expert cache is plain global LRU (we are ahead); **take** its engram prefetch-at-token-start.
- Speed references: sybil-solutions (3090 + 503 GiB 8-ch DDR4) ~20 tok/s to 261K (experts all in RAM + AVX2 CPU tier);
  JigSawPT (5090 + 126 GiB, NVMe streaming, MXFP4) 5.12 tok/s on new content.

---

## Next steps (in order)

1. **Calibration runner + GPTQ screening — DONE; full run in progress (2026-10-10).**  `head` step BUILT
   (`v41_calib.py head [--layer 40]`, writes `stats/head_L40.json`: ppl / top-1 per domain, fit vs held-out); expect
   held-out ppl in the single digits on web/code and a sane top-1 (~0.5-0.7) - a broken layer anywhere shows up as a
   ppl far off.  Then `v41_alloc.py --layers 0-39 --per-proj` for the full map (cross-check rarely-routed experts
   against smalinin's V4.1 imatrix).
2. **Quantize pass — BUILT** (§Export pipeline; GPTQ at the chosen tier).  **GSQ refinement** (optional, plugs in
   where `v41_quant.quant()` calls GPTQ) (screening already gives the GPTQ init and the tier per expert). Quantizer modules from
   `~/AI/gsq-rco/GSQ/src/quantization` (2-bit grid = Q2_0 group 64; ternary group 128 = PTQ1_0; 3-bit → project onto
   Q3_K). Record each expert's activation-weighted error at all three widths (GPTQ init is cheap; run GSQ refinement
   only at the chosen width) → RCO-lite knapsack on `error × saliency` under the exact byte budget → second pass at the
   chosen widths. Optional Hadamard rotation per tensor (measure both).
3. **PTQ1_0 on the GPU** — IN PROGRESS (§In flight 2): done in the grouped-expert kernels (`iq_kernels.cu` FORK path,
   which is what DS4's experts use); the plan below for `native_mmvq.cu` is only needed for dense use.  Original note:
   add to `src/kernels/cuda/native_mmvq.cu` (+ header `native_mmvq.hpp`: `supported(143)`,
   `weight_bytes`, dispatch) with q8_1 activations: per 128-block, 4 q8_1 sub-blocks, dp4a of `(digit-1)` with q8.
   Reference: llama-tq `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh` (MIT). Then the FWHT op for rotated tensors, then prefill
   (dequant → fp16 GEMM is exact for ternary). Later: R8 row-interleaved CPU repack (ik #2564: 3.4×).
4. **GGUF writer — DONE** (§Export pipeline).  Open: rotation flags (if Hadamard is used), mtp.* (DSpark) export,
   a load test in DwarfStar for the dense half (their loader will refuse our expert types).
5. **Strata-DS4 V4.1 port** (V41_PLAN.md "Plan B"): loader; dense half (single-pass HC, ratio 0/2/1 compressors,
   kv/index source sharing, two-level indexer, fp4 compressed KV, **no q norm**); engram op with prefetch;
   384-expert tier (Q2_0 / Q3_K / PTQ1_0); then DSpark. Golden numbers from step 1.
6. **Validation**: per-domain held-out ppl of mixed vs all-Q2_0 control vs the FP4 original; then speed with soft-prune.

---

## Other open items (DS4 V4, from earlier today)

- **b33:** DS4 V4 decodes 21.2 tok/s at both 256K and 512K (`--pos-offset` probe, `iq4_nl` KV); soft-prune neutral in
  the short probe — **needs a real 64K-document run** (prefill + decode, unpruned vs pruned).
- **Strata upstream merge: DONE 2026-10-10** (Mal: keep tracking upstream).  Upstream main `fb58e0db` merged
  (`cc55d4c9`), fork code isolated into `src/kernels/cuda/fork/` + `tools/ds4/cmake/` (`20618210`); gates in
  `bench/upstream-merge-2026-10-10/`.  The old `~/AI/Strata-DS4-0141` worktree is obsolete.  Upstream features that
  live in the Qwen engine path (Foresight swap / router look-ahead prefetch, prefill CPU share, MMVQ_IL rows table,
  --expert-cache-per-layer) are worth porting into the DS4 path as ideas.  GLM is doing the same merge (branch
  `glm-upstream`) and takes DS4's shared-file resolution.
- **Ideas queued for DS4** (`RESEARCH_2026-10-09.md`): rank-gated `--arena-adapt` admission, persisted expert state,
  elastic KV / 32K VRAM KV window for 512K, adaptive MTP draft length, batched NVMe reads.
- **MiMo-Audio 7B** was deleted at Mal's request (sources: `~/AI/Strata-MiMo/docs/mimo/MODEL_FILES.md`). He wants it
  back later: `hf download XiaomiMiMo/MiMo-Audio-7B-Instruct` (+ `-Tokenizer`) — **download to ext4 first** (`hf`
  hangs on exFAT file locks), then move to `NVME1TB/Models/Mimo-Audio/`.

---

## Working rules (this box / Mal)

- GPU is shared with `strata-glm` over quetza-relay; lock `~/.quetza-data/conductor/ds4-gpu.lock`; full-model runs via
  `bench/ds4-2026-10-08/q.sh` + `tools/ds4/memguard.sh` (never a CPU-only full-model load — it froze the box once).
- CPU work at `nice -n 15`, 4 threads while GLM benchmarks. Don't rebuild `build-ds4-gpu` while a batch is queued.
- Never `pgrep -f`+kill a pattern that appears in your own command line (killed own shell once) — use PIDs/PGIDs.
- Verify downloads with **O_DIRECT** reads; RAM has a known bad lane (soft-offlined).
- Commits end with the `Co-Authored-By` trailer; **no pushes without Mal's OK.**
- Measure before concluding; reproduce before diagnosing; commit each verified step.

---

## Engine (night of 10-10) - tools/v41/engine/

| file | what | gate |
|---|---|---|
| `ds41_geometry.hpp` | keys, full tensor map (conditional sets), per-expert tier maps, full-length GGUF arrays (the reader samples 64) | `test_ds41_geometry` PASS + mutations |
| `ds41_dense.{hpp,cpp}` | V4.1 dense half: one pass = n tokens at pos0 (decode n=1, prefill chunk); single-pass HC; window ring [ring \| pass rows]; compressed KV on kv sources (ratio 2 pooled, 1 plain) read by later layers; indexer (fused ggml_lightning_indexer) on index sources, top-k reused; layer-20 candidate blocks (pool max + pinned newest block); flash-attn with the QUERY AS BATCH (each query its own window + top-k rows), sinks; Engram rows from HF shards 47/48; snapshot/restore (rings only) | CPU vs `ds41_golden.py`: 3-layer file, 21 tok cos>=0.99996 top-1 21/21; 1688 tok (top-512 of 844) cos>=0.99983; decode == prefill |
| `ds41_ops.hpp` | model.py's KV/indexer rounding (FP8/pow2, FP4/E4M3, FP4/pow2) as exact stock-ggml float tricks, `act_quant` default on | bit-identical to torch on 2M values (NaN-overflow case saturates) |
| `ds41_engram.{hpp,cpp}` | n-gram hash (GGUF arrays) + parallel preads of rows + scales | via the dense gate (layer 1) |
| `ds41_golden.py` / `compare_ds41.py` | model.py on the GGUF's own weights (CPU, layer by layer) / per-layer cos, routing sets, logits | - |
| `ds41_moe.{hpp,cpp}` | fork of tools/ds4/ds4_moe: per-expert format CLASSES (gu,d pair); per-class launches / pool runs / LRU; chunk = pass per class | CPU path via ds41_generate on the 3-layer file; GPU path compiled, NOT RUN |
| `ds41_generate.cpp` | `-m F --ids ... [--ppl] [--prefill-chunk N] [--serve] [--engram D:D:D] [--rank r.bin] [--vram-lru]` | CPU smoke on the 3-layer file |

Facts found tonight:
- model.py's KV rounding matters: without it 3-layer outputs move cos 1.6e-3 and 16% of routing sets (DwarfStar
  emulates it too).  Engine with it vs golden with it: cos >= 0.9992 (step-function amplification of Q8 noise).
- The CPU native-expert buffers were V4-sized: raised for V4.1 (kNativeActBytes 5840, kNativeHBytes 2628,
  kNativeFFMax 2304, ActQ 5120 wide).  Q2_0 classes need ActQ activations (Strata's Q2_0 kernels), not nact.
- The writer's `--experts-in-place` file is shorter than its last expert region until the quant pass writes it
  (the partial test file was truncate-extended); the real file is complete after step 4.
- NEVER `pkill -f` with a pattern in your own command line: it killed this session's shell once tonight.
- 3-layer test file: /home/mal/AI/v41-dev/v41-L0-2h.gguf (holes = zero experts); goldens in v41-dev/g1, g2, g2q.

40-layer CPU gate (2026-10-11 00:30, dense-only test file /home/mal/AI/v41-dev/v41-all-h.gguf, zero experts, 300 tok):
- real top-k (512 >= visible rows), no act-quant: block outputs cos >= 0.996 every layer (most >= 0.9993), logits
  mean cos 0.99995, top-1 295/300 vs model.py
- top-32 / 8 candidate blocks (forced selection): selected-row sets identical for 272/300 queries at L2, mean Jaccard
  0.98-0.994 at every index layer (L24+ candidate layers incl.); differences = boundary ties (ggml F16 activations)
- engine decode == chunked prefill over 40 layers: logits cos 1.000000, top-1 300/300 (act-quant on, top-32)
- zero experts make |x| ~3e4 by L39 (unnatural), amplifying noise; the real gate is ppl on the filled file.

Real file (2026-10-11 01:55): calibration done 01:29 (teacher head gate ALL/eval ppl 4.915, top-1 0.701; code 1.74,
agent 2.12, reason 5.83, zh 6.83, agentic_long 9.79); tier map = all-Q2_0 bytes, proxy error -17%; layout passes
test_ds41_geometry (9 classes model-wide, 7 in L0); GSQ pass started 01:41 at ~13 min/layer -> ETA ~10:30.
test_ds41_moe (CPU) on the real L0, all 7 classes mixed per token: rel 1.2-1.5e-2 vs dequant+F32 = PASS (the Q8
activation rounding, as DS4's tier).  Held-out eval token files for engine ppl: /home/mal/AI/v41-dev/eval/*.i32
(the head gate's eval sequences: compare engine ppl with stats/head_L40.json and the GSQ pass's QMIX head).

GSQ pass night notes (2026-10-11): L7 OOM'd 3x on a 1.88 GiB fp32 torch.cat of the propagated output (fixed:
chunked into bf16, 4d0..; passed try 4); L13 died on q3k_split (an all-zero super-block whose fp16 d did not round-trip;
fixed: search/re-round, logged; passed on the chain's retry); L14 (Engram table on Team CX2) is I/O-bound like in
calibration (8294 s there) -> ETA ~13:00.  Engram row reads from Team CX2 will matter for engine speed too: measure.

DRIVES (found 2026-10-11 06:45): Crucial1TB, Crucible (Engram L1) and Team CX2 (Engram L14) are mounted fuseblk =
ntfs-3g (FUSE, single-threaded small random reads: why calib/quant L14 took 2.3 h - wchan request_wait_answer); SSD NVME
is kernel ntfs3.  After the chain: remount Crucible + Team CX2 as ntfs3 (udisksctl / mount -t ntfs3) and measure the
engine's Engram rows/token before/after - decode reads 48 rows/token from them.
Size options (Mal asked for a Maya-size ~90 GiB): /home/mal/AI/v41-dev/size_sweep.py - calibration proxy with
prune-or-tier per expert: 152 GiB current mix 1.88 (all-Q2 2.17); prune+mixed 120 GiB 0.37, 99 GiB 0.45, 89 GiB 0.54
(57% pruned), 79 GiB 0.68.  Pruning cost only measured on the calibration corpus (OOD risk) -> recommended: keep the
152 file, soft-prune at runtime (route-bias penalty on the coldest experts, GLM's recipe), decide on a smaller file
after real ppl + tok/s.

### Engine next steps (in order)
1. When the layout exists (step 3 of the chain): `test_ds41_geometry` on the real file; full 40-layer dense gate on
   the CPU with zero experts (golden `--zero-experts` is NOT written yet: experts are being filled during step 4 -
   race), small S (~200) so ratio-1 + all index layers run; DS41 candidate logic at small S needs an env override of
   candidate_topk_blocks on both sides (not written).
2. GPU (after step 4 finishes, ~07:00, Mal's GPU): `test_ds41_ops --cuda` (bit-exactness of the rounding on CUDA),
   then `build-ds4-gpu/ds41_generate --backend cuda --experts gpu --prefill-chunk 512 --ppl` through memguard on a
   held-out text; compare ppl/top-1 with the calibration head gate (`stats/head_L40.json`); act-quant on/off ppl.
3. Speed: router predictor (prefetch), rank file from calibration counts (`stats/LNN.json` experts[e].n ->
   int32 pairs), vram-lru, arena budget, chunk MMQ, decode graph reuse; target > GLM (25 tok/s decode, 400 tok/s
   prefill); 512K context (caches: ~1.5 GB at 512K with BF16 storage - set kv_type BF16).
4. Serving: ds41-engine.sh / configs / strata-ds41-quetza wrapper (copy GLM's); tool-call + coherence checks.
