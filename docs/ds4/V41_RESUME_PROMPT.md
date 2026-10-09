# DeepSeek-V4.1-Flash: GSQ-RCO quant + Strata-DS4 port — HANDOFF / RESUME

**Last updated:** 2026-10-09 ~22:55 by `strata-ds4-gpu`.
**Read order:** this file → [V41_PLAN.md](V41_PLAN.md) (design, V4→V4.1 deltas, all research) →
[V41_FEASIBILITY.md](V41_FEASIBILITY.md) (sizes, speed estimates) → [RESEARCH_2026-10-09.md](RESEARCH_2026-10-09.md)
(community ideas) → `../../V41_PLAN`-adjacent [../../RESEARCH](../RESEARCH_2026-10-09.md).
**Repo:** `~/AI/Strata-DS4`, branch `deepseek4`. Local only — ahead of the public `ds4/main` at `924516a`.
**Nothing from this work is pushed;** push only when Mal says.

---

## Paste-able resume prompt

> You are continuing the DeepSeek-V4.1-Flash GSQ-RCO quant + Strata-DS4 port in `~/AI/Strata-DS4` (branch `deepseek4`).
> Read `docs/ds4/V41_RESUME_PROMPT.md` (this file, all of it), then `docs/ds4/V41_PLAN.md`.
>
> **1. Claim the GPU channel.** `relay_rename` yourself to `strata-ds4-gpu`, then `relay_peers`; `strata-glm` shares the
> single RTX 4090 (see "GPU coordination" below — the lock is `~/.quetza-data/conductor/ds4-gpu.lock`, blocks ≤10–12 min,
> announce on relay before taking it). If you only need CPU, skip this.
>
> **2. Check the download** (it runs detached and survives your session):
> `tail /media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/sequence2.log` and count `*.ok` markers in the three
> model dirs (table in §Download). If it is dead (no `fetch_v41`/`curl` in `ps`), re-run `tools/v41/fetch_sequence2.sh`
> with `setsid nohup … &` — it resumes exactly where it stopped and self-heals shard 47.
>
> **3. Run the per-shard check on any newly verified shard:** `tools/v41/shard_check.py` (see §Built). Cheap, CPU-only,
> but its weight-space errors do NOT rank experts (all identical) — validation only.
>
> **4. Continue the pipeline.** The immediate next task is **step 1: the calibration runner `tools/v41/v41_calib.py`**
> (§Next steps has the full spec). Write it, smoke-test it on CPU (layer 0 needs only shards 2+3, no engram, no GPU),
> then gate it against a reference before trusting it. Then per-expert GPTQ→GSQ (step 2), the PTQ1_0 CUDA kernel
> (step 3), the GGUF writer (step 4), the Strata port (step 5).
>
> **Rules:** work solo (no conductor workers); measure before concluding; commit each verified step (Co-Authored-By
> trailer); no pushes without Mal's OK. Never kill processes by a `pgrep -f` pattern that matches your own command
> line — use PIDs/PGIDs. Run CPU work at `nice -n 15`, 4 threads while GLM benchmarks.

---

## Status snapshot (2026-10-09 22:55)

| item | state |
|---|---|
| Download | **shards 1–10 verified** (+ config, mascot); **shard 11 in flight** (~3.7/7.4 GB). Next: 11–16, then 48, then 17–46 |
| GPU | RTX 4090, **12.5/24.5 GB used, 15% util** — GLM (`glm_generate`, pid 1424967) holds ~12 GB under `ds4-gpu.lock` |
| Calibration runner | **not written yet** (step 1) — this is the immediate next task |
| PTQ1_0 ternary | **CPU done + tested** (commit `50e9428d`); GPU kernel not started |
| GSQ output pack | not started; target drive `/media/mal/SSD NVME` (163 GiB free) |
| Blocker | real calibration/GSQ needs a GPU block; the CPU smoke test does not |

---

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
- Strata's native expert kernels already run Q2_0, Q2_K, Q3_K, IQ2_XXS, IQ1_M; PTQ1_0 is CPU-only so far.
- The vendored ggml (`third_party/llama.cpp`) has `GGML_TYPE_COUNT = 43`: **type 143 must never reach ggml**
  (it would read past the traits table). CPU path wraps it via a local traits object in `native_expert.cpp`.
- PTQ1_0 = id 143 in llama-tq and ik_llama.cpp (#2550). Block = 28 B: `qs[0..15]` 5 trits/byte (values m+16n),
  `qs[16..23]` 5 trits/byte (80+m+8n), `qh[0..1]` 4 trits/byte (120+h+2n, shifted to the top trit), fp16 `d`.
- DwarfStar's CUDA SSD expert cache is plain global LRU (we are ahead); **take** its engram prefetch-at-token-start.
- Speed references: sybil-solutions (3090 + 503 GiB 8-ch DDR4) ~20 tok/s to 261K (experts all in RAM + AVX2 CPU tier);
  JigSawPT (5090 + 126 GiB, NVMe streaming, MXFP4) 5.12 tok/s on new content.

---

## Next steps (in order)

1. **Calibration runner `tools/v41/v41_calib.py`** — the immediate task. Needs GPU blocks for the real run; CPU is
   enough for a smoke test.
   - `sys.path[:0] = [tools/v41/torch_kernels, ~/AI/gsq-rco/dsv41-inference]`; build DeepSeek's `Transformer`
     **layer by layer**, loading each layer's tensors straight from its shard (safetensors mmap; **no** TP convert),
     keeping FP8/FP4 storage. torch is **2.9.1** (not ≥2.10): it has `float4_e2m1fn_x2` + `float8_e8m0fnu` — enough;
     experts can also be dequantized per expert via `kernel.dequant_fp4_weight`.
   - Embed (shard 2) → for L in 0..39: **engram** (L=1 → shard 47, L=14 → shard 48; rows by safetensors offset,
     **never** load the whole table) → Block L → save the MoE input (post `ffn_norm`), the routing (indices+weights),
     and the block output (hc stream `[T,4,5120]`) to disk for the next layer.
   - Calibration text: `tools/ds4/make_reap_prompts.py` (en + zh + code) + reasoning traces; start with ~64 × 2048
     tokens for the smoke run, then 256–512 × 2048.
   - **Gate it** like `ds4_ref`: compare final logits at a few positions against pfeifferj's patched llama.cpp or
     DwarfStar on the same tokens, and check **routing agreement** (JigSawPT: the gate amplifies errors ~14×).
   - Accumulate **REAP saliency** per (layer, expert) on the way (same formula as `tools/ds4` / Strata-GLM).
2. **Per-expert GPTQ → GSQ** on the saved routed inputs. Quantizer modules from
   `~/AI/gsq-rco/GSQ/src/quantization` (2-bit grid = Q2_0 group 64; ternary group 128 = PTQ1_0; 3-bit → project onto
   Q3_K). Record each expert's activation-weighted error at all three widths (GPTQ init is cheap; run GSQ refinement
   only at the chosen width) → RCO-lite knapsack on `error × saliency` under the exact byte budget → second pass at the
   chosen widths. Optional Hadamard rotation per tensor (measure both).
3. **PTQ1_0 on the GPU**: add to `src/kernels/cuda/native_mmvq.cu` (+ header `native_mmvq.hpp`: `supported(143)`,
   `weight_bytes`, dispatch) with q8_1 activations: per 128-block, 4 q8_1 sub-blocks, dp4a of `(digit-1)` with q8.
   Reference: llama-tq `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh` (MIT). Then the FWHT op for rotated tensors, then prefill
   (dequant → fp16 GEMM is exact for ternary). Later: R8 row-interleaved CPU repack (ik #2564: 3.4×).
4. **GGUF writer**: `deepseek41` names as in pfeifferj's header (`blk.N.ffn_{gate,up,down}_exps`,
   `attn_q_a/q_b/kv/output_a/output_b`, `engram_embd/wkv/q/k`, `hc_attn_*`, …), per-tensor types, rotation flags;
   engram copied as raw FP8 rows + scales (or Q8_0). Validate by loading in DwarfStar / pfeifferj's runtime if possible.
5. **Strata-DS4 V4.1 port** (V41_PLAN.md "Plan B"): loader; dense half (single-pass HC, ratio 0/2/1 compressors,
   kv/index source sharing, two-level indexer, fp4 compressed KV, **no q norm**); engram op with prefetch;
   384-expert tier (Q2_0 / Q3_K / PTQ1_0); then DSpark. Golden numbers from step 1.
6. **Validation**: per-domain held-out ppl of mixed vs all-Q2_0 control vs the FP4 original; then speed with soft-prune.

---

## Other open items (DS4 V4, from earlier today)

- **b33:** DS4 V4 decodes 21.2 tok/s at both 256K and 512K (`--pos-offset` probe, `iq4_nl` KV); soft-prune neutral in
  the short probe — **needs a real 64K-document run** (prefill + decode, unpruned vs pruned).
- **Strata upstream v0.1.41 merge:** worktree `~/AI/Strata-DS4-0141` (branch `ds4-on-0141`), 8 files / 14 hunks;
  graft recipe in `RESEARCH_2026-10-09.md`; Mal undecided (full merge vs cherry-pick the q8_1 clamp fixes
  #606/#1448).
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
