# DeepSeek-V4.1-Flash: GSQ-RCO quant + Strata-DS4 port - HANDOFF / RESUME (2026-10-09 ~18:40)

Read this first, then [V41_PLAN.md](V41_PLAN.md) (design, V4 -> V4.1 deltas, all research) and
[V41_FEASIBILITY.md](V41_FEASIBILITY.md) (sizes, speed estimates).  Repo: `~/AI/Strata-DS4`, branch `deepseek4`
(local, ahead of the public `ds4/main` at 924516a - nothing of today is pushed; push only when Mal says).

---------------------------------------------------------------------------------------------------------------------
## Paste-able resume prompt

> You are continuing the DeepSeek-V4.1-Flash work in ~/AI/Strata-DS4 (branch deepseek4).  Read
> docs/ds4/V41_RESUME_PROMPT.md, then docs/ds4/V41_PLAN.md.  First: rename yourself on the relay to
> `strata-ds4-gpu` (relay_rename) and check relay_peers - `strata-glm` shares the GPU (lock
> ~/.quetza-data/conductor/ds4-gpu.lock, <=10-12 min blocks, announce before using it).  Then check the download:
> `tail /media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/sequence.log` and the .ok markers.  Run
> `tools/v41/shard_check.py` on any newly verified shard.  Next task: build the calibration runner (step 1 of
> "Next steps" below), then per-expert GPTQ/GSQ, then the PTQ1_0 CUDA kernel.  Work solo (no conductor workers),
> measure before concluding, commit each verified step.

---------------------------------------------------------------------------------------------------------------------
## Goal (Mal's decisions, all made)

1. **Our own quant of DeepSeek-V4.1-Flash, uncensored**, from `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`
   (510.3 GB; MPOA abliteration of dense weights only - routed experts, engram, router, norms are byte-identical to
   the base; MMLU 86.96 -> 82.74, HarmBench 100% at effort off AND max).
2. **GSQ (IST-DASLab) quantization of the routed experts, mixed tiers, budget-neutral at all-Q2 size** (~153 GB of
   experts, 2.25 bpw average):  sensitive experts **Q3_K** (3.44 bpw), default **Q2_0** (2.25, = GSQ's native 2-bit
   grid), weakest **PTQ1_0 ternary** (1.75).  Every 1% moved to ternary pays for 0.42% moved to Q3_K.  Mal: "don't
   lobotomize it" - all-Q2 is the fallback, the mix only ships if it beats an all-Q2 control on held-out per-domain
   ppl (incl. never-calibrated languages, as in FINDINGS s37).  Dense / shared expert / router / engram stay >= Q8_0.
3. **Hadamard rotation for the ternary tier** (and test it on Q2): rotate the weight's input dim by a block-128
   Hadamard with random signs before GSQ (calibration activations rotated the same way), runtime FWHT on the
   activation.  Per-tensor flag in the GGUF.
4. **Engram (203 GB = 189 GiB) never in RAM**: rows read from SSD (48 rows / 12 KB per token, prefetched at token
   start because the hash depends only on token ids).  Mal will move it to NVMe after GSQ (needs ~145 GiB freed on
   NVME1TB; SSD NVME's 163 GiB is for the expert pack).
5. **Run it in Strata-DS4** (port V4.1 into our engine, reuse soft-prune + arena-adapt + VRAM LRU + RAM arena).
   Expected 6-10 tok/s at first (NVMe-bound: ~55-60% of the pack fits in ~85-90 GB fast memory), then DSpark.

---------------------------------------------------------------------------------------------------------------------
## Where things are on disk

| what | path |
|---|---|
| source shards (all but engram) | `/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/` (Crucial 1TB, SATA, ~414 GiB free at start) |
| engram layer 1 (shard 47, 101.5 GB) | `/media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/` |
| engram layer 14 (shard 48, 101.5 GB) | `/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/` |
| planned GSQ output (~151 GiB) | `/media/mal/SSD NVME/...` (NTFS Windows drive, PCIe4, 163 GiB free; mounted with `udisksctl mount -b /dev/nvme0n1p3`) |
| GSQ + RCO repos | `~/AI/gsq-rco/GSQ`, `~/AI/gsq-rco/RCO` (cloned, not installed; GSQ wants uv + flash-attn; we reuse its quantizer modules) |
| DeepSeek's V4.1 reference code | `~/AI/gsq-rco/dsv41-inference/` (model.py, engram.py, kernel.py = tilelang, convert.py, ...; MIT) |
| DwarfStar (antirez/ds4) = V4.1 C reference | `~/AI/ds4-ref` (checked out fc80bd6, 2026-10-07): `ds4_engram.c`, `ds4_deepseek41_cuda.cuh`, `gguf-tools/deepseek41_quantize.py` |
| msuiche GLP control vector (alternative uncensor, base-model only) | `~/AI/Strata-DS4-private/v41/glp.gguf` (39 rank-1 dirs, alpha 0.5) |
| video transcript (Local Explained, V4.1 hardware) | `~/AI/Strata-DS4-private/v41/video-UMrKQWelWp0-transcript.txt` |
| pfeifferj V4.1 GSQ-RCO (3.0 bpw, Q2_K x26 / Q3_K x14 + Q8_0 dense) | HF `pfeifferj/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF` - its `runtime/` has a patched llama.cpp = second oracle |

Shard layout (one layer per shard): 1 vision, 2 embed, **3..42 = layers 0..39** (dense + 384 experts each),
43 head+norm, 44-46 DSpark/MTP, **47 = engram layer 1, 48 = engram layer 14**.

## Download (running in the background, survives the session)

- Driver: `tools/v41/fetch_sequence2.sh` (setsid/nohup), log `.../Crucial1TB/.../sequence2.log` (the earlier
  `fetch_sequence.sh` / `sequence.log` are superseded); each file's own progress in `fetch.log` of its directory.
  **Order (matched to GSQ, 2026-10-09 20:00):** 47 -> **backbone layers 2-13 (shards 5-16)** -> 48 -> the rest of
  the backbone (17-46).  Rationale: 47 is the only Engram shard GSQ needs for layers 0-13, so those layers'
  backbone shards go first so GSQ can run while downloading; 48 is packed in before layer 14.  The script
  self-heals: if it starts and no 47 fetch is running (re-run after reboot), it fetches 47 itself.
- `tools/v41/fetch_v41.sh DEST [REPO]` - resumable (curl -C -), `.ok` marker per verified file, sha256 via
  **O_DIRECT** (page cache lies on this box), env JOBS=1 FIT=1 MARGIN_GB=10 ONLY=regex SKIP=regex ORDER=name|size.
  Re-run the same command any time; it skips verified files and resumes partial ones.
- To restart by hand after a reboot (or just re-run `fetch_sequence2.sh`, which does all of this in order):
  ```
  cd /media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 && JOBS=1 FIT=1 ONLY='model-00047-of' ~/AI/Strata-DS4/tools/v41/fetch_v41.sh "$PWD"
  cd /media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 && JOBS=1 FIT=1 ONLY='model-000(0[5-9]|1[0-6])-of' ~/AI/Strata-DS4/tools/v41/fetch_v41.sh "$PWD"
  cd "/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8" && JOBS=1 FIT=1 ONLY='model-00048-of' ~/AI/Strata-DS4/tools/v41/fetch_v41.sh "$PWD"
  cd /media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 && JOBS=1 FIT=1 SKIP='model-0004[78]-of' ~/AI/Strata-DS4/tools/v41/fetch_v41.sh "$PWD"
  ```
- Speed ~13-14 MB/s (one dip to 1 MB/s): ~9 min per 7.4 GB shard, ~2 h per engram shard, ~11-12 h total.
- **Never edit a running bash script in place** (bash reads it incrementally): write a copy and `mv` over it.
- The HF token is visible in `ps` (curl -H) - don't paste process listings into anything shared.

---------------------------------------------------------------------------------------------------------------------
## Built and verified today (commits on `deepseek4`)

| commit | what | verified |
|---|---|---|
| 28dd12dc, 6ce93ed3, 626dbac0 | `docs/ds4/V41_PLAN.md` - plan, deltas, mixed tiers, research rounds 1-2 | - |
| 28dd12dc, c67b47ef | `tools/v41/fetch_v41.sh`, `fetch_sequence.sh` | tiny-repo test incl. corrupted-file recovery |
| f3bc404d | `tools/v41/torch_kernels/kernel.py`: torch stand-in for DeepSeek's tilelang `kernel.py` (act_quant / fp4_act_quant / fp8_gemm / fp4_gemm / sparse_attn with sink / hc_split_sinkhorn, rounding rules copied from the kernels) | `tools/v41/tests/test_torch_kernels.py` 15/15; `tests/selftest_model_cpu.py` runs DeepSeek's model.py (prefill + decode + DSpark) on it |
| e1076fc3 | `tools/v41/shard_check.py` - per-shard validation + per-expert weight-space tier errors -> `bench/v41/shard-check/` | layer 0: 2,334 tensors, 0 issues |
| 50e9428d | **PTQ1_0 ternary on the CPU**: `include/strata/kernels/cpu/ptq1_0.hpp`, `src/kernels/cpu/ptq1_0.cpp` (pack / unpack / quantize_ref / dequant, Q8_0 dot scalar + AVX2), wired into `native_expert.cpp` via a type-trait wrapper | `tests/cpu/ptq1_0_test.cpp` 16/16 (target `ptq1_0_test` in build-ds4): bytes identical to llama-tq's reference quantizer at n = 5120 / 2304 / 128, AVX2 == scalar bitwise, == float reference, mutation caught |
| 42c77429 | ds4_dense: stale-graph unbind compares buffer pointers (GLM's gallocr audit) | ASan CPU gate swa16/tame, shared + per-graph allocator: PASS |
| c3ca0410 | `docs/ds4/V41_FEASIBILITY.md` | - |
| 9221882a | `docs/ds4/RESEARCH_2026-10-09.md` (Infernix, Strata 0.1.41, community ideas for DS4) | - |

### Facts established (measured or read from source)
- **V4.1 has NO per-head q RMS norm** (V4 has; JigSawPT's port kept it = fluent text but layer-39 corr 0.78).
- FP8 dense: **32x32** E8M0 block scales.  FP4 experts: int8-packed, **low nibble = even index**, E2M1 codes
  {0,.5,1,1.5,2,3,4,6, negatives}, one E8M0 scale per 32 along K.  Tensor names: `layers.N.attn.{wq_a,wq_b,wkv,
  wo_a,wo_b}.{weight,scale}`, `layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}` (I8 [2304,2560] / [5120,1152]),
  `ffn.shared_experts.*`, `ffn.gate.{weight,bias,bias_vl}`, `hc_{attn,ffn}_{fn,base,scale}`, `attn.attn_sink`.
- **Weight-only priors are useless for allocation**: all 384 experts of layer 0 have the same RTN error to 3
  decimals (ternary 0.443 / Q2 0.378 / Q3 0.183) - FP4 training gives every expert the same per-32 distribution.
  Allocation must come from calibration (activation-weighted error x REAP saliency).
- `transformers` has no deepseek_v41 (only v4) -> reference = DeepSeek's own model.py + our torch kernels.
- Strata's native expert kernels already run Q2_0, Q2_K, Q3_K, IQ2_XXS, IQ1_M; PTQ1_0 now on the CPU only.
- The vendored ggml (`third_party/llama.cpp`) has GGML_TYPE_COUNT = 43: **type 143 must never reach ggml**.
- PTQ1_0 = id 143 in both llama-tq and ik_llama.cpp (#2550); block: qs[0..15] 5 trits/byte (values m+16n),
  qs[16..23] 5 trits/byte (80+m+8n), qh[0..1] 4 trits/byte (120+h+2n, shifted to the top trit), fp16 d; 28 B.
- DwarfStar's CUDA SSD expert cache is plain global LRU (we are ahead); take its engram prefetch-at-token-start.
- sybil-solutions (3090 + 503 GiB 8-ch DDR4): ~20 tok/s to 261K, experts all in RAM + AVX2 CPU tier (needs their
  RAM bandwidth).  JigSawPT (5090 + 126 GiB, NVMe streaming, MXFP4): 5.12 tok/s new content.

---------------------------------------------------------------------------------------------------------------------
## Next steps (in order)

1. **Calibration runner** `tools/v41/v41_calib.py` (needs GPU blocks; CPU works for a smoke test):
   - Build DeepSeek's `Transformer` layer by layer with `sys.path[:0] = [tools/v41/torch_kernels, ~/AI/gsq-rco/dsv41-inference]`;
     load each layer's tensors straight from its shard (safetensors mmap, no TP convert), keep FP8/FP4 storage,
     torch>=2.10 is NOT installed (2.9.1 has float4_e2m1fn_x2 + float8_e8m0fnu - enough; experts can also be
     dequantized per expert with `kernel.dequant_fp4_weight`).
   - Embed (shard 2) -> for L in 0..39: engram (L=1: shard 47, L=14: shard 48; rows by safetensors offset, never the
     whole table) -> Block L -> save the MoE input (post ffn_norm), routing (indices + weights) and the block output
     (hc stream [T, 4, 5120]) to disk for the next layer.  Calibration text: `tools/ds4/make_reap_prompts.py`
     (en + zh + code) + reasoning traces; start with ~64 x 2048 tokens for the smoke run, then 256-512 x 2048.
   - Gate it like ds4_ref: compare a few positions' final logits against pfeifferj's patched llama.cpp or DwarfStar
     on the same tokens, and check routing agreement (the gate amplifies errors 14x - JigSawPT).
   - Accumulate REAP saliency per (layer, expert) on the way (same formula as tools/ds4 / Strata-GLM).
2. **Per-expert GPTQ -> GSQ** using the saved routed inputs: GSQ's quantizer modules from `~/AI/gsq-rco/GSQ/src/quantization`
   (2-bit grid = Q2_0 with group 64; ternary with group 128 = PTQ1_0; 3-bit -> project onto Q3_K).  Record each
   expert's activation-weighted error at all three widths (GPTQ init is cheap; run GSQ refinement only at the
   chosen width) -> RCO-lite knapsack on error x saliency under the exact byte budget -> second pass at the chosen
   widths.  Optional Hadamard rotation per tensor (measure both).
3. **PTQ1_0 on the GPU**: add to `src/kernels/cuda/native_mmvq.cu` (+ header `native_mmvq.hpp`: supported(143),
   weight_bytes, dispatch) with q8_1 activations: per 128-block, 4 q8_1 sub-blocks, dp4a of (digit-1) with q8.
   Reference: llama-tq `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh` (MIT).  Then the FWHT op for rotated tensors, then
   prefill (dequant -> fp16 GEMM is exact for ternary).  Later: R8 row-interleaved CPU repack (ik #2564: 3.4x).
4. **GGUF writer** for our pack: `deepseek41` names as in pfeifferj's header (blk.N.ffn_{gate,up,down}_exps,
   attn_q_a/q_b/kv/output_a/output_b, engram_embd/wkv/q/k, hc_attn_*, ...), per-tensor types, rotation flags; engram
   copied as raw FP8 rows + scales (or Q8_0).  Validate by loading in DwarfStar / pfeifferj's runtime if possible.
5. **Strata-DS4 V4.1 port** (V41_PLAN.md "Plan B"): loader, dense half (single-pass HC, ratio 0/2/1 compressors,
   kv/index source sharing, two-level indexer, fp4 compressed KV, NO q norm), engram op with prefetch, 384-expert
   tier (Q2_0 / Q3_K / PTQ1_0), then DSpark.  Golden numbers from step 1's runner.
6. Validation: per-domain held-out ppl of mixed vs all-Q2_0 control vs FP4 original; then speed with soft-prune.

## Other open items (DS4 V4, from earlier today)
- b33: DS4 V4 decodes 21.2 tok/s at both 256K and 512K (`--pos-offset` probe, iq4_nl KV); soft prune neutral in the
  short probe - needs a real 64K document run (prefill + decode, unpruned vs pruned).
- Strata upstream v0.1.41 merge: worktree `~/AI/Strata-DS4-0141` (branch `ds4-on-0141`), 8 files / 14 hunks;
  graft recipe in RESEARCH_2026-10-09.md; Mal undecided (full merge vs cherry-pick q8_1 clamp fixes #606/#1448).
- Ideas queued for DS4 (RESEARCH_2026-10-09.md): rank-gated `--arena-adapt` admission, persisted expert state,
  elastic KV / 32K VRAM KV window for 512K, adaptive MTP draft length, batched NVMe reads.
- MiMo models were deleted on Mal's request (sources: ~/AI/Strata-MiMo/docs/mimo/MODEL_FILES.md).  Mal wants
  MiMo-Audio 7B back later (`hf download XiaomiMiMo/MiMo-Audio-7B-Instruct` / `-Tokenizer`, download to ext4 first -
  `hf` hangs on exFAT file locks - then move to NVME1TB/Models/Mimo-Audio/).

## Working rules (this box / Mal)
- GPU shared with `strata-glm` over quetza-relay; lock `~/.quetza-data/conductor/ds4-gpu.lock`; full-model runs via
  `bench/ds4-2026-10-08/q.sh` + `tools/ds4/memguard.sh` (never a CPU-only full-model load: froze the box once).
- Run CPU work at `nice -n 15`, 4 threads, while GLM benchmarks.  Don't rebuild build-ds4-gpu while a batch is queued.
- Never `pgrep -f`+kill a pattern that appears in your own command line (killed own shell once); use PIDs/PGIDs.
- Verify downloads with O_DIRECT reads; RAM has a known bad lane (soft-offlined).
- Commits end with the Co-Authored-By line; no pushes without Mal's OK.
