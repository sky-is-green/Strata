# DwarfStar (antirez/ds4 @fc80bd6) and Strata forks: V4.1-specific findings, 2026-10-10

Checked: antirez/ds4 (fc80bd6, 2026-10-07: the same commit our writer copied names from), VecSzn/PolyStrata,
mw00/project-maya, bodhi37/strata (Qwen/GLM only, no DeepSeek code), sectilelabs (GInfer, repo still empty).
**Only DwarfStar has DeepSeek V4.1 code.**  Paths below are in the ds4 repo.

## Use, ranked

1. **Official V4.1 quality fixtures: our ship gate.**  `gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router/`
   (38 MB): 112 held-out prompts (100 short + 12 long, up to 23K tokens) with official DeepSeek API continuations
   (thinking off, temp 1, top-20 logprobs, 64 tokens), plus DwarfStar's own scores for its Q2 builds
   (`results/base-*.tsv`, `fix-*.tsv`: per-case teacher-forced NLL, API top-1 agreement, top-20 recall).
   Plan: once ds41_generate can score, run the same manifest on our GSQ file -> per-case NLL / top-1 agreement
   directly comparable with DwarfStar's Q2 (`api_top1_rate`, `avg_nll`).  Scorer: `score_official` +
   `compare_scores.py`.  Caveat: our source is the dealignai UNCENSORED checkpoint, so it drifts from the official
   API a little on its own; compare quants against each other, and ideally score the BF16/FP4 teacher too
   (the head gate's teacher numbers are the in-house version of this).
2. **Mixed-layer evidence (`gguf-tools/mixed/README.md`).**  On V4 Flash, upgrading only the LAST SIX layers'
   routed experts (37-42 of 43) from 2-bit to Q4 made a 91 GB file measurably closer to full Q4 (153 GB) than plain
   2-bit.  Independent support for edge-weighting late layers; check what our `v41_alloc.py --edge-weight` gives
   the last layers of V4.1 (40 layers -> ~34-39) before the export.
3. **Validator `gguf-tools/deepseek41_validate_gguf.py`.**  Audits a complete V4.1 GGUF's layout and re-encodes
   sampled experts against the source; Engram as 264-B rows in a disk shard.  Our uniform-type tensors and dense
   half use DwarfStar names, so its layout checks are a free second opinion on our writer.  It will NOT accept our
   per-expert mixed tensors (`ffn_<role>_exps.<type>.weight` + type maps): DwarfStar has no per-expert tier
   format, mixing is whole-layer only (splice tool).  So **DwarfStar cannot load our mixed file**; it could load
   an all-uniform control file.
4. **Reasoning effort / think level (serving shim).**  DwarfStar's V4.1 takes `--think-level 1..100`
   (0 = off; `--think` = 75, `--think-max` = 100); server maps OpenAI `reasoning_effort` names and
   `chat_template_kwargs` (`ds4_server.c:1037-1170`, `ds4_deepseek41_reasoning_effort_text`, render at
   `ds4_server.c:3377`).  Copy that mapping into serve_ds41.py so clients' `reasoning_effort` behaves the same.
5. **Tool-text escaping (`ds4_tool_text.h`).**  Tool bodies escape only their own closing delimiter (`&lt;` +
   the rest of the tag, `&amp;` chains).  Our DSML -> Qwen `<tool_call>` rewriter must unescape the same way, or
   tool arguments containing the closing tag break.
6. **Engram (`ds4_engram.[ch]`, 327 lines).**  4-gram, 8 heads x 24 columns, 256-dim rows, 264-B rows
   (row + 8-B scale, = our plan's 2 preads/row stored adjacently in their own shard), read through a separate
   UNCACHED fd (never mmap); hash history snapshot = 3 tokens (copying it snapshots the hash state: relevant to
   our multi-turn snapshot/restore).  Port the hash from here as the CPU reference for P4.
7. **Kernels**: `ds4_deepseek41_cuda.cuh` (633 lines) + `ds4_cuda.cu` - V4.1 text on CUDA (target DGX Spark);
   reference for P3 (mHC, compressor, indexer).  V4.1 + CUDA requires `--power 100`.  DSpark spec decode exists
   for V4 Flash (needs a support GGUF); no V4.1 DSpark found.

## Non-DeepSeek (for strata-glm, FYI)
PolyStrata: cold experts computed on the CPU from the mmap'd GGUF instead of skipped (vs our `--skip-miss`);
drafting from the model's draft block + repeats of the conversation; first-64-token parity vs llama.cpp.
