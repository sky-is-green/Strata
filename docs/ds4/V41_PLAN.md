# DeepSeek-V4.1-Flash: our own GSQ-RCO quant + Strata-DS4 engine support (plan, 2026-10-09)

Source checkpoint: `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8` (510.3 GB; dense weights abliterated, routed experts,
engram, router, norms byte-identical to the base). Size study: [V41_FEASIBILITY.md](V41_FEASIBILITY.md).

## What V4.1 changes vs V4-0731 (from DeepSeek's `inference/model.py`, mirrored in ~/AI/gsq-rco/dsv41-inference)

| | V4-0731 (our engine today) | V4.1 |
|---|---|---|
| shape | 43 layers, hidden 4096, 256 experts | 40 layers, hidden 5120, 384 experts top-6 + 1 shared, moe_inter 2304 |
| router | sqrtsoftplus + bias | same, plus `bias_vl` for image tokens; route_scale 1.5 |
| compressed attention | ratios 4 (CSA) / 128 (HCA), every layer its own compressor | ratios 0 / 2 / 1; **kv_source_layers [2,8,14,20]**: later layers with the same ratio READ the source's compressed KV; **index_source_layers [2,8,14,20,24,28,32,36]** share one indexer's top-k |
| indexer | one level, top-k 512 | **two levels**: layer 20 picks 2048 candidate blocks of 8, later indexers top-512 only inside those |
| compressed-KV storage | fp8 | fp4 (groups of 16, E4M3 scales); index keys fp4 (groups of 32, E8M0) |
| hyper-connections | hc 4, pre-mix per sublayer | hc 4, **single-pass**: each sublayer's `hc_mixes` produces the NEXT sublayer's pre-mix (attention uses the previous FFN's) |
| engram | none | layers 1 and 14: n-gram hash (orders 2-4, 8 heads = 24 rows) into a 384M x 256 fp8 table, `wkv` -> per-hc-copy key + shared value, sigmoid(signed-sqrt dot) gate added to the residual BEFORE the block |
| swiglu clamp | one limit | per-layer arrays (exp / shexp) |
| MTP | 1 layer | DSpark: 3 layers on target layers 37-39, block 5, its own 128-expert MoE + Markov / confidence heads |
| FP8 block scale | 128x128 | **32x32** (getting this wrong silently rescales every weight) |

## Research: what the video / DwarfStar do that we can use

DwarfStar (antirez/ds4, local ~/AI/ds4-ref updated to fc80bd6) already runs V4.1 (`ds4_engram.c`,
`ds4_deepseek41_cuda.cuh`, `gguf-tools/deepseek41_quantize.py`) - our C reference for the port, next to DeepSeek's
Python model (golden numbers).  Takeaways:
1. **Engram never resident**: rows are pread from the GGUF (posix_fadvise RANDOM + NOREUSE).  The hash depends only on
   token ids, so the 48 row reads/token are **issued before the forward starts** (`ds41_engram_prefetch_start`) and
   joined at layers 1/14 - SSD latency fully hidden.  12 KB/token.  Put the table on any SSD with good random reads.
2. **Their CUDA SSD expert cache is a plain global LRU** (one VRAM slot pool, clock stamps, "protect all hits before
   evicting for a miss", copy from the mmap'd GGUF).  Strata-DS4 already does more (VRAM LRU per layer + 70 GiB RAM
   arena + predicted prefetch + soft-prune + arena-adapt) - nothing to take except the hit-protection rule.
3. **Streamed prefill = wide layer batches**: process a whole prompt chunk per layer and read that layer's experts once,
   overlapping the next layer's reads (Metal) or staging into the device cache (CUDA); "resident prefix layers"
   (`--ssd-streaming-full-layers N`) keep whole early layers resident.  Our chunked prefill (`--prefill-chunk 4096
   --chunk-mmq --chunk-prestage`) is the same idea - extend prestage to read layer L+1 from NVMe while L computes.
4. **Prompt / KV cache to disk** (`--kv-disk-dir`): short appends to a long conversation skip re-prefill (GLM: 30.8 ->
   2.9 s).  Worth adding once V4.1 decodes.
5. Video numbers (5090 + 128 GB + Gen5 SSD, native 4-bit, 5.1 tok/s SSD-bound; 8-ch DDR4 TR Pro resident 3-bit 20 ->
   25.6 tok/s with DSpark): when experts spill, **disk bandwidth is the ceiling, not the GPU**; when resident, RAM
   bandwidth.  Our lever = make the working set fit (GSQ ~1.8-2.25 bpw + soft prune), then DSpark.

## Plan A - GSQ-RCO quant pipeline (~/AI/gsq-rco, tools/v41)

Target formats Strata can already run: **Q2_0** (= GSQ's native symmetric 2-bit grid {-2,-1,0,1} x fp16 scale per 64;
2.25 bpw), Q2_K / Q3_K (GSQ "refine in K-quant format" mode; pfeifferj's V4.1 GSQ-RCO uses Q2_K x26 / Q3_K x14
layers, dense Q8_0).  Sub-2-bit (to fit ~120-130 GB) needs GSQ ternary -> **TQ1_0 (1.69 bpw), no Strata kernel yet**.
Budget examples (543.6B routed weights): all Q2_0 153 GB; 50% TQ1_0 + 50% Q2_0 = 1.97 bpw, 134 GB; RCO-chosen mix at
1.85 bpw = 126 GB.

0. `tools/v41/fetch_v41.sh DEST` - resumable, O_DIRECT sha256-verified download (tested).
1. **v41_ref** (single GPU, layer streaming): DeepSeek's model.py with torch fallbacks for the tilelang kernels
   (fp8/fp4 dequant matmul, sparse_attn gather, hc sinkhorn), weights loaded per layer straight from the HF
   safetensors (no TP convert).  Runs calibration tokens layer by layer, saving each MoE block's input + routing
   (activations on disk; 0.5M tokens x 5120 bf16 = 5 GB per layer-input, hc stream x4).  Doubles as the golden
   oracle for the engine port (like ds4_ref).
2. **Per-expert GSQ**: for each layer, each expert gets its routed tokens; GPTQ init (Hessian from those tokens) ->
   Gumbel-softmax refinement (GSQ's quantizer modules, Lion, 10 epochs) against the FP4 teacher output.  Shared
   expert + dense: Q8_0 (as everyone does).  Engram: stays FP8 rows (lossless, on SSD).
3. **RCO-lite allocation**: per-tensor sensitivity (teacher-KL / output-MSE at each candidate type, plus REAP saliency
   from the routing stats we already collect) -> exact byte budget knapsack.  Full RCO manifold search is optional.
4. **Export** a `deepseek41` GGUF with pfeifferj's / llama.cpp's tensor names (seen in their header), so their patched
   llama.cpp (runtime/ in their repo) is a second oracle.
Compute estimate (4090): GPTQ+GSQ ~15-40 GPU-hours total (per-expert Gumbel steps are memory-bound); calibration
forward a few hours.  Runs in blocks shared with strata-glm.  Calibration: en+zh+code (make_reap_prompts.py) +
reasoning traces, ~256-512 x 2048 tokens.

## Plan B - V4.1 in Strata-DS4

Order (each step gated against v41_ref goldens on a mini fixture first, then the real model):
1. GGUF loader for `deepseek41` names + KV (hidden 5120, 384 experts, per-layer clamps, source-layer lists).
2. Dense half: single-pass HC ordering, ratio 0/2/1 compressors, KV/index source sharing (non-source layers bind the
   source's caches), two-level indexer (candidate blocks at layer 20), fp4 compressed-KV storage.
3. Engram op: CPU hash (port ds4_engram.c logic or DeepSeek engram.py), async pread prefetch at token start, GPU
   gate/add at layers 1/14.
4. MoE tier: 384 experts x 40 layers, expert size from the GSQ GGUF types (Q2_0 / TQ1_0 / Q2_K), soft-prune +
   arena-adapt carried over.
5. DSpark (3-layer draft) after decode works.
