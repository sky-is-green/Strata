---
license: mit
base_model:
- dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8
- deepseek-ai/DeepSeek-V4.1-Flash
tags:
- deepseek
- deepseek-v4.1
- moe
- quantized
- 2-bit
- mixed-precision
- gptq
- uncensored
- strata
- consumer-gpu
---

<div align="center">

# DeepSeek-V4.1-Flash · GSQ mixed-tier (152 GiB)

**A 552B-parameter MoE running on one gaming PC — RTX 4090 (24 GB) + 90 GB RAM + NVMe**

**Prefill ~250 tok/s (exact) · Decode ~26 tok/s (~27 with the opt-in ggml patch)**

Runs on **[Strata-DSV4.1](https://github.com/Indras-Mirror/Strata-DSV4.1)**, an engine built for exactly this model and
this class of machine, on top of [Strata](https://github.com/Niko1221/Strata).

</div>

> **Upload in progress (started 2026-10-12):** the 16 model parts arrive over about a day at home upload speed.
> `SHA256SUMS` already lists every part; `reassemble.sh` refuses to join until all of them are present and intact.

> **Experimental research release** from a single developer, measured on one PC.
> This file is **not** a llama.cpp / Ollama / LM Studio GGUF. It uses per-expert quant types those runtimes don't load.
> Use it with Strata-DSV4.1.

## What this is

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) has 40 layers with 384 routed experts each
(top-6 per token), a shared expert, Hyper-Connections, CSA sparse attention, and an **Engram** n-gram memory. Even at FP8
its routed experts alone are far bigger than any consumer PC's RAM.

This repository holds a **GSQ ("graded sensitivity quantization") mixed-tier** version of
[dealignai's abliterated FP8 checkpoint](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8):

- **Routed experts: ~2.0–2.5 bits per weight on average, chosen per expert.** Each of the 15,360 experts gets its
  own format from calibration statistics:
  **PTQ1_0** (ternary, 1.75 bpw), **Q2_0** (2.25 bpw) or **Q3_K**. Experts that matter more (and earlier layers) get
  richer formats; rarely-used experts get the cheapest ones.
- **Dense half at Q8_0:** attention, shared expert, output head. The embedding, compressor and indexer stay F16; norms,
  router and mHC stay F32.
- Quantization is GPTQ-style (Hessian-weighted error feedback, per expert), computed straight from the FP8/FP4 shards.

## Why

We wanted a frontier-class open model running **locally, on hardware people actually own**, without slicing it down to a
distilled toy. The design question was where the bits matter. A MoE spends most of its parameters on experts that
each token touches only rarely, so we spend the bit budget where the calibration says it counts, and run the rest
as cheaply as the quality numbers allow.

The engine then treats the experts as a three-tier cache:

| Tier | Holds | Fed by |
| --- | --- | --- |
| VRAM cache | the hottest ~1,300–1,600 experts | LRU + routing bias toward resident experts |
| Pinned RAM arena | ~6,700 experts (60 GiB) | DMA straight to the GPU |
| NVMe file tier | the other ~8,700 experts | O_DIRECT reads, prefetched a layer ahead |

## Quality

Perplexity on held-out data against the FP8 original ("teacher"):

| Domain | GSQ ppl | Teacher ppl |
| --- | ---: | ---: |
| All (head gate) | 5.645 | 4.915 |
| agent | 2.118 | 2.121 |
| reasoning | 6.19 | 5.83 |
| code | 1.98 | 1.74 |
| zh | 7.64 | 6.83 |

Top-1 token agreement: 0.674 (GSQ) vs 0.701 (teacher). In the engine, a 4096-token code prompt gives ppl 2.380 and top-1 0.822.

**Prompt processing is exact:** every prefill speed-up in the engine keeps the last-token logits byte-identical.
**Decode uses two selection-only routing biases by default:** a bonus for VRAM-resident experts and a penalty for
NVMe ones. They trade a little perplexity for speed (code 2.38 → 2.87). The mixing weights stay exact, every expert
stays reachable, and `--route-bias 0 --route-bias-file 0` gives fully exact routing, which is much slower.

## Speed (RTX 4090 24 GB · Ryzen 7 5700X · 90 GB DDR4 · NVMe)

| | |
| --- | ---: |
| Prefill, 8K-token prompt | 221 tok/s one-shot · 250–269 tok/s in the server |
| Prefill, 16K / 20K prompt | 288 / 266 tok/s |
| Decode (256-token prompt) | 25.5–26.3 tok/s · 26.6–28.9 with the opt-in fused ggml op |
| Decode in a real agent session (20–32K context) | 22.7–28.2 tok/s |
| Model load | ~50 s |

How each number was reached, step by step:
[V41_PROGRESS.md](https://github.com/Indras-Mirror/Strata-DSV4.1/blob/main/docs/ds4/V41_PROGRESS.md).

## Files

| File | What |
| --- | --- |
| `V41-GSQ.gguf.part-NN-of-MM` | the 163 GB model, split into ~10 GiB byte ranges (Hub file-size limit) |
| `reassemble.sh` | checks every part, joins them into `V41-GSQ.gguf`, verifies the result |
| `SHA256SUMS` | hashes of every part and of the joined file |
| `V41-GSQ.gguf.experts.json` | per-expert offsets and formats |
| `rank.bin` | expert hotness ranking, used to seed the VRAM cache and RAM arena |
| `tokenizer/` | V4.1 tokenizer + chat template |
| `V41-MTP-Vision.gguf` | the **DSpark** speculative-draft stack (`mtp.0-2`: experts MXFP4, a lossless repack of the original FP4; the rest Q8_0) plus the **vision tower** (Q8_0). Engine support is in progress. |

### Reassemble

```bash
hf download IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored --local-dir V41-GSQ
bash V41-GSQ/reassemble.sh V41-GSQ                 # needs ~163 GB free on top of the parts
bash V41-GSQ/reassemble.sh V41-GSQ --delete-parts  # or: delete each part once appended (~163 GB total)
```

## Running it

You also need the **Engram tables** (202.8 GB). They aren't in this repo because they're unchanged FP8 rows from the
original checkpoint, so there is no point hosting a third copy. Download the two Engram shards
(`model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors`, 101.5 GB each) plus
`model.safetensors.index.json` and `config.json` from
[dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8).
The abliteration left Engram byte-identical to the base model, and
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) has the same two shards,
so either source works. Then extract them to fast drives (one table per drive is ideal):

```bash
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 model-00047-of-00048.safetensors \
   model-00048-of-00048.safetensors model.safetensors.index.json config.json --local-dir FP8-engram
```


```bash
git clone https://github.com/Indras-Mirror/Strata-DSV4.1 && cd Strata-DSV4.1
# build (CUDA, sm_89 shown) - see the repo README for the full line
cmake -S . -B build-ds4-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
      -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
ninja -C build-ds4-gpu ds41_generate
python3 tools/v41/extract_engram.py FP8-engram --to layers.1=/fast/a --to layers.14=/fast/b
STRATA_NO_LARGEPAGES=1 build-ds4-gpu/ds41_generate -m V41-GSQ.gguf --ids-file prompt.i32 -n 128 \
     --backend cuda --experts gpu --engram "/fast/a:/fast/b" --rank rank.bin
```

An OpenAI- and Anthropic-compatible server with tool calls (`tools/v41/serve/`) is in the repo too.

**Minimum hardware:** an NVIDIA GPU with 24 GB, ~90 GB RAM (the RAM arena is 60 GiB by default; `--arena-gib`
lowers it, at some speed cost), and fast NVMe with room for the model plus the Engram tables (~366 GB total).

## Credits

- **DeepSeek** for [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (MIT).
- **[dealignai](https://huggingface.co/dealignai)** for the abliterated FP8 checkpoint this is quantized from (MIT).
  This release inherits that abliteration: the model has **no refusal behaviour**, and you are responsible for how you
  use it.
- **[Strata](https://github.com/Niko1221/Strata)** (Niko1221), the tiered MoE engine this work builds on;
  **ggml / llama.cpp** for the compute backend.
- Quantization pipeline, engine port and measurements: **Indra's Mirror** ([Strata-DSV4.1](https://github.com/Indras-Mirror/Strata-DSV4.1)).
