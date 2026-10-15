# DeepSeek-V4.1-Flash GSQ on AMD (gfx1100, HIP) - how this checkout runs it

This page records how the V4.1 engine was brought up on **2x Radeon RX 7900 XTX (gfx1100, wave32) + Ryzen 9 9900X +
128 GiB RAM**, ROCm 7.14, and what the numbers are. The engine (`tools/v41/engine/`) was written for CUDA; on this
box the **whole three-tier expert path runs on HIP**: the dense half and the routed experts are on the GPU, the RAM
arena holds the ranked experts, and the file tier reads the rest from the GGUF.

## What runs where

| Half | Where | Notes |
| --- | --- | --- |
| Dense half (attention, router, shared expert, Engram add, head) | RX 7900 XTX via `ggml-hip` | `--backend cuda` picks the GPU by type, so the HIP build keeps the same flag |
| Routed experts (PTQ1_0 / Q2_0 / Q3_K, per-expert formats) | RX 7900 XTX, through the `DS4_MOE_CUDA` tier compiled as HIP | `--experts gpu`; VRAM cache + 60 GiB pinned RAM arena + file tier, the fork's grouped kernels from `strata_kernels` |
| CPU expert pool | unused with `--experts gpu` | `--experts cpu` still works (disk-bound, ~1.4 tok/s) |
| Engram tables | **not installed** | the engine warns and skips the add; the two 101.5 GB shards are not in the GSQ repo (its `NEED-TO-DOWNLOAD-ENGRAM` files name the source) |

## Build

The engine's own sources and `strata_kernels` compile in minutes. ggml's HIP backend is the expensive part (the
flash-attention template instances did not finish in >40 min at 20 jobs on this box), so the build reuses a ggml that
the same llama.cpp checkout already built:

```sh
# 1. a llama.cpp checkout with the DSV4 hyper-connection ops (upstream has them), built once for gfx1100
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 ...
cmake --build /path/to/llama.cpp/build -j

# 2. this engine, linking that ggml instead of recompiling it
cmake -S . -B build-hip-pre -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DSTRATA_GGML_DIR=/path/to/llama.cpp -DSTRATA_GGML_PREBUILT=/path/to/llama.cpp/build \
      -DBUILD_SHARED_LIBS=ON -DSTRATA_BUILD_TESTS=OFF
make -C build-hip-pre -j ds41_generate

# or, without STRATA_GGML_PREBUILT, the same configure compiles ggml from STRATA_GGML_DIR itself
```

`STRATA_GGML_PREBUILT` is a dev shortcut (empty by default): headers and archives come from the same source tree, so
nothing can drift, and the normal CUDA/CPU paths are untouched.

### The port changes in this tree

1. `CMakeLists.txt`: with `STRATA_ENABLE_HIP` the `BUILD_SHARED_LIBS OFF` / `GGML_STATIC ON` FORCEs are skipped
   (ggml-hip refuses a static link and wants its own shared build). CUDA and CPU builds are unchanged.
2. `CMakeLists.txt` + `tools/v41/cmake/ggml_prebuilt.cmake` (+ `ggml_prebuilt_stubs.cpp`): the prebuilt-ggml option
   above. The stub answers the one backend registration the prebuilt `libggml.a` references but this engine does not
   link (Vulkan on this box); ggml's registry ignores a NULL registration, so Vulkan stays out of the run.
3. `tools/v41/engine/v41_engine.cmake`: `ds41_dense` links `ggml-hip` and keeps the `DS41_HAVE_CUDA` compile path
   (ggml's HIP backend is the CUDA backend's sources compiled as HIP, so it exports `ggml_backend_cuda_*`); the expert
   tier gets a `ds41_moe_hip` flavour - the same `DS4_MOE_CUDA` source, `strata_hip_runtime` force-including the
   compat header - and `ds41_generate` prefers it (HIP tier, else CUDA tier, else CPU tier).
4. `tools/v41/engine/ds41_moe.cpp`: `set_cache_enabled()` touched `im_->gpu->cache`, a member of the CUDA-only `Gpu`
   struct, so the CPU-only flavour did not compile at all; the access is now behind `#if defined(DS4_MOE_CUDA)`.
5. `include/strata/hip_compat/cuda_profiler_api.h`: the one CUDA header the tier includes that ROCm has no
   counterpart for (`cudaProfilerStart/Stop` -> `hipProfilerStart/Stop`).
6. `tools/v41/engine/v41_engine.cmake`: the DSpark gate `test_ds41_dspark` takes `DS41_HAVE_CUDA` when `ggml-hip`
   exists too (the gate asks for a GPU backend by type and refuses without one; `ggml-hip` exports the same
   `ggml_backend_cuda_init`), so the draft stack can be scored on the ROCm card.

The tier itself needed no source change: it carries **no device code** (the kernels live in `strata_kernels`, which
the HIP backend already compiles for gfx1100), and every CUDA runtime call it makes was already in the compat shim.
The MMQ prompt path (`strata_mmq`, `ds41_transcode`) stays CUDA-only; without `DS4_MOE_MMQ` the tier runs the prompt
through its grouped kernels, as a CUDA build without that option does.

Local, not upstream: `local/run-one-shot.sh` (one-shot), `local/ds41-engine-local.sh` + `local/ds41-local.json` (the API
server), `local/encode_prompt.py` / `local/detok.py` (tokenize a prompt into `--ids-file` and back).

## One-shot run

```sh
./local/run-one-shot.sh --ids-file local/prompts/code_review.i32 -n 96     # experts gpu by default, EXPERTS=cpu to fall back
```

Two env settings matter on this machine:

- `STRATA_NO_LARGEPAGES=1` - pin the RAM arena with 4 KiB pages (~17 s for 60 GiB); with THP it takes minutes.
- `--experts cpu` additionally needs `STRATA_ROUTED_HOST=1` (the CPU tier has no device input for the layer-major
  prompt batch: `run_chunk_dev` returns false, "tier run refused") and reads blobs better through the page cache
  (`STRATA_BLOB_MMAP=1` or `STRATA_BLOB_DIRECT=0`). Neither applies to the GPU tier.

## Measured (2026-10-15, this box, the 151.8 GiB joined GSQ file, no Engram)

Load: **17.2 s** (60 GiB arena pinned and seeded: 6708 experts in RAM, 8652 file tier, 16.7 s of it the seed).

| Run | Prompt | Decode |
| --- | ---: | ---: |
| Code-review prompt (113 tok) + 96 tok, one-shot, VRAM cache 613 experts | **18.7 tok/s** | **8.71 tok/s** (93.6% VRAM hits) |
| The same prompt through the API server (`--serve`, ctx 32768) | 16.2 tok/s | **12.9 tok/s** |
| 17-token chat prompt, 8 GiB arena (first test of the tier) | 6.2 tok/s | 10.69 tok/s (95.0% VRAM hits) |
| `--experts cpu` (page-cached blobs), for comparison | 0.9-3.2 tok/s | 1.3-2.5 tok/s |

The decode tokens are **identical** between the CPU tier and the HIP tier on the same prompt, so the port did not
change what the model computes. Quality check on the code-review prompt (greedy, thinking off): the model returns the
missing error handling and the dropped line separators, in order - a correct review of the function it was given.

Where the time goes now (one-shot, 96 tokens): 21336 expert hits in VRAM, 1186 PCIe misses, 352 file-tier reads;
`prefetch 5024 issued / 1812 used`. The expert cache is what sets decode: 5 GiB of free VRAM after the dense half at
ctx 32768 buys only 613 experts, so a smaller `--ctx` or a bigger margin is the lever.

## DSpark on this box (the draft stack runs on HIP)

The Q2_K sidecar loads as a HIP graph (`2.91 GB of weights on the card` with the 64-experts-per-layer prune, 5.14 GB
without) and the one-shot `STRATA_DSPARK=` path works. Same prompt, same flags, `-n 96`, one-shot:

| | decode | tokens per verify pass |
| --- | ---: | ---: |
| no DSpark | 9.12 tok/s | 1 |
| DSpark depth 4 | 16.50 tok/s | 2.474 |
| DSpark depth 4 + `STRATA_DSPARK_KEEP=…:64` | 22.70 tok/s (‑n 32) | 2.818 |
| DSpark depth 2 | 11.96 tok/s | 2.286 |
| through `--serve` (depth 4 + prune, one 96-token reply) | **17.5 tok/s** | 2.74 |

The draft stack itself is healthy on HIP: the repo's own gate, `test_ds41_dspark` (now built for the HIP backend too:
`DS41_HAVE_CUDA` + `ggml-hip`), scores **2.423 accepted drafts/pass = 3.423 tokens/pass, first draft 92.3%** over 26
steps of a recorded trunk stream - in the same range as the CUDA number in `V41_PROGRESS.md` (3.27).

**The confidence trim is what is not safe here.** `STRATA_DSPARK_CONF=0.6` narrows the verify width to the leading
drafts the confidence head is sure about, which - by the engine's own comment - only changes the verify width and is
lossless in principle. On this model it is not: at depth 4 with the prune it produced a *different and wrong* answer
on the code-review prompt (a `std::string` repetition loop), deterministically, while depth 4 without the trim, depth 2
and the trim without the prune all answered correctly. The greedy stream moves with the verify width at near-ties
(the engine's `--ctx`/DSpark notes say as much: "DSpark output = plain greedy output holds up to ties only"), and a
narrowed width lands on a worse stream. So this checkout runs DSpark **without** `STRATA_DSPARK_CONF` (see
`local/ds41-local.json`), and the trim stays a knob to re-measure, not a default.

## Not done yet

- **The Engram tables** (202 GB): download `model-0004{7,8}-of-00048.safetensors` from
  `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8` (or `deepseek-ai/DeepSeek-V4.1-Flash`), extract with
  `tools/v41/extract_engram.py --to layers.1=DIR_A --to layers.14=DIR_B`, and pass `--engram DIR_A:DIR_B`. Until then
  the engine is not the real model.
- **The MMQ prompt path** (`--chunk-mmq`, PTQ1_0 transcode in `ds41_transcode.cu`): CUDA-only; it is what makes big
  prompts faster on NVIDIA, and porting it means porting two small kernels plus `strata_mmq`.
- **The second 7900 XTX** is idle: the v41 engine has no peer-device/multi-GPU support (that is the Qwen engine's).
- The upstream `ds41-engine.sh` (memguard's `systemd-run` RAM cap + the shared GPU lock) does not work in a session
  without a user D-Bus; `local/ds41-engine-local.sh` is the plain wrapper used here.

## API server

```sh
STRATA_HOME=$PWD /path/to/venv/bin/python3 tools/v41/serve/serve_ds41.py \
    --engine strata --config local/ds41-local.json --port 8150
# OpenAI /v1/chat/completions, Anthropic /v1/messages, web UI on http://127.0.0.1:8150/
```

`STRATA_HOME` is a new env hook in `serve_ds41.py` (default `~/AI/Strata`, unchanged): this checkout ships its own
`serve/`, so the shim needs no installed Strata. Two signature drifts against the newer `serve/` were fixed in the
shim: `Service.run(..., force=...)` and `OutputParser.finish(reason)`.
