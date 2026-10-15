# DeepSeek-V4.1-Flash GSQ on AMD (gfx1100, HIP) - how this checkout runs it

This page records how the V4.1 engine was brought up on **2x Radeon RX 7900 XTX (gfx1100, wave32) + Ryzen 9 9900X +
128 GiB RAM**, ROCm 7.14, NVMe (model on `/dev/nvme0n1p4`, one Engram table per drive), and what the numbers are.
The engine (`tools/v41/engine/`) was written for CUDA; here **the whole three-tier expert path runs on HIP**: the
dense half and the routed experts are on the GPU, the RAM arena holds the ranked experts, the file tier reads the
rest, and the Engram tables are read from two files.

## What runs where

| Half | Where | Notes |
| --- | --- | --- |
| Dense half (attention, router, shared expert, Engram add, head) | RX 7900 XTX via `ggml-hip` | `--backend cuda` picks the GPU by type, so the HIP build keeps the same flag |
| Routed experts (PTQ1_0 / Q2_0 / Q3_K, per-expert formats) | RX 7900 XTX, through the `DS4_MOE_CUDA` tier compiled as HIP | `--experts gpu`: VRAM cache (~2100 experts) + 60 GiB pinned RAM arena (6708) + file tier (8652) |
| Engram tables (202.8 GB) | two files, one per NVMe, read at run time | `--engram A:B`; the raw-device fast path needs `/dev/nvme*n1` access this user does not have, so it uses filesystem preads (see Engram below) |
| CPU expert pool | unused with `--experts gpu` | `--experts cpu` still works (disk-bound, ~1.4 tok/s) |

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
      -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
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
   compat header - and `ds41_generate` prefers it (HIP tier, else CUDA tier, else CPU tier). With `strata_mmq` present
   the tier also takes the MMQ prompt path (`DS4_MOE_MMQ`), and the DSpark gate takes `DS41_HAVE_CUDA`.
4. `tools/v41/engine/ds41_moe.cpp`: `set_cache_enabled()` touched `im_->gpu->cache`, a member of the CUDA-only `Gpu`
   struct, so the CPU-only flavour did not compile at all; the access is now behind `#if defined(DS4_MOE_CUDA)`.
5. `include/strata/hip_compat/cuda_profiler_api.h`: the one CUDA header the tier includes that ROCm has no
   counterpart for (`cudaProfilerStart/Stop` -> `hipProfilerStart/Stop`).
6. `src/prefill/moe_mmq.cu` + `CMakeLists.txt` (`strata_mmq_api_version()`): the MMQ host API moved after the pinned
   commit (`ggml_cuda_mmq_get_config`'s `prec_src1`, `mul_mat_q_case<type, prec_src1>`); the file carries both
   spellings behind `STRATA_MMQ_PREC_SRC1`, detected from the ggml tree in use.
7. `src/prefill/ggml_cuda_host.cu`: its host symbols are now weak. The V4.1 engine links `ggml-hip`, which defines the
   same `ggml_cuda_error` / `ggml_cuda_info` / `ggml_backend_cuda_context::*`, and the link failed with duplicate
   symbols; weak lets the real backend's copy win while a build without it (the HIP MMQ parity tests) still links.

The tier itself needed no source change: it carries **no device code** (the kernels live in `strata_kernels`, which the
HIP backend compiles for gfx1100), and every CUDA runtime call it makes was already in the compat shim.

Local, not upstream: `local/run-one-shot.sh` (one-shot), `local/ds41-engine-local.sh` + `local/ds41-local.json` (the
API server), `local/encode_prompt.py` / `local/detok.py` (tokenize a prompt into `--ids-file` and back),
`local/engram_index.py` (merge the per-layer Engram indexes).

## One-shot run

```sh
./local/run-one-shot.sh --ids-file local/prompts/code_review.i32 -n 96   # experts gpu by default, EXPERTS=cpu to fall back
```

Two env settings matter on this machine:

- `STRATA_NO_LARGEPAGES=1` - pin the RAM arena with 4 KiB pages (~15 s for 60 GiB); with THP it takes minutes.
- `--experts cpu` additionally needs `STRATA_ROUTED_HOST=1` (the CPU tier has no device input for the layer-major
  prompt batch: `run_chunk_dev` returns false, "tier run refused") and reads blobs better through the page cache
  (`STRATA_BLOB_MMAP=1` or `STRATA_BLOB_DIRECT=0`). Neither applies to the GPU tier.

## The Engram tables (202.8 GB)

DeepSeek-V4.1 reads an n-gram memory at run time and the GSQ repo does not host it: its two
`model-0004{7,8}-of-00048.safetensors.NEED-TO-DOWNLOAD-ENGRAM.txt` markers name the source. What was done here, and
what it needs:

```sh
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 model.safetensors.index.json config.json \
    model-00047-of-00048.safetensors --local-dir SRCDIR     # ~101.5 GB, holds layers.1.engram.*
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 model-00048-of-00048.safetensors --local-dir SRCDIR
# layers.1 -> shard 47, layers.14 -> shard 48 (the index says which)
python3 tools/v41/extract_engram.py SRCDIR --to layers.1=/home/vincenzo/v41-engram/layers1 \
                                         --to layers.14=/run/media/vincenzo/Dati/v41-engram/layers14
python3 local/engram_index.py  <both dirs>      # one index naming all four tensors; the engine reads the first it finds
```

Notes from doing it here: the extraction needs ~101 GB of free space per layer and checks every tensor's size plus
2000 random rows; the source shards can be deleted right after their layer is extracted. `extract_engram.py` insists
on a `--to` for every engram layer it finds in the index, so one shard at a time is easiest with a two-tensor
index.json in a scratch directory (that is what was done; `local/engram-src*`).

At run time `Ds41Engram::open()` reads the first `model.safetensors.index.json` among the `--engram` directories and
resolves each tensor's filename against all of them, which is why the merged index (distinct filenames per layer) is
what makes the two-drive split work. The engine prefers raw block-device reads when a file has ≥1024 extents and the
device node is readable; `/dev/nvme0n1p1` is `root:disk` here, so it prints
`1973 extents, /dev/nvme0n1p1 not readable - filesystem reads` and uses preads - measured cost ~1 ms/token, which is
why it is left alone.

## Measured (2026-10-15, this box)

Load with the full configuration (256K context, Engram, DSpark): **16.5 s** (arena: 6708 experts in RAM, 8652 file
tier, seeded in 15.8 s).

| Run (server, 256K ctx, Engram, DSpark) | Prompt | Decode |
| --- | ---: | ---: |
| Code-review prompt, 113 tokens in / 96 out | 6.6 tok/s (cold first request) | **18.0-21.5 tok/s** |
| 1536-token prompt / 16 out | **142 tok/s** | 23.4 tok/s |
| 3072-token prompt (one-shot, quiet box, no DSpark) | 165-172 tok/s | - |

Decode A/B with everything else equal (one-shot, 96 tokens, Engram on): **10.35 / 12.68 tok/s without DSpark,
20.98 / 20.55 tok/s with it** - DSpark is worth ~1.7x here.

Where the time goes (one-shot, 96 tokens, DSpark, `pass_v` split over 41 verify passes of 7.05 tokens):
**dense 99.9 ms, tier 365.1 ms**, taps 0.4 ms, logits 1.1 ms per pass. The expert tier - VRAM-hit expert kernels plus
the PCIe/file misses - is ~78% of the decode; 93-97% of the expert reads are VRAM hits at ~2100 slots (14 GiB), with
~29 PCIe misses and ~4.6 file reads per token.

Quality: greedy, thinking off, the code-review prompt gets the missing error handling (and the dropped line
separators / the unbounded `s +=` growth) in order, with the Engram tables enabled the answer points at the
`std::ifstream` open check first. The CPU tier and the HIP tier produce **identical tokens** on the same prompt.

## DSpark on this box (the draft stack runs on HIP)

The Q2_K sidecar loads as a HIP graph (`2.91 GB of weights on the card` with the 64-experts-per-layer prune, 5.14 GB
without) and the one-shot `STRATA_DSPARK=` path works. The draft stack itself is healthy on HIP: the repo's own gate,
`test_ds41_dspark` (now built for the HIP backend too), scores **2.423 accepted drafts/pass = 3.423 tokens/pass,
first draft 92.3%** over 26 steps of a recorded trunk stream - the same range as the CUDA number in
`V41_PROGRESS.md` (3.27). In the server with Engram on, DSpark runs at **3.17-3.20 tokens/pass**.

**The confidence trim is what is not safe here.** `STRATA_DSPARK_CONF=0.6` narrows the verify width to the leading
drafts the confidence head is sure about, which - by the engine's own comment - only changes the verify width and is
lossless in principle. On this model it is not: at depth 4 with the prune it produced a *different and wrong* answer
on the code-review prompt (a `std::string` repetition loop), deterministically, while depth 4 without the trim, depth
2 and the trim without the prune all answered correctly. The greedy stream moves with the verify width at near-ties
(the engine's own notes say as much: "DSpark output = plain greedy output holds up to ties only"), and a narrowed
width landed on a worse stream. So this checkout runs DSpark **without** `STRATA_DSPARK_CONF` (see
`local/ds41-local.json`); the trim stays a knob to re-measure, not a default.

## The MMQ prompt path on HIP

`-DSTRATA_PREFILL_MMQ=ON` builds `strata_mmq` for HIP and the tier takes the CUDA path (`ds41_transcode.cu` compiled
as HIP, `DS4_MOE_MMQ=1`); changes 6 and 7 above are what it took. Measured on a 3072-token prompt, quiet box,
alternating runs (three each): **`--chunk-mmq` 172.4 / 168.8 / 155.9 tok/s (median 168.8) vs `--no-chunk-mmq`
165.1 / 164.6 / 161.0 (median 164.6)** - +2.5%, inside the spread. The PTQ1_0 -> Q2_0 transcode plus the MoE-MMQ
kernels are a wash on RDNA3; they are left on (the engine's default) but they are not where the prompt time is.

## The second GPU

The project's **parent engine** (the Qwen one in this same tree) does support several cards: contiguous layer ranges,
one per GPU, each with the expert cache for its own layers (`docs/MULTI_GPU.md`, `src/core/peer_experts.hpp`,
`--gpus` / layer split / `--peer-device` in the server). The **V4.1 engine does not**: `ds41_generate` has no device
flag, `Ds41MoeTier`'s `Gpu` state is one device (one cache, one arena, one stream set) and `ds41_dense` builds one
backend. On this box the second 7900 XTX is therefore idle.

What it would buy, measured rather than guessed: the expert tier is 78% of decode, but 93-97% of its reads already hit
VRAM. Shrinking the cache shows the slope: `--slots 1200` 19.30 tok/s (92.5% hits), `--slots 400` 14.98 tok/s (80.0%
hits) - so a second card's extra ~14 GiB (2x the slots) is worth roughly +10-20%, not 2x. (`--slots 2100` measured
5.34 tok/s with 7.3% hits: an explicit budget larger than the free VRAM thrashes; leave `--slots auto`.) The bigger
win would be *splitting the work* - a peer expert tier, where the second card computes its own resident experts in
parallel with the first card's dense half, or a layer split as the parent engine does. Both are new code in
`ds41_moe.cpp` / `ds41_dense.cpp` (the parent's peer-expert machinery is for a different engine and cannot be reused
as is); a realistic estimate is days of work with real correctness risk, and the ceiling is ~2x.

## Not done yet

- **`--kv-fp4`** (the bit-exact 4-bit KV cache that makes the 4090's 512K context cheap) needs the
  `tools/v41/ggml/kv-fp4.patch` packers in the ggml tree (`cpy-utils.cuh`); the tree here does not carry them, so the
  server runs the default BF16-with-activation-quant KV. At 256K that is affordable, at 512K it is the first thing to
  port.
- **The second card**, as above.
- **`--chunk-mmq` tuning**: the Q2_0 MoE-MMQ kernels have no RDNA3-specific config here; the A/B is a wash.

## API server (256K context)

```sh
STRATA_HOME=$PWD /path/to/venv/bin/python3 tools/v41/serve/serve_ds41.py \
    --engine strata --config local/ds41-local.json --port 8150
# OpenAI /v1/chat/completions, Anthropic /v1/messages, web UI on http://127.0.0.1:8150/
```

`local/ds41-local.json` is this machine's configuration: `--experts gpu`, `--arena-gib 60`, `--ctx 262144` (256K),
`--engram` to the two table directories, `--prompt-batch 2048`, DSpark depth 4 with the 64-expert prune and no
confidence trim, plus `STRATA_NO_LARGEPAGES=1`. `STRATA_HOME` is a new env hook in `serve_ds41.py` (default
`~/AI/Strata`, unchanged): this checkout ships its own `serve/`, so the shim needs no installed Strata. Two signature
drifts against the newer `serve/` were fixed in the shim: `Service.run(..., force=...)` and
`OutputParser.finish(reason)`.
