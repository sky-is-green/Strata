<h1 align="center">Strata-DSV4.1 - AMD (gfx1100) branch</h1>

<p align="center"><b>DeepSeek-V4.1-Flash (40 layers x 384 experts) on ONE RX 7900 XTX - 152 GiB of experts across VRAM, pinned RAM and NVMe, plus 189 GiB of Engram on disk</b><br>
The AMD port and tuning of <a href="https://github.com/Indras-Mirror/Strata-DSV4.1">Strata-DSV4.1</a>, itself built on <a href="https://github.com/Niko1221/Strata">Strata</a> - Linux - ROCm/HIP - MIT</p>

<p align="center"><b>RX 7900 XTX (gfx1100, ROCm 7.14) + 128 GiB RAM:&nbsp; prefill 240-363 tok/s &nbsp;·&nbsp; DSpark decode 24.3 tok/s &nbsp;·&nbsp; 256K context, 8-bit KV</b></p>

## What this branch is

This is the **AMD (gfx1100)** branch of the fork `vincalkr/Strata-DSV4.1`: the same engine and serve as upstream, ported
and tuned to run DeepSeek-V4.1-Flash on **one** RX 7900 XTX.  Every number on this page was measured on that machine;
the engine runs on a single card (`"gpu": 0` in [local/ds41-local.json](local/ds41-local.json)).

The platform is not ours.  The engine, the serve and the earlier DeepSeek-V4-Flash (0731) path come from
[Indras-Mirror/Strata-DSV4.1](https://github.com/Indras-Mirror/Strata-DSV4.1), which builds on
[Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT; `LICENSE` is kept as it is).  Their README - the NVIDIA
measurements, the 512K-context settings, the API and the V4-Flash engine - is kept verbatim in
**[docs/upstream/README.md](docs/upstream/README.md)**, with the localized copies next to it; read it for anything that
is not AMD-specific.  Our contribution is the AMD path: the port, the tuning, the measurements and these notes.

## The machine, and what is on disk

Two RX 7900 XTX are installed, but **one** is used.  128 GiB of RAM, one 2 TB NVMe, ROCm 7.14.  On disk:
`models/V41-GSQ/V41-GSQ.gguf` (152 GiB), the 5.3 GiB DSpark draft with its small files, and the two Engram tables in
`models/engram/` (189 GiB, layers 1 and 14).  The second card was tried as an expert tier and dropped: behind its
PCIe 4.0 **x2** bridge it is worth +10-20% as extra cache, not 2x (see the negatives below).

| measured here, one sequence | |
| --- | ---: |
| decode, 4K state, DSpark off | 13.7 tok/s |
| decode, 64K state, DSpark off | 14.0 tok/s |
| decode through the server, DSpark depth 4, ggml graphs off | 22.8 -> 24.3 tok/s |
| prompt, 5787 tokens | 240-245 tok/s |
| prompt, 63,617 tokens, fresh, through the server | 363 tok/s |

## The optimizations, each with what it was worth

| change | measured effect |
| --- | --- |
| ggml's fused dense ops (`local/build-ggml-fused.sh`: `dsv41-fake-quant.patch` + `dsv41-fuse.patch`) | +13% decode, +23% with DSpark; last-row logits top-1 identical, mean absolute difference 0.156 |
| `--arena-gib 75` (from 60) | +40% decode (16.4-17.4 -> 23.8-25.2 tok/s), prompt file wait 16.6 -> 11.0 s; 80 and above are worse or unstable |
| `GGML_CUDA_DISABLE_GRAPHS=1` | +6.6% decode, +3% prompt in the server |
| `STRATA_PRESTAGE_GIB=1.5` | +5.8% prompt (231.2 -> 244.5 tok/s) |
| `--kv-q8` (8-bit KV, not 4-bit) | lossless here: last-row cosine 0.998987, top-5 5/5, +0.37 GiB |
| DSpark depth 4 with `KEEP 64` | depth 5, depth 6, `KEEP 32` and `KEEP 128` all measured worse |
| `STRATA_SERVE_PASS_BIAS=0` | the upstream default (0.5/0.3) gives 2.2x the expert misses here: 12.1 against 25.1 tok/s |
| the v41 serve's thinking budget (`reasoning_budget_tokens 8192`, `reasoning_close_retry`) | a runaway reasoning loop is now closed at the budget and the model answers (tested with a 128-token budget) |
| the effort mapping aligned with the reference implementation | `low`/`medium`/`xhigh` resolve the way the reference server resolves them for V4.1 (75/75/75) |
| DSML tool-call parsing in the local path | tool calls reach the client, streaming and not |
| prompt batching | a fresh 63K prompt at `--prompt-batch 12288` can run out of memory and segfault the one-shot CLI; the server sizes the batch itself and does 363 tok/s |

## Where the time goes, and what is left

A runtime trace at 17K state: the GPU kernels are 32.2 ms of a 139 ms profiled token, and 215 `hipStreamSynchronize`
calls a token account for 105 ms - 95.3 of them after a device-to-host readback and 76.7 after the dense graph's
launch - with 4,238 kernel launches a token.  Decode is therefore **host/sync-bound, not GPU-bound**, and the
synchronizations are mostly real work being waited for (the experts' DMA and the launch latency), not idle overhead.
The lever that is left is the number of kernel launches.  The per-layer host handshake itself was investigated to the
end and closed: see [docs/ds4/V41_PIPELINE_OVERLAP.md](docs/ds4/V41_PIPELINE_OVERLAP.md).

## Tried and reverted

With the numbers, so they are not repeated: `--pf-b` below 1.0 (-19% decode; the default 1.43 sits just above that
cliff), DSpark depth 5 and 6, `KEEP 32` and `128`, `STRATA_DSPARK_ADAPT`, the adaptive arena, `STRATA_PROM=1`, the
peer expert tier on the second card (correct and bit-identical rows, 91.5 rows a token moved off card 0, but 11.8
against 12.6 tok/s), the transcode decode, `STRATA_HIP_WMMA` (the +5.4% was noise: v41 never calls that kernel), a
cache for `ggml_cuda_get_device` (120 calls a token, 0.06 ms) and the 685 per-token error queries (below 0.3 ms).

## Getting the model files

The model is not in git (355 GiB in total).  [local/fetch-model.sh](local/fetch-model.sh) builds exactly the layout
this branch uses:

1. the 16 parts of `V41-GSQ.gguf` plus `SHA256SUMS`, `rank.bin`, `V41-GSQ.gguf.experts.json`, the DSpark draft and the
   tokenizer, from [IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored)
   (resumable), joined and verified by the repo's own `reassemble.sh` - it checks every part's sha256 before appending
   and drops each part as it goes (`DELETE_PARTS=0` keeps them), so what stays is the joined `V41-GSQ.gguf`, exactly
   the tree the numbers above were measured on;
2. the two Engram tables, which are **not** in that repo: they are shards 47 (layer 1) and 48 (layer 14) of
   [dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8),
   fetched by `tools/v41/fetch_v41.sh` and extracted with `tools/v41/extract_engram.py` into `models/engram/layers1`
   and `models/engram/layers14`, with the merged index from `local/engram_index.py` (the engine needs it because the
   two tables live in different directories).

Space: 163 GB of parts for the GGUF (the join needs the same again only with `DELETE_PARTS=0`), and 101 GB per Engram table plus the 101 GB shard each comes from.  With two
disks, set `DEST_LAYERS1` and `DEST_LAYERS14`; on one disk use `ONE_AT_A_TIME=1`, which extracts and deletes one
shard at a time (peak about 163 + 101 + 101 GB).

```sh
local/fetch-model.sh /path/to/models     # DEST defaults to <repo>/models
```

It is resumable and it checks: every expected size comes from the Hugging Face API, a file counts as complete only
when its size matches (so a partial download is resumed with `curl -C -`, never trusted), the parts are joined by the
repo's own `reassemble.sh` (which verifies their sha256 first and the joined file at the end), and an interrupted join
is reported instead of being joined over.  Every parameter:

| variable | default | what it does |
| --- | --- | --- |
| first argument | `<repo>/models` | where the trees go |
| `GSQ_REPO`, `FP8_REPO` | the two repositories linked above | where to fetch from |
| `DRY_RUN=1` | 0 | print the whole plan (what is complete, what would be fetched, the join, the Engram steps) and change nothing |
| `DELETE_PARTS` | 1 | drop each GGUF part as it is appended; `0` keeps them and needs the space twice |
| `ONE_AT_A_TIME` | 0 | Engram one shard at a time: fetch 47, extract layer 1, delete 47, fetch 48, extract layer 14, delete 48 (peak about 163 + 101 + 101 GB on a single disk) |
| `DEST_LAYERS1`, `DEST_LAYERS14` | `<models>/engram/layers1`, `layers14` | put the two Engram tables on different disks |
| `KEEP_FP8` | 0 | keep the FP8 shards after the extraction (101 GB each) |
| `FORCE_JOIN` | 0 | remove an interrupted `V41-GSQ.gguf` and join again (by default it stops and says so) |
| `VERIFY_ROWS` | 200 | how many random Engram rows are checked against the source |
| `WORK` | `DEST` | scratch for the FP8 shards and the size lists |
| `HF_TOKEN` | `~/.cache/huggingface/token` | needed only for gated repositories |

## Documents

- [docs/ds4/AMD_HIP_NOTES.md](docs/ds4/AMD_HIP_NOTES.md) - every measurement and every setting on this machine.
- [docs/ds4/V41_PIPELINE_OVERLAP.md](docs/ds4/V41_PIPELINE_OVERLAP.md) - the per-layer host handshake, measured and closed.
- [docs/ds4/V41_AMD_PEER_TIER.md](docs/ds4/V41_AMD_PEER_TIER.md) - the second-card design that was measured and dropped.
- [docs/upstream/README.md](docs/upstream/README.md) - the upstream README (NVIDIA, 512K, V4-Flash), verbatim.
- [local/ds41-local.json](local/ds41-local.json) - the configuration these numbers were measured with.
