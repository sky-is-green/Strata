# The two-card split for the V4.1 engine (AMD)

**What it is.** The layers are split across two cards: the layers `[0, K)` on the
first, `[K, 40)` on the second. Each card then holds **only its own layers'
weights** and **only its own layers' experts** in its VRAM cache, so the two
together hold about twice the experts one card holds. The token crosses from
one card to the next **once per pass** - the residual and the pre-mix the
first stage's last `finish()` leaves behind - and not once per layer.

**Why.** On the reference box (2 x RX 7900 XTX, 128 GiB RAM) the expert tier
is **78% of a decode step** (`dense 99.9 ms, tier 365.1 ms` per pass), and
93-97% of the expert reads are VRAM hits at ~2100 slots (14 GiB). A 20 GB card
holds fewer slots than a 24 GB one, so the file tier does more of the work. The
split is the same lever the qwen4exp path already pulls: `docs/MULTI_GPU.md`
measures **41.4 -> ~70-71 tok/s** on 2 x RX 7900 XT 20 GB with 30 GB RAM -
this box - for the Scion-FlashNext 176B, with `--layer-split auto` and
`--expert-cache auto` giving 12007 + 10690 resident slots, **92% of the routed
pairs**, across the two cards.

**Not the peer tier.** `local/peer-tier-attempt.patch` gave a second card a
disjoint slice of the ranked experts and had it compute the rows the first card
missed (the same kernels, the same q8_1 activations, so the rows are
bit-identical). It measured **12.6 -> 11.8 tok/s, a 5% loss**: the peer
removed 35% of the routed rows, and the tier was only 23% of the step - so
that is ~3% of the step - against a per-layer hand-off (activation upload,
metadata, an event and a wait, per-row copies) on ~40 layers a token. A peer
*serves misses*; a split *owns layers*.

## The boundary is forced, and it is clean

A layer reads the compressed KV and the top-k of the **latest source at or
before it** (`Ds41Geometry::kv_src` / `idx_src`), so a source and every layer
that reads it have to be on one card. For V4.1:

- ratio 2 on layers 2..19, sources at 2 / 8 / 14 - groups {2..7}, {8..13}, {14..19}
- ratio 1 on layers 20..39, kv source 20 (read by every later layer), index
  sources 20 / 24 / 28 / 32 / 36 - groups {20..23}, {24..27}, {28..31}, {32..35}, {36..39}

The ratio-1 kv source at layer 20 is read by every later layer, so **the only
boundary with no cross-card state is 20**: card 0 takes layers 0-19, card 1
takes layers 20-39 (with the head and the DSpark drafter, which belong on the
last card). The candidate blocks are picked at layer 20 and read by the later
indexers - all on card 1.

The dense half refuses a boundary that would split a group, and says which
source it was.

## What is in the tree

- `ds41_dense.{hpp,cpp}`: `Ds41DenseConfig::layer_lo / layer_hi / own_head /
  own_embed`. A stage loads only its own layers' weights and keeps only its own
  layers' KV / indexer state; `write_state()` imports the hand-off (the
  residual and the pre-mix), `read_state()` exports it.
- `ds41_moe.{hpp,cpp}`: `Ds41MoeConfig::device` - the tier's device. Every
  device call in the tier runs with it current (`TierDev`), so a second tier
  lives on the second card.
- `ds41_generate.cpp`: `--layer-split K` and `--split-device D`. The two
  stages, their own tiers (each seeded with the profile's *own layers'*
  experts, so the cards never hold the same expert twice), the hand-off at the
  boundary, and the head on the last stage.

**Off by default.** One card, every layer, the path byte for byte as before.

## What is NOT wired yet

- `--serve` and the layer-major prompt batch (`--prompt-batch` > 1) refuse a
  split: the prompt's KV / indexer state has to land on the layer's own card,
  and the layer-major walk is not split yet. A one-shot run with
  `--prompt-batch 1` (the chunk path) is.
- The stages overlap across verify windows (the qwen4exp path's
  `--pipeline-windows`), which is where the larger part of that path's gain
  comes from. Not here yet: the split alone is the capacity half.

## Where the model lives (this box)

The device decides the speed, not the engine. The V4.1 model's reads are
random and small: the PLE table is 320,001,536 rows of 90 bytes, read 48 rows
a token, and the file tier reads a 1.38 MB expert blob per miss. On the
FUSE-mounted rotating disk (`/dev/sda2`, ntfs-3g) a random read is **7.5 ms
p50 / 10.5 ms p90**; on an SSD it is **0.19 ms**. The same effect we measured
on the Qwen3.8 model: with the model on the slow volume the decode ran at
3.39 tok/s, on the NVMe at 20.52 - the *storage*, not the engine.

**The USB enclosure is not an SSD.** The device the user mounted for this
(`/dev/sdb1`, XFS, "a new NVMe") turned out to be an NVMe inside a USB 3.x
enclosure (`uas` driver, `xhci-pci`): **45 MB/s** sequential, and random
reads are worse than that. A model whose file tier reads 1.38 MB expert
blobs at random cannot live there - the file tier alone would cost more than
the whole decode budget. So the model goes on the native NVMe (btrfs).

| what | where | why |
|---|---|---|
| `V41-GSQ.gguf` (163 GB) + the small files | the Crucial P310 (`/dev/nvme1n1p5`, btrfs, PCIe 4.0 x4); 93% full | the file tier: 1.38-7.75 MB blobs, random. Measured 10.99 tok/s here against 8.72 on the PCIe 3.0 Micron |
| the Engram table for layer 1 (101 GB) | the same P310 | shares the model's drive now; the Engram's reads are 256-byte rows issued in parallel, so the contention is small - not yet measured separately |
| the Engram table for layer 14 (101 GB) | the Crucial MX100 SATA SSD (`/dev/sda1`, XFS, 234 GB) | same; the reader issues its preads from an I/O pool, so one latency not 48 |
| the FP8 shards (101 GB each, transient) | the rotating disk (`/dev/sdb2`, ntfs-3g) | read once, sequentially, to extract the tables |
| the backup copy of the model | the USB enclosure (Samsung PM991a, `/dev/sdc1`, XFS) | written once, read only on a restore |
| a stale duplicate of the model | the Micron 2450 (`/dev/nvme2n1p1`, XFS, 256 GB) | left from the move; delete once the P310 copy has proven itself |

The P310 was at 79% before the move; the model needed 163 GB and only 148 GB were free. A 64 GB `work/swapfile`
sitting on it was the blocker. It was **not active** (`swapon --show` listed only zram) but still listed in
`/etc/fstab`, so it would have failed to activate at every boot. Deleting it brought the drive to 70%, and the model
took it to 93%. A 93%-full btrfs is fine for a read-only model and bad for the next big write: the next thing that
needs space on this drive has to free some first.

The device names move when drives are reseated: what matters is the volumes, not the `nvmeXn1` numbers.
On 2026-10-23 the case was re-arranged (the second GPU needed its slot back) and the Samsung that had held
the model went onto USB while the old USB volume came into the case as `nvme2`. The model's live copy is the
one on the native NVMe; the sha256 was re-checked there (`3897de64…`) before the symlink was pointed at it.

The two Engram tables and the GGUF are 365 GB over three volumes, each under
80% full. The GGUF moved off the work drive for two reasons: the drive was at
87% full (btrfs does not like it), and the space was needed for a table. The
SATA SSD is acceptable for a table even though it is the slow class of the
three, because the reader's I/O pool issues a token's ~48 preads in parallel
(`ds41_engram.cpp`, "a decode token's ~48 preads then ran serially, ~11 ms
cold" was the bug that pool fixed) - so the cost is one SATA latency, not 48.

**XFS on the new volumes, and the number behind it.** `ds41_engram.cpp:85`
records: a 101 GB table in 62,112 extents, written onto a 99%-full NVMe,
served **9K random 4 KiB reads/s** against **304K/s** for a 1,226-extent file
on the same partition. Keep these volumes under ~80% full and write each
table once, sequentially. The reader also takes the extent map once with
FIEMAP and preads the partition directly when a file has >= 1024 extents
(`STRATA_ENGRAM_RAWDEV`), which needs read access to `/dev/nvmeXn1` - on this
box the user is in `storage`, not `disk`, so that path is off until
`usermod -aG disk`.

## The box's memory, and the trap in it

31.7 GiB of RAM, ~8.5 GiB of it the desktop. The engine's host memory is the
expert arena (host RAM for the ranked experts) plus the page cache for the
163 GB model. Two facts decide whether a run is stable:

1. **`vm.swappiness` on this box was 150, not the 10 in `/etc/sysctl.d/60-swappiness.conf`.**
   `/usr/lib/udev/rules.d/30-zram.rules` sets `SYSCTL{vm.swappiness}="150"`
   the moment the zram device appears - after `systemd-sysctl` has applied
   the config - so the config never wins. And this box's swap is **zram**,
   which is RAM. With swappiness 150 the kernel "swaps" the arena and the
   model pages into zram, which *consumes* the memory it was supposed to
   free: zram's counters showed **1.9 TiB** of pages churned through it, and
   the global OOM killer took the engine, the session and Steam. With
   swappiness 10 the kernel drops the clean page cache (the model file, free
   to re-read) instead, and the run is stable. `/etc/udev/rules.d/30-zram.rules`
   with `"10"` makes it stick across boots.
2. **A cgroup cap does not stop a *global* OOM.** `systemd-run --user --scope
   -p MemoryMax=18G -p MemorySwapMax=0` kept the desktop alive (the kernel's
   OOM killer only reached the scope) but the *global* limit still fired: the
   engine's scope peaked at 15 GB, and 15 GB + the desktop's 8.5 GB + zram's
   compressed pages is already the whole machine. So the cap has to leave
   headroom for everything else: **12 GB** on this box, with `--arena-gib 4`.

## To reach the ceiling

The goal is the most tok/s this box can do, and the two cards should be used
more and more. The levers, in the order they pay:

1. **The capacity split** (this change). Each card holds its own layers'
   experts, so the two hold about twice what one card holds. On the reference
   box the expert tier is 78% of a decode step; the VRAM-resident share goes
   from ~13% of the expert slots to ~26%, and the file tier's traffic halves.
2. **The drafter on the last card.** The MTP head belongs to the last stage
   (the qwen4exp path measured 52-59 tok/s with the drafter on the first
   card against 68-71 with it on the last). DSpark is already in this engine;
   it needs the same treatment the split gives the dense half.
3. **The stage overlap across verify windows** - the big one. The qwen4exp
   path's `--pipeline-windows 2` gives every stage **two verifiers, one for
   the even windows and one for the odd ones**, so stage 0 can run window
   K+1 while stage 1 still reads window K's hand-off. The stages then overlap
   and the step's wall clock is the slower stage, not their sum. On this box
   that path measured 41.4 -> ~70-71 tok/s.
   It needs the serve path (the verify windows), the drafter (so there are
   windows to overlap) and a second copy of the per-pass state per stage (the
   residual, the pre-mix, the routed sums - the position-indexed KV and
   indexer state is shared).
4. **The async tier and the adaptive knobs** (`--adapt-async 1`,
   `STRATA_ADAPT_LAG=2`): +6% on the qwen4exp path.

Without speculation there is only one window at a time, so the stage overlap
has nothing to overlap: the split alone is a capacity win, and the overlap
needs the drafter. That is the order to build it in.

**The dense half is the floor.** 7.6 GB of dense weights are read per token at
~850 GB/s (84% of the card's peak) on the reference 4090; the split does not
change that (the layers are serial), only the overlap does.

## Measured

All on this box (2x RX 7900 XT 20 GiB, Ryzen 7 7800X3D, 31.7 GiB RAM), engine `build-hip-pre/ds41_generate`,
`STRATA_DEC_PROF=1`, `--prompt-batch 1` on the 75-token `local/prompts/code-review.i32`, greedy, one run each
unless noted. The two cards' VRAM totals are identical, so "one card" means HIP device 1 of 2 and the other card
is idle (the desktop lives on it).

Displacement of a token, one card, 4 GiB arena, 1253 VRAM slots, 14871 file-tier experts, at two model locations:

| phase (ms/token) | model on the Micron 2450 (PCIe 3.0 x4) | model on the Crucial P310 (PCIe 4.0 x4) |
|---|---:|---:|
| begin | 1.86 | 0.62 |
| predict | 0.17 | 0.15 |
| attn_router (dense half) | 50.62 | 47.33 |
| tier (experts) | 55.76 | 36.51 |
| - of which file reads | 40.3 | 22.1 |
| finish | 5.56 | 4.88 |
| **decode** | **8.72 tok/s** | **10.99 tok/s** |

The whole difference is the file tier. Same 671 file-tier misses in 70 tokens (9.6 a token) on both drives; the
per-miss cost fell from 4.2 ms to 2.3 ms. Agrees with a raw aligned O_DIRECT probe of the two volumes: 8 MiB random
reads p50 3.40 ms (2.35 GB/s) on the Micron, 2.12 ms (3.77 GB/s) on the P310.

What did **not** help:

| change | decode | why |
|---|---:|---|
| arena 4 GiB -> 8 GiB (489 -> 972 arena experts) | 8.69 tok/s (no change) | the arena is seeded from `--rank`, a static profile; the extra experts are not the ones this text routes |
| VRAM slots 1253 -> 400 | 3.96 tok/s | 1884 file-tier misses instead of 607; the cache is what keeps the tier cheap |
| the DSpark drafter (`STRATA_DSPARK=the MTP head`, depth 4, layer-major prompt) | 7.17 tok/s against 10.14 without | 2.464 tokens a pass but the verify pass costs 332.5 ms = 3.4 single-token steps, and only 37% of drafts are accepted (21.4% of the depth-4 window). It needs a much higher acceptance rate to pay for the verify |
| `--kv-q8` | 10.42 vs 10.14 | inside one run's spread |
| `--pf-b 2.0` | 10.41 vs 10.14 | inside one run's spread; the author's cliff below 1.0 did not appear here |
| `--prompt-batch 2048` (the harness's serve setting) | 10.14 vs 10.99 at `--prompt-batch 1` | the batch path defers the VRAM seed until after the prompt; at decode time it settles a slightly worse cache |

What did: **software streaming.** The engine's two read-ahead mechanisms were both off, and both work on this box:

| config | decode | tier | file wait |
|---|---:|---:|---:|
| O_DIRECT decode reads (`STRATA_DECODE_DIRECT` default) | 10.99 | 36.5 ms | 22.1 ms |
| `STRATA_SPEC_TOP=4` (predicted experts read into a stash) | 11.80 | 24.4 ms | 10.0 ms (568 read-ahead, 414 used) |
| `STRATA_ADVISE_AHEAD=2` + buffered reads | 12.07 | 18.9 ms | 5.4 ms |
| `STRATA_ADVISE_AHEAD=4` | 13.00 | 17.7 ms | 4.2 ms |
| **`STRATA_ADVISE_AHEAD=6`** | **13.34** | **16.9 ms** | **3.5 ms** |
| `STRATA_ADVISE_AHEAD=12` | 13.23 | 16.7 ms | 3.2 ms |

`STRATA_ADVISE_AHEAD` predicts layer l+k's routing and issues `posix_fadvise(WILLNEED)` for the predicted
file-tier experts, so the kernel fills the page cache while the card is busy with the dense half; the tier then
finds them in cache. It needs buffered decode reads (`STRATA_DECODE_DIRECT=0`) because O_DIRECT bypasses the page
cache. The predict graph it adds costs 4.6-7.8 ms a token and saves 16-19 ms of waiting. At lead 6 the remaining
file wait is 3.5 ms of a 75 ms token (4.7%).

### Requantizing the dense half (opt-in, a quality trade)

`STRATA_DENSE_REQUANT=<ggml type>` re-quantizes every **Q8_0** dense matrix (attention projections, the shared
expert, and `output.weight` unless `STRATA_DENSE_REQUANT_HEAD=0`) in memory at load; the file is not touched, so it
is reversible by restarting. It is a quality trade, off by default.

Measured (fused build, `--pf-b 0`, `STRATA_ADVISE_AHEAD=6`, buffered decode reads, 70 tokens, one run each):

| dense type | decode | attn_router | tier | VRAM hit rate | file reads |
|---|---:|---:|---:|---:|---:|
| **Q8_0 (default)** | 16.7-17.1 | 30.6-31.5 ms | 17.8-19.6 ms | 96.0% | 629 / 70 tokens |
| q6_K | 18.45 | 27.5 ms | 18.0 ms | 96.5% | 578 |
| q5_K | 19.54 | 26.4 ms | 16.5 ms | 96.5% | 574 |
| **q4_K** | **21.84** | **25.6 ms** | **12.2 ms** | **98.8%** | **197** |

Two effects, and the second is the bigger surprise: halving the dense bytes saves ~5 ms in the dense phase, **and
frees ~2.5 GiB of VRAM**, which the tier spends on experts (+~600 slots) - so the file-tier reads fall by 5x and the
tier loses another ~6 ms. The VRAM effect only appears at q4_K; q6_K and q5_K gain from the dense phase alone.

That is a real gain for a real compromise: the dense weights become lossy, so it needs a quality gate before it
could ship (the engine has `--ppl` for exactly this). It stays OFF in `local/bench-v41.sh` and
`local/ds41-hivebench.json` until that gate has run.

### The long-context crash, and the fix (2026-10-24)

Any prompt past ~512 tokens died with:

```
top_k: nrows=309 ncols=2048 k=512
ggml_cuda_compute_forward: TOP_K failed
ggml-cuda (strata mmq): invalid configuration argument
```

Pre-existing (reproduced on the pre-fused binary and with graphs disabled). The chain: the DSA indexer calls
`ggml_top_k` over the whole candidate set, so `ncols` tracks the context (~2x the position). `GGML_OP_TOP_K` has its
own cuda implementation whose fallback calls hipCUB's *segmented* sorts directly, and on this ROCm **both**
`DeviceSegmentedSort` and `DeviceSegmentedRadixSort` fail with "invalid configuration argument" for segment widths
past ~1024. The bitonic kernel can't help either (one block = 1024 threads).

The fix was already in the tree and unapplied: `tools/v41/ggml/rank-argsort.patch` is a tiled rank sort (each thread
owns several columns via `c += blockDim.x`, keys in shared memory), so it handles any row width. Three changes:

1. apply the patch (one CUDA-ism needed fixing: its warp shuffle passes 3 args, HIP's shim wants a width);
2. raise its gate from `ncols <= 1024` to `<= 16384` - 4 bytes per column of shared memory, so 16384 columns is
   exactly gfx1100's 64 KiB;
3. point the engine's two indexer top-k calls at `ggml_argsort_top_k` instead of `ggml_top_k`, because that routes
   through the argsort op (the patched rank path) rather than `GGML_OP_TOP_K`'s own hipCUB fallback.

Verified after the change: 1,333 tokens -> PPL 2.2662 (top-1 0.794); 5,389 tokens -> **PPL 2.8739 (top-1 0.755)**
at 20.7 tok/s prefill; decode unchanged at 16.24 tok/s with tokens identical to the reference. Before the fix the
first of those crashed.

Ceiling: 16384 columns is ~8K tokens of context. Beyond that the rank kernel would need its keys in global memory
(no shared array) - not written yet.

### Quality, measured (a 5,389-token code prompt, `--ppl`, the real gate)

| arm | decode | PPL | top-1 |
|---|---:|---:|---:|
| fused ops, Q8_0 dense (default) | ~17 | **2.8739** | 0.755 |
| composite ops (unfused), Q8_0 dense | ~15 | **2.8712** | 0.755 |
| fused ops, q4_K dense | **21.84** | **2.9084** | 0.753 |

Two corrections to what an earlier 358-token corpus suggested:

* The fused ops are **not** a 4.5% quality cost. On a real corpus they cost **0.09%** (2.8712 -> 2.8739) for ~7%
  decode - essentially free. The 4.5% was short-context noise; 358 tokens cannot resolve it. An earlier note also
  argued the fused ops were free *because the parity tests pass* - that reasoning was wrong even though the
  conclusion turns out close to right. This measurement, not the parity test, is the evidence.
* The dense requant to q4_K costs **1.2%** perplexity (2.8739 -> 2.9084, top-1 0.755 -> 0.753) for **+28% decode**.
  A fair trade, and now enabled in `local/bench-v41.sh` and `local/ds41-hivebench.json`; the engine default stays OFF.

### Where the token went at 13.34 tok/s (superseded, see the profile below)

| phase | ms | share |
|---|---:|---:|
| attn_router (dense half, 40 layers) | 47.0 | 63% |
| tier compute | 13.4 | 18% |
| tier file wait | 3.5 | 5% |
| finish + predict + begin | 9.8 | 13% |

The dense half is now the target. For scale: 40 layers of dense weights (~10 GiB) at this card's ~800 GB/s HBM
bandwidth is a ~12.5 ms floor, so 47 ms is 3.8x off it. The author's own rocprofv3 trace on a gfx1100 card puts
**4,238 kernel launches a token at ~5.6 us = ~24 ms** in launch overhead, i.e. half of this phase, and their fused
DSV41 ops (`GGML_OP_DSV41_FAKE_QUANT`, `GGML_OP_DSV41_HC_MIXES`, in `tools/v41/ggml/*.patch`) remove ~3,000 of
those launches for a measured +5% on CUDA (never re-measured on HIP, where launch overhead is higher).

### Where the token actually goes (21.8 tok/s, profiled 2026-10-24)

Measured with **rocprofv3 1.1.0** out of the CachyOS `rocprofiler 7.2.4-2.1` package, extracted under `/tmp` and run
from there (nothing installed, no system change).  The engine ran with the current best flags, `--kernel-trace
--stats`, a 75-token prompt and `-n 1` / `-n 5`; the difference between the two traces is exactly four decode
tokens, which cancels the model load and the prefill.  Repro: `local/dense-trace.sh`, then
`local/trace-analyze.py`.

**Kernel time is 24.1 ms of the ~45 ms token.**  By family, over the four-token window:

| family | ms/token | calls/token | us each |
|---|---:|---:|---:|
| dense GEMV (`mul_mat_vec_q` Q4_K, and the f16/f32 ones) | 10.09 | 321 | 21.9 |
| tier's fused expert kernels (`native_*_multi`) | 5.36 | 101 | 47 |
| elementwise / `rms_norm` / `rope` | 3.47 | ~900 | 3 |
| PCIe expert copies (`__amd_rocclr_copyBuffer`) | 1.34 | 777 | 1.7 |
| hyper-connections | 1.23 | 355 | 3.5 |
| attention (`flash_attn_tile`) | 0.83 | 40 | 18.2 |
| quantize / swiglu | 0.66 | 399 | 1.7 |
| top-k / argsort | 0.48 | 74 | 6.6 |
| other | 0.63 | | |
| **kernel-busy** | **24.1** | 4,164 | |

The other ~21 ms, from `STRATA_DEC_PROF` and the kernel totals above:

* **attn_router 25.2 ms** against roughly 17 ms of kernels: **~8 ms of host time around 40 layers** (graph
  submits and one synchronize per layer).
* **tier 12.1 ms** against 6.7 ms of kernels and copies: **~5.4 ms**, of which ~1.8 ms is file wait.
* **predict 5.0 ms**: the `STRATA_ADVISE_AHEAD=6` predictor, one tiny synchronised graph per layer.  It pays for
  itself - without it the file wait is 22.1 ms - but it is the single largest piece of removable host work.
* begin 0.5 ms, finish 1.4 ms.

**Correction to the paragraph above.**  The author's "4,238 launches at 5.6 us = 24 ms of launch overhead" does not
apply to this build.  Ours are 4,164 *graph nodes*, and the graphs are already captured: with
`STRATA_LOG_DEBUG=1` (a new opt-in switch, default off) the engine prints 210 `CUDA graph warmup complete` and
thousands of `Graph id reused` for a five-token run, because `GGML_CUDA_GRAPHS=ON` is set in the ggml build this
engine links.  So there is no 24 ms of launch overhead waiting to be reclaimed here, and "remove 3,000 launches"
is not a lever on this box.

What that leaves, in order of value per risk:

* The Q4_K GEMV is 4.02 GB in 7.03 ms = **572 GB/s**.  The card's spec is 800 GB/s; we have not measured this
  card's practical ceiling with a copy kernel, so treat 572 as "close enough that the kernel is not the first
  target" rather than as a percentage.
* Replacing the advise-ahead predictor with the previous token's actual routing would remove the 5.0 ms of syncs
  and leave the computation untouched (the hint only warms the page cache, so tokens stay identical).  Untested.
* The ~8 ms in the dense loop is host submit time.  A profiler cannot see it; it needs counting the submits.

### The ahead-predictor was removable (measured 2026-10-24)

The 5.0 ms `predict` phase above was `STRATA_ADVISE_AHEAD=6`: one synchronised prediction graph per layer, whose
only effect is a call to `advise_file` (a page-cache hint).  Routing barely changes between adjacent tokens, so
`STRATA_ADVISE_PREV=1` hints layer l with the experts it actually routed on the previous token and drops the
prediction graph entirely.  A hint cannot change the arithmetic - `advise_file` never moves an expert - but it is
opt-in and the engine default stays off.

200-token runs, same prompt, alternating arms, three each (fused build, q4_K dense, advise-ahead 6, direct 0):

| arm | tok/s | predict ms | attn_router ms | tier ms |
|---|---:|---:|---:|---:|
| ahead-predictor | 22.27 / 22.65 / 22.86 (mean **22.59**) | 3.76-4.07 | 26.3-26.8 | 10.8-11.4 |
| **previous token** | **23.80 / 25.36 / 24.82 (mean 24.66)** | **0.18** | 25.1-26.0 | 11.1-12.3 |

**+9.2%**, and every variant run beat every baseline run.  Quality: PPL **5.6463 either way** on the 5,389-token
corpus at `--prompt-batch 1` (the per-token path that actually runs the changed code), and the 75-row logit dump is
byte-identical.  That 5.6463 is not comparable to the 2.9084 baseline above: different prefill batch sizes produce
different bits, which is why the gate compares arms, not absolutes.  Enabled in `local/bench-v41.sh` and
`local/ds41-hivebench.json`.



### The prefetch was a loss on this box (biggest single finding)

`--pf-b` (predicted prefetch into a VRAM slot) defaults to 1.43. It only works when the predicted expert is
already in the host arena: an expert outside it still costs its DMA, sent from a pinned **zero blob**
(`ds41_moe.cpp`: "a guess outside the arena still costs its DMA"), and the guess is then unused. With 4 GiB of
arena holding 489 of 16,613 experts, essentially every guess is outside, so the engine was issuing ~903 dummy
7.75 MB DMAs per 70 tokens — **~100 MB of PCIe traffic per token** — plus a predict graph per layer, for
`0 used`. That traffic was stealing bandwidth from the dense half.

| config (fused build, advise 6, buffered reads) | decode | attn_router | finish |
|---|---:|---:|---:|
| `--pf-b` default 1.43 | 13.71 | 41.9 ms | 4.9 ms |
| `STRATA_PF_NO_DUMMY=1` (no DMA for a guess outside the arena) | 15.42 | 33.3 | 4.9 |
| **`--pf-b 0`** | **18.36 / 17.42** | **30.4 / 32.2** | **1.5** |

`STRATA_PF_NO_DUMMY=1` recovers most of it by skipping the useless DMA; `--pf-b 0` also removes the per-layer
predict graph from the finish path (`finish` 4.9 -> 1.5 ms). The streaming does not depend on it: the file wait
stays at 3.4 ms because `STRATA_ADVISE_AHEAD` does that job. Lead sweep with prefetch off: 4 -> 16.10,
**6 -> 18.36 / 17.42**, 12 -> 16.67 tok/s.

This is specific to a box without the RAM for a large arena. On the author's 128 GiB rig prefetch is worth
keeping (their `--pf-b` cliff below 1.0 is real); here it is pure waste. Set in `local/bench-v41.sh` and
`local/ds41-hivebench.json` (which also drops `--arena-gib` 12 -> 4: 12 GiB of *pinned*, unevictable arena on a
31.7 GiB box, measured no faster than 4).

### The fused dense ops (opt-in build, on)

`tools/v41/ggml/dsv41-fake-quant.patch` and `dsv41-fuse.patch` collapse the per-layer activation rounding
(`GGML_OP_DSV41_FAKE_QUANT`) and the two hyper-connection mixes (`GGML_OP_DSV41_HC_MIXES`) into single nodes. The
patches were written against the author's llama.cpp, but they apply to the PTQ1_0 tree this engine links
(`work/prism-qwen4exp-shrink`, commit `986cff62c`, clean before patching, `git checkout .` reverts), fake-quant
first and then fuse because the fuse hunks stack on fake-quant's enum. Built with `-j6` (the full ggml rebuild at
20 jobs wants ~80 GB and this box has 31.7; it took ~15 min at 6), then the engine re-configured with
`-DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON`.

Gates, in order:

- `test_ds41_fq --cuda` and `test_ds41_hcmix --cuda`: **0 values differ bitwise**, "fused == composite, bit for bit".
- Decode token identity against the pre-fused binary at the same settings: the first 16 tokens are **identical**
  (`85525 201 650 2595 18121 6849 2366 10689 11699 31513 374 11 875 361 688 3452`).

Measured on the same libraries, alternating binaries:

| | attn_router | decode |
|---|---:|---:|
| pre-fused | 46.8 / 50.6 ms | 12.81 / 13.34 tok/s |
| **fused** | **41.9 / 41.4 ms** | **13.71 / 13.62 tok/s** |

~9 ms a token off the dense half (18%). The author measured +5% on CUDA; HIP's launch overhead is higher, as they
predicted. Run-to-run decode noise on this box is ~4%, so the phase timer is the reliable signal here.


### Peer-to-peer and hipFile

`hipDeviceCanAccessPeer` returns 1 in both directions on this box, so the platform permits PCIe peer-to-peer DMA
between the two cards. AMD's cuFile equivalent is **hipFile** (direct transfers between GPU HBM and storage,
bypassing host memory; direct-to-GPU storage I/O since ROCm 7.2, GA in 7.14). This box runs ROCm **7.2.4**, so the
runtime is new enough, but hipFile is not packaged for Arch and would have to be built from ROCm/hipFile.

Measured reason to be careful about the payoff: the file wait is now 3.5 ms a token, so a perfect NVMe->VRAM path
can save at most ~4% today. Its real value is architectural - prefetching a predicted expert straight into a VRAM
slot with no host bounce and no arena - which is what this box lacks (31.7 GiB of RAM against the reference's 128).

**Gate run 2026-10-23, before building anything** (`tools/ais-check/ais-check` from ROCm/hipFile, whose three checks
can be reproduced directly):

| check | result |
|---|---|
| kernel `CONFIG_PCI_P2PDMA` | **True** (`/lib/modules/7.3.0-4-cachyos/build/.config`) |
| HIP runtime `hipAmdFileRead` / `hipAmdFileWrite` | **True** (both resolve via `hipGetProcAddress`, HIP 7.2.53211) |
| amdgpu `kfd_ais_rw_file` in `/proc/kallsyms` | **False** |

So the kernel and the HIP runtime are ready and only the **driver half** is missing: the in-tree `amdgpu` of kernel
7.3.0-cachyos does not carry AMD's AIS hooks, and hipFile's INSTALL asks for `amdgpu-dkms >= 30.20.1`. Not a
visibility artifact: 5,626 amdgpu/kfd symbols are readable in `/proc/kallsyms` and none matches `ais`. On Arch the
DKMS driver is AUR-only, needs to build against a CachyOS-patched 7.3 kernel, and would replace a working in-tree
driver - a system-level swap with real risk for a ceiling of ~6% of the token. Not worth it today; the trigger to
revisit would be the kernel gaining the hooks, or a model that outgrows VRAM (where a direct-to-VRAM cache is the
whole point).

Also note hipFile supports only **ext4 and xfs**, and this model currently lives on **btrfs**. A prototype would
have had to use the Micron (XFS, now 231 GB free) - which is PCIe 3.0 and therefore the slower of the two.


The blob mirror (`--build-mirror` / `STRATA_BLOB_MIRROR`) was considered and dropped without a copy: the engine's own
note says the fast case is a file of ~1,226 extents against 62,112 extents' 9K reads/s, and this model file has
**1,326 extents** on the P310 (`filefrag -v`). There is nothing for a repack to fix.

The reference numbers for comparison: the author's AMD box (7900 XTX, 128 GiB RAM, 75 GiB arena) measures
12.80-14.00 tok/s at 4K-64K state, with `attn_router` 57.5 ms and `tier` 14.5 ms. Our dense half is already faster
than that (47.3 vs 57.5 ms) on a much shorter state; our tier is 2.5x theirs because they hold most experts in
RAM and we cannot.

### The two-card split (work in progress)

`--layer-split 20 --split-device 1` runs, and on the same GPU as well as on two (the author's gate for the
hand-off). Two defects were found and one fixed:

- **Fixed:** the decode loop passed `rdev1` (the first stage's routed tensor) to `run_to_dev` for *every* layer,
  including the second stage's. Stage 2 wrote its expert sum through a device pointer belonging to card 0; the HIP
  runtime faulted with "page not present" on node 2. It now passes `rdev`.
- **Open, speed:** the tier costs 8.8 ms per layer in a section that does nothing when `nk == 0` (the split
  reports `0 cpu`), identical on one GPU and two, unchanged with `--pf-b 0`. 420 ms/token against 55.8 on one
  card, so the split is currently a 4x *loss*.
- **Open, correctness:** its tokens differ from one card from token 2 onward. That is not yet evidence of a bug:
  a split at 400 slots/card keeps a different resident set than one card at 400, so different experts run on the
  CPU and round differently. The author's equivalent test used a fixed cache. A gate that removes the confound is
  needed before the split can be trusted.

Re-run 2026-10-24, after the streaming and routing changes.  With the cache sized from free VRAM (`--arena-gib 1`,
no `--slots`) the two-stage load was stopped by the guard at **16.7 GiB of GTT** before the second stage started;
stage 1 alone had reserved 1.82 GB of dense weights and **16.02 GiB of expert slots**.  The overflow was the cache
sizing, not the box: the same run with **`--slots 400`** loaded both stages cleanly - card0 6.0 GiB, card1 8.0 GiB
of VRAM, GTT under 0.1 GiB, 19.9 GiB of RAM free.  So the byte budget over-allocates for this 9-format mixed-size
model, and the excess lands in system memory.

That does not make the split useful yet. 400 slots a stage is 3.2 GB of cache against one card's 13.7 GB, and
prefill fell to 4.5 tok/s.  Beating one card needs roughly 1,200-1,600 slots a stage - the range the byte budget
blew up in - and decode then aborted with `[ds41_dense] missing weight blk.20.ffn_norm.weight`, a stage-2
weight-range bug the prefill path does not hit.

Verdict: still parked, but for three specific reasons now (the mixed-size cache sizing, the boundary weight, the
8.8 ms/layer stall), not because the box cannot fit it.  Even with all three fixed the ceiling is small: layers run
sequentially, so card 2 is idle while card 1 computes, and the only gain is fewer file-tier misses - about 1.5 ms
of a 41.6 ms token.

Neither defect is the box's memory. The earlier global OOMs were blamed on the engine's cgroup, which was wrong:
its peak was 6.9 GB of a 12 GB cap, and at the OOM the kernel's own dump accounted for only ~17.6 GB of 29.6 GB.
The missing ~12 GB is consistent with amdgpu's GTT, whose budget here is 29.4 GB of system RAM and whose pages are
charged to the driver rather than the process, so no cgroup cap can see them. `local/mem-guard.sh` now watches
`mem_info_gtt_used` and `MemAvailable` every 0.3 s and kills only the engine; it fired at 1.8 GB available and
saved the box where the 2 s watchdog lost the race.


---

## Single-card lane pass — 2026-10-24

### Baseline reproduced
25.42 tok/s mean (25.96 / 25.28 / 25.01), q4_K PPL 2.9084 / top-1 0.753. Phase: attn_router 25.0, tier 11.3, finish 1.4, begin 0.5 ms.

### Instrumentation added
`STRATA_DEC_PROF=2` now counts graph submits / synchronizes / host copies (and their ms) in both the dense half and reports them every 64 steps. Dense per token: **82 submits (40 attn + 40 finish + 2 other), 40 syncs, 137 host copies**; submit 27 ms (attn 25, finish 1.4), sync 0.5–2.5, hostcopy 2.0 ms.

### Experiments run
| change | result | decision |
|---|---|---|
| Merge per-layer `finish()` into next `attn()` graph | submits 82→43, tokens identical, but attn_router unchanged (~25 ms) and decode slightly slower (23–24) | **reverted** — per-graph overhead is not the cost; per-kernel dispatch is |
| Disable CUDA graphs (one-shot, 3 runs) | 23.1–23.6 vs 24.1–24.5 tok/s | **reverted / stays parked** — loss on this path |
| `--kv-q8` | PPL 2.9049 vs 2.9084 (top-1 identical), decode 25.15/25.73/24.61 vs 25.05/26.07/23.14 (noise) | **not adopted** — no measurable decode win |
| DS41Engine sync-bound analysis (external) | 215 syncs/token, 105 ms, 76% of wall; tier holds ~175 of them | **next engineering task** — extend the counter to the tier, then reduce per-layer syncs |

### System-level
`local/power-tune.sh` added (CPU governor + NVMe APST + C-states), +11–21% expected. Requires root; not auto-applied.

### Conclusion
No merged change reaches ≥26 tok/s on this pass. The dense graph merge and graph-disable are dead ends on this path; the remaining ~8 ms is per-kernel HIP dispatch inside the 40 attention graphs, and the sync-bound tier (per DS41Engine) is the unmeasured cost. Next lever: tier-side instrumentation + sync reduction, then kernel fusion (custom ops) if warranted.

### Lane closed — 2026-10-24

No merged change reached ≥26 tok/s on the single-card decode path. Final best: **25.42 tok/s mean** (baseline 24.66 → +3.1%, inside the ~4% run-to-run noise band for a "merged" claim).

Exhausted levers (all reverted or not adopted): graph-merge of finish into attn, graph-disable, --kv-q8, DSpark (slower on this path), tier ternary alternatives. Remaining gap (~0.6 tok/s) is per-kernel HIP dispatch inside the 40 attention graphs; closing it needs new fused kernels, out of scope for this lane.

Durable deliverables: `STRATA_DEC_PROF=2` counter (dense), `local/power-tune.sh`, the runtime-trace methodology, and this report. Serve path already reaches 26–30 tok/s with DSpark on contexts where it helps.
