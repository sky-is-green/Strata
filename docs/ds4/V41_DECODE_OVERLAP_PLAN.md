# V4.1 decode: how to get the GPU busy (implementation plan)

Companion to `V41_RESUME_PROMPT.md` §UPDATE 2026-10-12 ~20:45.  Written 2026-10-12 ~21:00 after the
host-call attribution (conductor, `nsys/dec3.sqlite`) and the independent overlap scan
(`~/.quetza-data/conductor/reports/decode-overlap-scan.md`).  Every number below is measured, not inferred.

## 1. The problem, measured

Per decode token (256-token prompt, greedy, 4.5 s trace window, 96 tokens):

| | ms/token |
|---|---|
| GPU busy (union of all kernels+memcpy+memset) | 18.27 |
| **GPU idle** | **28.60 (61%)** |
| — host blocked in `cudaStreamSynchronize` | 26.21 total; owns 11.28 of the gaps |
| — **no single call owns the gap** (~884 short host calls) | 14.68 |
| — of that, **host is idle too (= a pure pipeline bubble)** | **12.21** (scan §3.3) |

1680 host CUDA calls/token: `cudaLaunchKernel 417.7` + `cudaMemcpyAsync 387.1` + `cudaStreamSynchronize 257.7`
+ `cudaGraphLaunch 80.7`.  **By stream:** `14 = gp.s (the tier)` 322 kernels/tok, 89.2 syncs = 5.03 ms;
`16 = ggml's dense stream` **121.7 syncs/tok = 17.02 ms** (3/layer, mean 139.9 us).
**Max concurrent kernels = 1** over the whole window - the two streams never overlap.

Two independent attributions agree on all of this (counts within window trim), which is why the numbers are
trusted.

## 1b. BASELINE for slice A - measured on `gen.v7` (packing + S4 in), `nsys/dec7.sqlite`

Reproduce with `python3 ~/AI/v41-dev/nsys/calls.py nsys/dec7.sqlite 96`.

| metric | gen.v7 | dec3 (before packing/S4) |
|---|---|---|
| GPU busy / idle | 18.42 ms/tok / **60.7% idle** | 18.27 / 61% |
| **MAX CONCURRENT KERNELS** | **1** | 1 |
| host calls/token (total) | **1606** | 1680 |
| `cuKernelGetName` | 418.5 | - (not in the top list) |
| `cudaLaunchKernel` | 418.5 | 417.7 |
| `cudaMemcpyAsync` | **350.7** | 387.1 |
| `cudaStreamSynchronize` | **217.5** | 257.7 |
| `cudaEventRecord` | 106.1 | 105.4 |
| `cudaGraphLaunch` | 80.6 | 80.7 |
| `cudaStreamWaitEvent` | 10.0 | 10.1 |
| stream sync ms/tok | s16 17.13, s14 4.73 | s16 17.02, s14 5.03 |
| kernels/token | s14 322.9, s16 56.3 | s14 322.1, s16 56.2 |

**The two changes show up exactly in the counts and nowhere else - which is the point:**
- S4 removal = **-40.2 syncs/token = 1.0/layer**, precisely as designed, and still speed-neutral because the wait
  moves into the sum sync (the tier profile proves it).
- Metadata packing = **-36.4 memcpyAsync/token, NOT the ~280 that "one copy per class" predicted** - most groups
  have only 2-3 distinct format classes among 6 experts, not 9.  That is why it measured neutral, and it is now
  explained rather than just observed.
- `cuKernelGetName` is issued **once per kernel launch** (418.5/token) - a pure diagnostic, ~0.09 ms/token,
  removable for free if anyone wants the last crumbs.

**Slice A's acceptance test:** `MAX CONCURRENT KERNELS > 1` in an `STRATA_DMA_OVERLAP=1` trace, plus forced-token
identity across >=3 runs (it is a race, so one clean run is not evidence).

## 1c. Slice A VERDICT + the volume finding it exposed (2026-10-12 ~21:35)

**Slice A is implemented and CORRECT, but its ceiling is small - and measuring it exposed where the bytes are.**

- Implemented: `STRATA_DMA_OVERLAP=1` (default off) puts the PCIe group's blobs on a new copy stream `gp.s_dc`
  with one event handoff before the miss kernels (see 3b).  It demonstrably took effect: `gp.s` (stream 14)
  memcpy fell **69.87 -> 43.27 MB/token** and the new stream holds the **26.59 MB** the PCIe group moves.
- **Correctness PASS**: 5 forced-token runs (2 off, 3 on) all byte-identical to the forcing file - no corruption,
  which is the bar that matters because this is a race.
- **Speed: no gain.**  off 26.87 / 26.96 tok/s, on 26.52 / 26.55.  GPU idle 60.7% -> 59.3% (only ~0.5 ms/token
  filled).  The ceiling is arithmetic: the PCIe group moves 26.6 MB/token = **~1.7 ms at 15.8 GB/s**, and only
  part of that hides.  Below the noise floor of tok/s, so this is a keep-or-revert-on-taste change, not a win.
- **METRIC BUG FOUND AND FIXED:** `calls.py` swept only `CUPTI_ACTIVITY_KIND_KERNEL`, so a kernel||DMA overlap is
  invisible - it read `MAX CONCURRENT KERNELS = 1` for BOTH traces.  It now sweeps kernels+memcpy+memset and is
  named `MAX CONCURRENT ACTIVITIES`.  On these traces it reads **2 for both** - i.e. slice A did not raise the
  peak concurrency either, consistent with the tiny gain.

**THE FINDING - the H2D volume is one activity, and it is almost all PREFETCH:**

| stream | MB/token | what |
|---|---|---|
| `s_pf` (prefetch) | **274.71** | the speculative expert prefetch - 29.2 DMAs/token of 9.5 MB |
| `gp.s` / demand (pcie+file+metadata) | 69.87 -> 43.27 | the actual misses |
| ggml's stream / s_cg | 0.93 | activations |
| stream 13 | 0.58 | - |

**274.71 MB/token at 15.8 GB/s is 17.4 ms/token of PCIe time - comparable to the entire 37.5 ms token - and the
engine's own counters say only 2499 of 8065 prefetch DMAs per 256 tokens are ever USED (31%; ~10/token vs
~31.5/token issued).**  So roughly two thirds of the expert traffic (and 5x the demand traffic) is speculative
guesses that miss.  Even if it overlaps, it competes for the same PCIe path as the demand copies that ARE on the
critical path - which is the more likely explanation for why the demand-copy overlap (slice A) bought so little.

**Therefore the next slice is not more overlap, it is LESS PREFETCH VOLUME:**
1. skip a prefetch whose expert is already VRAM-resident (is it even checked? - verify in `prefetch()`), and
2. cut the issued set: today `kPredW=16` / `kMaxPf=6` issues ~29 DMAs/token for ~10 used - try issuing only the
   top 4-8, or only when the prediction score is above a percentile (the Fate widening, scout §1), and
3. re-measure with `MAX CONCURRENT ACTIVITIES` plus the `prefetch N issued / D outside / U used` counters, not tok/s.
The prefetch's own break-even was measured NET NEGATIVE once already (resume doc: finish graph +1.5 ms, issuing
+1.3 ms, vs -1.8 ms tier) - this volume number says why, and says the fix is to send far less.

## 2. Why it is serial - the root cause

Per layer the chain is `attn_router(l)` -> **[host reads the top-k ids]** -> `tier.run_to_dev(l)` ->
`finish_predict(l)` -> `attn_router(l+1)`.  The host cannot launch the tier until it has the ids, and the tier's
own launch decisions (which tier each expert is in, the PCIe budget, arena slot -> device pointer) are host loops
over host-side tables (`gp.arena.slot_of`, `gp.cache`, `skip_miss`, the residency bias).  So every layer is a
host round trip, the device queue drains between layers, and the expert H2D copies (328 MB/token, scan §3.1) sit
between the syncs instead of under the compute.

## 3. Targets, in order

### Slice A - overlap the expert H2D with compute (scan's #1; ceiling 11-14 ms/token; output EXACT)
The weights are a *copy*, so reordering/overlapping changes no bits.
- The destination is a fixed per-layer staging region (`gp.d_stage`, `gp.d_pf` + `pf_at` stride, `B` = blob
  stride) and the source is a stable pointer for the token (`gp.arena.ptr(layer, e)`, a pinned file-stage buffer,
  or a prefetch slot).
- Today the copies are issued on `gp.s` (the compute stream) *interleaved* with the launches
  (`ds41_moe.cpp:2513` "pcie dma" in `run()`; the `gpu_run` equivalents at the PCIe/file group builds), so they
  serialise against the kernels on the same stream.
- Change: issue the *demand* copies for layer `l` on a **copy stream** (`gp.s_cp`, already exists,
  `ds41_moe.cpp:2787`) with an event handoff to `gp.s` (`cudaStreamWaitEvent`, the pattern is already used at
  2654/2708/3052/3084/3164/3197), and **double-buffer**: once layer `l`'s experts are known, the copies for the
  next layer's *predicted* experts (already produced by `finish_predict`, `ds41_dense.cpp:1376`) go out
  immediately, so they run under `attn_router(l+1)` + `tier(l+1)`'s kernels.
- Requires per-layer (not one-shot) staging regions: currently `gp.d_stage` is one region per call, and the
  prefetch slots are `pf_at`-strided.  Give it two (ping-pong) regions and an event per region.
- Verify: forced-token identity (`STRATA_FORCE_IDS`), then the nsys per-stream call counts and the **overlap
  metric** (max concurrent kernels > 1 is the whole point).

### Slice B - collapse the host calls (prerequisite for C; the packing part is coded)
1. **`launch_group` metadata packing - CODED** (`ds41_moe.cpp:1900`): every format class of a group is packed
   into `h_meta` contiguously and copied up in ONE `cudaMemcpyAsync` (was one per class; `STRATA_GRP_ONE_COPY=0`
   restores).  ~7 fewer calls/layer (~280/token).
2. **CUDA-graph the tier's class launches.**  `native_expert_grouped` already takes ptr/st/ng/dst/tok as DEVICE
   pointers into the metadata block, so a captured graph's node params are FIXED - only the block's contents
   change per token.  Pack into **fixed per-class slots**, launch all 9 classes every layer (absent -> n=0; the
   kernel must accept `ng == 0`), capture once at init.  Per layer: 1 memcpy + 1 `cudaGraphLaunch` instead of
   ~20 calls.  **884 -> ~80 calls/token.**
3. `finish_predict`'s blocking readback (`ds41_dense.cpp:1391`, `ggml_backend_tensor_get`) is the third ggml
   sync/layer and exists only to feed `tier.prefetch`.  `finish(l)` and `attn_router(l+1)` are on the same
   stream, so no host sync is needed between them: issue the readback async, run `attn_router(l+1)`, then sync
   ONCE for both.  Trade-off to measure: the prefetch then overlaps `tier(l+1)` instead of `attn(l+1)`.
   Control experiment: `pf_b=0` (plain `finish()`, no predict graph, no readback).

### Slice C - move the selection on-device (kills S1 and the 12.21 ms bubble; unlocks real pipelining)
A small kernel that reads the router ids straight from `im.o_ids`, looks up a device-side
`(tier, device-pointer)` table (built once per token from `slot_of`/the cache), and writes the metadata block
for `native_expert_grouped` **in place**.  Then the host never needs the ids: it only ENQUEUES
`attn_router` -> tier -> `finish_predict` for every layer and reads the result at the end, running tens of layers
ahead of the device.  S1 (the last unavoidable host wait) disappears, the bubble disappears, and the copies
overlap by construction.  Prerequisite: slice B's fixed metadata layout.  This is the one that changes the
architecture; A and B are the steps that pay for themselves on the way.

## 3b. Slice A - implementation gotchas found by reading the code (READ BEFORE EDITING)

- **`gp.s_cp` does NOT exist on the decode path.**  It is created only in the prefill/chunk path
  (`ds41_moe.cpp:2834`) and every user guards on it (`2708`, `2733`).  Slice A needs a stream created next to
  `gp.s` in the context init (`ds41_moe.cpp:1194`, where `gp.s`/`gp.s_pf` are made) plus its destroy in the same
  place as theirs, not a reuse of `s_cp`.
- **The arena/pool question is ANSWERED - the ordering is simpler than it first looks.**  The CPU pool's reader
  threads write only arena **victim** slots, and a victim is only ever an arena expert with `last_use < im.tick`,
  i.e. NOT routed by this call (`ds41_moe.cpp:2174-2175`); the PCIe DMAs read THIS call's arena experts, which all
  have `last_use >= im.tick`, so **source and victim sets are disjoint by construction**.  Arena slots are
  indexed per `(layer, expert)` (`NXA`-strided `slot_of`), and a layer is visited once per token, so there is no
  cross-layer aliasing either.  The host also joins the reader threads before any DMA is issued
  (`for (auto& t : rd) t.join();`, `2198`), so "writes before the async copy call" holds in program order.
  Precedent in-tree for the pattern that IS needed: `if (direct && gp.s_pf) cudaStreamSynchronize(gp.s_pf), "pf
  sync"` at `2167` (no prefetch DMA may read a slot the readers are about to overwrite).
- **So the change is:** issue the `pcie dma` copies (`2068/2073/2079`) plus the `lru stage copy` D2D (`2070`) on
  the new copy stream, `cudaEventRecord` on it, then a single `cudaStreamWaitEvent(gp.s, ev)` immediately before
  `launch_group(1, np, ...)` at `2086` - one wait covers the whole group, and the kernels keep running on gp.s.
  `gp.s_cp`-style discipline: nothing else may be issued on the copy stream within the layer without its own
  ordering.
- d_stage regions are disjoint per group (pcie at `[0, np)`, file at `[np, np+nfg)`, stride `B`), so the pcie and
  file copies can share the one copy stream; the file group's copies are issued after its reader threads join
  (`2198`), so they order after the reads for free.
- Ship it behind an env flag (`STRATA_DMA_OVERLAP=1`), default OFF, until the nsys overlap metric
  (max concurrent kernels > 1) and forced-token identity both hold across several runs.  The forced-token check
  is the correctness bar but it is a RACE-check, not a proof: run it several times, and read the overlap number
  as the real evidence that the copies moved off the compute path.
- Expected size: the pcie DMA is ~3.3 ms/token and the file DMA ~1 ms (scout's miss math), so overlapping them
  with the resident-expert compute is worth ~4 ms/token - real but BELOW the 5 ms noise floor of tok/s, which is
  why the metric must be the nsys overlap count, not throughput.  (The scan's 11-14 ms ceiling assumed
  overlapping the copies with the attention too, which needs slice C first.)

## 4. How to measure (mandatory - the noise floor will lie to you)

With IDENTICAL work (same tokens, same 59511 VRAM hits / 255 cpu) the tier phase swung **17.63 - 23.06 ms (30%)**
between runs on a loaded box, against 16.5-17.2 on a quiet one.  A ~2 ms host-time effect is invisible in tok/s.

1. Correctness bar is always **forced-token identity**: `STRATA_FORCE_IDS=<file>`, then `diff` the emitted tokens
   against the forcing file - 0 differing lines.  Any kernel/launch change must pass this before its speed is
   discussed.
2. Speed for host-call work is measured by **nsys call COUNTS** (`nsys profile -t cuda,osrt
   --cuda-graph-trace=graph`), not tok/s: count `cudaLaunchKernel` / `cudaMemcpyAsync` / `cudaStreamSynchronize`
   per token and the **max concurrent kernels** (the overlap metric).  A helper exists at
   `~/AI/v41-dev/nsys/idle_attrib.py`.
3. CPU-heavy work (builds, torch, the local analysers, the HF upload's hashing) must NOT run beside a timed
   decode run - it is what produced the 30% swing.
4. The gate for a numerics-affecting change is still `test_ds41_moe_gpu` (worst rel ~1.44e-2) plus ppl/KL on
   code/agent/reason against the saved `dq/*_q8.logits` if the change can move bits.
