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
