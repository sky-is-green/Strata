# V4.1 on two GPUs: a peer expert tier - implemented, measured, and NOT worth it

Status: **implemented and measured, negative result.** The code is not in the tree; it is saved as
`local/peer-tier-attempt.patch` (901 lines) if the situation ever changes. The measurements that size it, and the one
that kills it, are below; the single-GPU numbers are in `AMD_HIP_NOTES.md`.

## The result

| | decode tok/s (3 reps, round robin, no DSpark) | prompt tok/s |
| --- | --- | --- |
| without `--peer-device` | 12.6 [12.6 7.9 13.9] | 14.2 |
| with `--peer-device 1` | 11.8 [12.7 11.4 11.8] | 15.7 |

The implementation was correct: the peer took **91.5 routed rows per token** (~35% of the 258 rows a token routes)
off card 0 (its hits went 10959 -> 6713, every other counter identical), the tokens were **identical** with the peer
on and off, and with `STRATA_PEER_CHECK=1` (card 0 also computes the peer's rows and compares) **4299/4299 rows were
bit-identical** - the kernels and the q8_1 activations really are the same on both cards.

It still lost ~5%, because of where the time is:

```
STRATA_DEC_PROF=1, 64 steps, no DSpark, ms/token:
begin 5.49 · predict 0.38 · attn_router 51.70 · tier 19.44 · finish 5.77   (12.1 tok/s)
```

The expert tier is **23%** of a decode step; the dense half (`attn_router`) is **63%**. A peer that removes 35% of
the tier removes ~3% of the step and pays a per-layer hand-off (activation upload, metadata, an event and a wait,
per-row copies) on ~40 layers a token. No amount of store tuning changes that arithmetic.

## Why

The decode budget on this box (one RX 7900 XTX, 96-token reply, DSpark, `pass_v` split over 41 verify passes of 7.05
tokens): **dense 99.9 ms, tier 365.1 ms** per pass - the expert tier is ~78%. Inside the tier, for 96 tokens:
21336 VRAM hits, 2827 PCIe misses (~29/token), 443 file-tier reads (~4.6/token), `file 1110 ms` (11.6 ms/token).
The remaining ~17 ms/token of the tier is the DMA + kernel time of those misses, on the critical path of a layer,
plus ~12 ms/token of resident-expert kernels.

A second card cannot make the *dense* half faster (it fits one card and is 22% of the time). It can take the miss
work: read the blob into pinned host memory (already done today), DMA it to the **peer**, run the same grouped
kernel there, and hand back only that expert's row (`n_embd` floats). That overlaps the miss path with the primary's
resident-expert kernels instead of serialising it, and the peer's 24 GB can additionally hold a cache of the
most-routed non-resident experts, removing part of the read entirely.

Ceiling: if the whole miss path (file + DMA + kernel, ~28 ms/token) came off the critical path, decode would go from
~20 tok/s to ~30-35 tok/s at the same hit rate. The extra cache is the +10-20% measured by shrinking `--slots`
(see the notes); the overlap is the rest.

## What the code already gives us

`ds41_moe.cpp`'s decode path already mixes **two producers** of expert rows and one summer:

- the primary's resident experts: `native_expert_grouped` on `gp.s` into `gp.d_parts` (one row per routing index);
- the CPU share (`--pcie`): the `ExpertPool` computes rows, they are copied into pinned `gp.h_parts` and then
  `cudaMemcpyAsync` into `gp.d_parts` at their routing index (line ~2744);
- `strata::kernels::decode_rows_sum(gp.d_parts, weights, K, H, ...)` sums them into the layer's output (line ~2759).

So a peer is a third producer that deposits rows in `gp.d_parts` the same way the CPU does: **no change to the sum,
the weights, the routing or the dense half**. That is the whole reason this is a day of work and not a rewrite.

The miss plan for a layer is built around lines 2500-2540 (`fg_k` = file-tier experts computed on the card,
`cpu_keep` = the ones the CPU pool takes) and the peer takes rows out of `cpu_keep` first (they are the ones read
from the file tier; the arena-held ones are cheap already).

## Design

New config (`Ds41MoeConfig::peer_device`, `--peer-device N`, `STRATA_PEER_DEVICE=N`, default -1 = off):

```cpp
struct Peer {                      // one per process, only when peer_device >= 0
    int dev = -1;
    cudaStream_t s = nullptr;
    cudaEvent_t  done = nullptr;   // recorded on the peer after its last D2H for a layer
    void *d_parts = nullptr;       // kMaxParts * H floats, the peer's own rows
    void *d_x = nullptr, *d_xq = nullptr, *d_scratch = nullptr;   // activations (F32 + q8_1) + kernel scratch
    void *h_parts = nullptr;       // pinned landing pad (kMaxParts * H floats)
    std::vector<strata::kernels::NativeExpertLayout> gll;          // per layer, as the primary's gp.gll
    uint8_t *h_meta = nullptr; void *d_meta = nullptr;             // the same per-class metadata blocks
    size_t meta_bytes = 0;
};
```

Per layer, in the decode path, after the primary's resident launches are issued on `gp.s` and the blobs for the
peer's experts are in pinned host memory (`h_fstage`, already filled by the reader threads at line 2531):

1. `cudaSetDevice(peer.dev)`; copy the layer's metadata block (contents changed by `set_meta`, host-side) and each
   blob into the peer's staging (`cudaMemcpyAsync` on `peer.s`), then the activations (`d_xq`, `d_x`);
2. launch `native_expert_grouped(peer.gll[layer], ...)` on `peer.s` with the peer's group/entry metadata;
3. `cudaMemcpyAsync(peer.h_parts, peer.d_parts, nrows*H*4, D2H, peer.s)`, `cudaEventRecord(peer.done, peer.s)`;
4. on the primary's stream: `cudaStreamWaitEvent(gp.s, peer.done)`, then `cudaMemcpyAsync(gp.d_parts + idx*H,
   peer.h_parts + j*H, ...)` per row (or one strided copy) and let `decode_rows_sum` see them - the primary's
   `decode_rows_sum` launch already waits for its own stream, so nothing else changes;
5. `cudaSetDevice(0)`.

A peer *cache* (its own `strata::core::ExpertCache` over its VRAM, seeded from `rank.bin` and adapted by routing
counts, exactly like the primary's) turns step 1's blob copy into a hit: the blob never leaves the peer and only the
row comes back. The `ExpertCache` class and the `D3`-style adaptation the primary already uses are per-device
objects, so this reuses code rather than inventing a protocol.

## Risks and how they are checked

- **Streams/events on the wrong device**: every `cudaSetDevice` is paired; the C++ RAII in the current code has no
  device guard, so the new block must be the only place that switches. Verified by a debug print of
  `cudaGetDevice()` around the block and by the correctness gate below.
- **Correctness**: an expert must give the same row on either card (same kernels, same q8_1 activations). The gate is
  the one used all along: the same prompt must produce **identical tokens** with `--peer-device 1` and without it
  (greedy, no DSpark first, then with DSpark). Any drift shows up there.
- **VRAM budget**: the dense half + 14 GiB of primary slots + the peer's staging/cache must fit the two cards'
  24 GB each; the peer's cache is sized from *its* free VRAM (a second `ggml_backend`-style query with
  `cudaSetDevice(1)`), the same rule the primary uses.
- **PCIe topology**: the two cards are on PCIe Gen4 x16 slots; a peer-to-peer copy is *not* assumed - the design
  goes through pinned host memory (one D2H 20 KB/expert, one H2D), which costs ~2x20 KB per expert at ~10 GB/s, i.e.
  microseconds, and works on any topology. `cudaMemcpyPeerAsync` can replace it later if it turns out to be a win.

## What was left out of the attempt

The peer dispatch was wired into the decode path (`gpu_run`) only: the DSpark verify path (`gpu_run_n_ex`) and the
prompt chunk path (`run_chunk`) never call it (so the DSpark A/B in `local/bench_peer_dspark2` is machine noise - the
peer computes 0 rows there), and the store is not re-split after a `--serve` re-seed. None of that would change the
verdict: the tier is not the majority of the step.

## When it would be worth doing

When the dense half stops being 63% of the step. Today's decode at 256K is dense-bound (see the phase profile in
`AMD_HIP_NOTES.md`, and the context-size dependence: 12.7 / 14.1 / 9.8 tok/s at ctx 4K / 32K / 256K with identical
tokens). Make the dense half 3x faster and the tier becomes the majority again - then this patch, which is correct
and measured, is the next thing to pick up.
