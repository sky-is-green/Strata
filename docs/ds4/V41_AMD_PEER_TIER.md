# V4.1 on two GPUs: a peer expert tier (design)

Status: **design written, not implemented.** The measurements below are what sizes it; the single-GPU numbers are in
`AMD_HIP_NOTES.md`.

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

## Order of work

1. `--peer-device N` + the `Peer` struct + activation/metadata staging, taking the `cpu_keep` rows for the **decode**
   path only (`gpu_run`, one token); verify identical tokens; measure.
2. The verify path (`run_multi`/`run_chunk`, i.e. DSpark's passes) - the same producer, `kMaxParts` rows per pass.
3. The peer's own expert cache + adaptation; measure the extra hit rate.
4. Only if it pays: the prompt path.
