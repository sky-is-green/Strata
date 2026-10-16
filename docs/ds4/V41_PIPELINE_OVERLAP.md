# V4.1 decode: the per-layer handshake (work in progress, branch `v41-pipeline-overlap`)

Status: **branch open, no engine change yet.** This document is the plan and the gate. `main` stays the working
configuration; nothing here is measured yet, and nothing gets merged back until the token-identity gate below passes
and the phase timer shows a gain.

## The measured problem

`attn_router` is 43.35 ms of a 72.2 ms token at 4K state and 57.66 ms of 75.2 ms at 64K (`STRATA_DEC_PROF`, 64 steps,
one-shot). Inside that phase, from `rocprofv3 --runtime-trace --kernel-trace` on a 17K-state decode (32 tokens):

| | per token |
| --- | ---: |
| wall | 139 ms (profiled; ~75 unprofiled) |
| GPU kernels | 32.2 ms (23%) in 4,238 launches |
| outside the kernels | 106.9 ms (77%) |
| `hipStreamSynchronize` | **215 calls, 105.0 ms** |

And the syncs, attributed by the call immediately before them:

| call before the sync | per token | what it is |
| --- | ---: | --- |
| `hipMemcpyAsync` | 95.3 | device->host readback, ~2.4 per layer |
| `hipGraphLaunch` | 76.7 | dense (and router) graph launched, then synchronized, ~1.9 per layer |
| `hipGetLastError` | 42.5 | launch -> error query -> sync |

Two micro-optimizations were tried and **reverted** because they won nothing: making the 685 per-token error queries
optional (< 0.3 ms/token) and caching `ggml_cuda_get_device` (120 calls/token, 0.06 ms/token). The cost is the waits,
not the calls.

## What is actually serialized, and what can move

Per layer the chain is: dense(l) -> router ids(l) -> tier(l) -> merge -> dense(l+1). That dependency is real: the
tier cannot start before the routing is known, and the next dense cannot start before the merge. So the fix is not
"overlap the layers" in general; it is to remove the *number of host round trips per layer*:

1. **Two graph launches and two syncs per layer (76.7/token).** The dense half and the router look like separate
   graph computes, each followed by a synchronize. If both live in one captured graph, or if the router's top-k is
   computed on the device and only its 6 indices per layer are read, one of the two syncs per layer disappears.
2. **2.4 device->host readbacks per layer (95.3/token).** Candidates seen in the code: the routed ids
   (`ds41_moe.cpp` `gpu_run`, the `ids6` readback), the hidden state for the CPU pool (`:2220`, gated by `nk > 0`, so
   not ours), the tier's parts (`:2763`, only on the host-sum path), the state lengths. Packing what is genuinely
   needed for a layer into **one** readback would take ~2.4 syncs down to 1.
3. **The per-layer `layer sync` is already gone** on this path (`layer_synced` false for `run_to_dev`,
   `ds41_moe.cpp:2682-2691`), so it is not a target: the author measured it at 0.12 ms x 40.
4. **The `sum sync` (`:2761`) is on our path** (inside the `out_dev` branch) and is one of the ~40/token.

Realistic ceiling: if the per-layer handshake goes from ~5.4 syncs to ~2, that is ~3.4 x 40 = ~136 syncs/token x
~0.49 ms = a large part of the 105 ms/token measured under the profiler, i.e. plausibly **10-25% of the real token**.
That is an estimate, not a promise; the phase timer will say.


## Step 1 done: every readback in the decode path, and its consumer (corrected)

`grep cudaMemcpyDeviceToHost ds41_moe.cpp` in the tier gives four sites, and on this path (`run_to_dev`, `0 cpu` rows)
**none of them runs per layer**: `"x d2h"` (:2220) is gated by `nk > 0` (the CPU pool, idle here), `"d2h parts"`
(:2764, :2978, :3455) belongs to the host-sum / chunk / multi paths, `"lru demote"` moves an evicted expert's bytes,
`"out d2h"`/`"sal d2h"` are the chunk path.

The `x_state` mirror is **not** per layer either: `Ds41Dense::read_state` / `read_state_cols` are called from
`ds41_generate.cpp:587` (once a token), `:724` (once a pass) and `:949` (the chunk path). My first reading had this
wrong and the correction matters.

So the readbacks that actually repeat per layer are **the router's two**, inside the layer body:

| readback | per token | consumer | candidate |
| --- | ---: | --- | --- |
| `im.predict_out[il] -> buf` (the router's output) | ~40 | the host picks the top-6 experts | merge with the row below |
| `o -> top_ids` (the routed ids) | ~40 | the tier's dispatch | one packed readback (ids + weights, small) instead of a full row |
| `im.x_state -> x` (state mirror) | 1 / token (`:587`) | the tier's activation and the checkpoint | not a per-layer target |
| `im.x_state -> x` (`read_state_cols`) | 1 / pass (`:724`) | the verify window | not a per-layer target |
| `lg -> out` (logits) | 1 | sampling | needed |
| `im.fpredict_out[next_il] -> buf` | ~40 | the *next* layer's prediction (prefetch) | could be deferred past the current layer's tier |

80 of the 95.3 readbacks a token are those two per layer, and each `ggml_backend_tensor_get` on the HIP backend does a
`cudaMemcpyAsync` **and a synchronize** - exactly the "D2H then sync" the trace attributed 95.3 syncs a token to.
The first code step of the branch is therefore: let the device pick the top-6 (or read back the row and the ids in
**one** packed copy) so a layer costs one round trip instead of two. That is 40 syncs a token, reversible, and under
the token gate.


## Step 1b: what `predict()` really does (corrects the readback picture again)

`Ds41Dense::predict(il, top_ids, n_top)` (`ds41_dense.cpp:1469`) builds a **per-layer predict graph**, runs
`im.compute(graph, true)` - the `true` synchronizes - and then reads **16 ints**:
`ggml_backend_tensor_get(im.predict_out[il], buf, 0, sizeof(buf))`. So a layer's routing costs one graph + one sync +
one *small* readback, not one full row per readback: the "merge the two readbacks" candidate (step 2 of the plan) is
already minimal in the code and is not a lever.

Where the per-layer syncs come from, then, is the **number of graph computes**: the decode loop calls `predict()` for

- the current layer's router (`ds41_generate.cpp:538`) - its ids are needed before the tier of that layer, so this
  sync is real;
- `predict(l + k, ...)` / `predict(l + advise_ahead, ...)` (`:523`, `:532`) - these only feed the file prefetcher
  (`tier.advise_file`), i.e. a *hint*;
- `predict_multi` (`:656`, `:680`) for the verify window / D3.

Measured `hipGraphLaunch` is 76.7 per token, about 1.9 per layer, which matches dense graph + predict graph. So the
first real code step is **not** merging readbacks: it is deferring the *advise* predicts, whose ids are a hint and
whose sync the engine does not need to take before the layer's own work. Plan step order updated: (1) defer the advise
predicts, (2) then look at capturing the dense and predict graphs together, (3) then the two-layers-in-flight idea.


## Step 2: the predict graphs are 18 ms of `attn_router`, and the prefetch is worth more than they cost

Measured on the same decode (96 tokens), one-shot, this branch (no engine change yet):

| `--pf-b` | begin | predict | attn_router | tier | finish | decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (default) | 1.68 | 0.36 | **57.53** | 14.51 | 5.51 | 12.80 tok/s |
| 0 | 6.28 | **0.00** | **39.65** | **44.87** | 3.02 | 11.28 tok/s |

So the per-layer prefetch prediction (`dense.predict(l) -> tier.prefetch(l)`, `ds41_generate.cpp:538`, gated by
`--pf-b`) costs **~18 ms of a 57.5 ms phase, ~0.45 ms per layer** - a graph compute plus a synchronize plus a 16-int
readback. That is the per-layer handshake, measured at the phase level and consistent with the trace (76.7 graph
launches and 95.3 readbacks a token). Turning it off removes all of it and then the tier pays 30 ms more, because
the misses are served on demand instead of under the previous layer's attention: the prefetch is worth more than its
own cost, and `--pf-b 0` is **worse** overall (11.28 against 12.80 tok/s).

`file_pred` (which would run `kFP = 4` predictions per layer) is off by default and `advise_ahead` is 0, so on this
path exactly one predict per layer runs, for the prefetch.

**What this means for the plan.** The predict cannot be dropped; it has to get cheaper, and the only way that does
not lose the prefetch is to fold the prediction into the graph that is already being launched - the previous layer's
dense graph, whose output the router head would consume, so that a layer costs one graph, one sync and one readback
instead of two of each. That is a change in `ds41_dense.cpp` (`build_predict` plus the dense graph builder), i.e. the
"single capture dense + router" item, now promoted from step 4 to the next step.


## Step 3: the dense+router fusion already exists; the 18 ms is the prefetch's own host work

Reading the code settles it, and it moves the target:

- `build_predict`'s own comment (`ds41_dense.cpp:839`) says the finish graph already computes "finish (n = 1) + the
  next layer's routing prediction on the state it writes", and the decode loop consumes that prediction through
  `have_pred` (`ds41_generate.cpp:518`, set at `:582`, used at `:536`). So the "single capture dense + router" that
  the plan had as its big item is **already how this engine works**: one dense graph per layer emits the next layer's
  routing prediction, and the explicit `dense.predict(l)` only runs when no prediction is available (`:538`).
- The `hipGraphLaunch` count (76.7/token) is therefore not 40 predicts plus 40 dense graphs. The predicts are rare.
- What `--pf-b 0` actually removes is the **`tier.prefetch(l, top16, 16)` call** (`:537`, `:540`), i.e. 40 calls a
  token of host-side prefetch planning plus the DMA issue. That is the ~18 ms (0.45 ms a layer), and it buys 30 ms
  of tier time: worth it, but pure host work on the critical path.

**Next step, revised**: make the prefetch plan cheaper or less on the critical path - fewer ids, a cheaper residency
filter, or the plan built while the previous layer's DMA is in flight. This is a `ds41_moe.cpp` change and it is
*safer than the graph work*: the prefetch is a hint, so the tokens must stay identical (the experts that get computed
do not change, only what is warm), and the gate should pass trivially - which makes it a good first real edit on this
branch, with the phase timer as the judge (`attn_router` down, `tier` flat).


## Step 4: the prefetch budget is the lever, and it is already at its optimum

`--pf-b` is the prefetch budget in **experts per layer** (dithered; the engine's default is 1.43, `ds41_moe.hpp:284`).
Measured on the same decode, 96 tokens, one run each (phase timer):

| `--pf-b` | attn_router | tier | decode |
| ---: | ---: | ---: | ---: |
| 0 (off, step 2) | 39.65 | 44.87 | 11.28 |
| 0.5 | 67.38 | 23.15 | 10.44 |
| 1.0 | 67.28 | 19.19 | 11.19 |
| 1.43 (default) | ~43-44 | ~15-19 | 13.4-13.7 |
| **2.0** | 44.92 | 15.78 | **13.90** |
| 3.0 | 47.13 | 18.38 | 12.94 |
| 4.0 | 46.24 | 16.96 | 13.55 |

The shape is what matters, not the last digit: there is a **cliff below ~1.0 experts a layer** (attn_router jumps
from ~45 to ~67 ms and the token rate falls by a fifth) because the misses stop being asynchronous prefetches and
become synchronous on-demand fetches, and above ~1.5 the curve is flat within the noise. The author's 1.43 sits just
above the cliff, and 2.0 is not measurably better with one run a side (13.90 against 13.4-13.7 is inside the spread).

So the ~18 ms that the prefetch costs in `attn_router` is not waste: it is ~1.4-2 experts a layer x 7.75 MB x 40 =
**~450-620 MB per token of PCIe traffic, overlapped with the layer's attention**, and it buys 30 ms of tier time. The
handshake is doing useful work; the only way to shrink it is to need less data, which is the arena/cache size (75 GiB
here, the largest safe value measured) or a smaller expert format - not a scheduling trick.

**Where the cantiere stands.** Of the four items in the plan, three turned out to be already solved in the engine
(the state mirror is not per-layer, the router readback is already one small buffer, and the dense+router fusion is
exactly how `finish_predict` works), the fourth (deferring the prediction) does not pay because the prefetch is worth
more than its sync, and the prefetch's own cost is at its tuned optimum. What is left is the one thing the plan put
last: **two layers in flight**, i.e. giving the tier of layer l and the dense of layer l+1 something to overlap -
which the data dependence (routing needs the dense output, the next dense needs the merge) makes a redesign rather
than a patch.


## Step 5: why the chain cannot be overlapped further (the honest closure)

The per-layer chain is `dense(l) -> routing(l) -> tier(l) -> merge -> dense(l+1)`. Every arrow is a real data
dependency: the routing is computed from the dense output, the tier needs the routing, the merge needs the tier, and
the next dense needs the merge. "Two layers in flight" therefore does not mean running two layers at once; it can
only mean overlapping the *host work* of layer l+1 with the GPU work of layer l - and that requires knowing layer
l+1's routing, which requires its dense, which requires layer l's merge. The only way to break it is to *compute* with
predicted routing instead of real routing, which changes the model (the prefetch already uses predictions, but only to
warm the cache, never to compute).

What *is* overlappable, the engine already overlaps: the experts' DMA run under the layer's attention (the comment on
`tier.prefetch`), the next layer's arena experts are DMA'd while the current dense runs (`STRATA_PRESTAGE_GIB`), and
the file tier's reads are staged a layer ahead (`chunk_pf_prepare`).

And the 105 ms of sync time is not idle host time: it decomposes into real work that the host is waiting for -

| component | per token |
| --- | ---: |
| GPU kernels | 32.2 ms |
| PCIe DMA of the prefetched experts (~450-620 MB) | ~19 ms |
| launch latency (4,238 launches at the ~5.6 us/physical measured on gfx1100) | ~24 ms |
| the rest (copies, graph launches, readbacks) | the remainder |

So the handshake is not a scheduling bug to fix: the host waits because the GPU is doing kernels, moving experts over
PCIe, and paying per-launch cost. The only lever of the three that has room is the **number of launches**, which is a
kernel-fusion problem (4,238 a token: 676 elementwise, 479 quant/dequant, 925 copies/fill, 520 launches in the dense
and tier), not a pipeline problem. That is where this cantiere hands back: the pipeline is not the bottleneck, the
kernel count is.

## The gate (non negotiable)

- Every step runs the same three prompts (short, 1.5K, 63K) with `-n 96` and `STRATA_DEC_PROF=1`, greedy, and compares
  the generated ids against `main` byte for byte. A single differing id stops the change: this code path decides the
  routing, so a numeric difference would not be a rounding difference, it would be a different model.
- Measure `attn_router`, `tier` and the token rate, three runs a side, alternated.
- One sync removed per commit. If a step does not win outside the noise, it is reverted with its numbers in this file.

## Order of work

1. Re-read `gpu_run`/`run_to_dev` in `ds41_moe.cpp` and list every host readback and every synchronize in the decode
   path with its consumer (which host decision actually needs the bytes). Deliverable: a table in this file.
2. Merge the per-layer readbacks that have no consumer (e.g. counters or bookkeeping that could stay on the device).
3. Look at whether the dense graph and the router graph can be one capture (that one touches ggml's graph handling,
   so it comes after the engine-side wins).
4. Only if 2 and 3 hold the gate: consider two layers in flight (the routing of layer l+1 needs dense(l+1), which
   needs the merge of l, so this is a *shorter* pipeline than it sounds - measure before building).

## Baseline to compare against (same binary, `main`, this machine)

```
4K state:   begin 4.08  predict 0.46  attn_router 43.35  tier 18.68  finish 5.61  = 72.2 ms -> 13.67 tok/s
64K state:  begin 0.50  predict 0.40  attn_router 57.66  tier 11.43  finish 5.20  = 75.2 ms -> 14.00 tok/s
17K state (profiled): kernels 32.2 ms/token, syncs 215/token = 105 ms/token
```
