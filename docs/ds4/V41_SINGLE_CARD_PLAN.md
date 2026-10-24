# V4.1 single-card lane — handoff plan

This document is the whole brief for one lane: **make DeepSeek-V4.1-Flash decode faster on one RX 7900 XT**,
without changing what it computes. It assumes no prior conversation. The two-card split and the AIS/hipFile port
are deliberately out of scope here (see §6).

## 1. Where the work lives

| | |
|---|---|
| repo | `~/Desktop/work/strata-ds41` (symlink to a worktree of `github.com/sky-is-green/Strata`) |
| branch | `ds41-merge` |
| build | `build-hip-pre` — the fused build (`-DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON`) |
| engine | `tools/v41/engine/ds41_generate` + `ds41_dense.cpp` + `ds41_moe.cpp` |
| model | `models/V41-GSQ` → the 151.8 GB GSQ on the Crucial P310 NVMe |
| engram | `models/engram/layers14` + `models/engram/layers1` |
| rank | `models/V41-GSQ/rank.bin` |
| bench | `local/bench-v41.sh <label> [extra flags]` → log in `/tmp/opencode/v41/<label>.log` |
| quality gate | `local/ppl-gate.sh "" q4_K` (corpus `local/corpora/code-5389.i32`) |
| profiler | `local/dense-trace.sh` + `local/trace-analyze.py` (rocprofv3, see §7) |
| guard | `local/mem-guard.sh 8000 4000` — run every engine under it |

## 2. The measured baseline (2026-10-24)

Config (all in `local/bench-v41.sh`; do not change it casually):

```
STRATA_DENSE_REQUANT=q4_K  STRATA_ADVISE_AHEAD=6  STRATA_ADVISE_PREV=1  STRATA_DECODE_DIRECT=0
--pf-b 0 --arena-gib 4 --prompt-batch 1 --ctx 32768   (40 layers, fused build)
```

Three 200-token runs, alternating: **23.80 / 25.36 / 24.82 tok/s → mean 24.66**, ~41.6 ms/token.

| phase | ms/token | notes |
|---|---:|---|
| attn_router | 25.7 | attention + shared expert + router + HC; ~17.3 ms of it is kernels, so **~8 ms is host** |
| tier | 11.6 | 5.4 ms kernels + 1.3 copies + **~5 ms host/file** (file wait ~1.5 ms) |
| predict | 0.2 | after `STRATA_ADVISE_PREV=1` (was 4-5 ms as the ahead-predictor) |
| begin + finish | 0.5 + 1.5 | |
| **total** | **41.6** | kernel-busy ~24 ms |

Kernel breakdown (rocprofv3, 4-token window): GEMV 10.09 ms (321 calls, Q4_K part is 4.02 GB in 7.03 ms =
572 GB/s), tier fused experts 5.36, elementwise/norm/rope 3.47, PCIe expert copies 1.34 (777 copies/token),
hyper-connections 1.23, attention 0.83, quantize/swiglu 0.66, top-k/argsort 0.48.

Quality (do not regress): q4_K **PPL 2.9084**, Q8_0 2.8739, composite 2.8712 on the 5,389-token corpus at
`--prompt-batch 512`. At `--prompt-batch 1` the same corpus gives 5.6463 for both arms of the routing change —
batch size changes the bits, so **compare arms at the same batch, never across batches**.

## 3. The work, in order

### 3.1 The dense loop's ~8 ms of host time (biggest single item)

Evidence: `attn_router` is 25.7 ms wall against ~17.3 ms of kernels. Hypotheses to check, in order:

1. **How many graph submits and synchronizes per layer?** The loop builds several small graphs per layer
   (`ggml_new_graph_custom`, 64/128/8192 nodes) and calls `ggml_backend_graph_compute` once per graph; one
   `ggml_backend_synchronize` per layer is structural (the router's top-k must reach the host to pick experts).
   Count them, then time them.
2. **Can adjacent graphs merge?** e.g. the layer's norm/rope/HC ops into the attention graph. The engine already
   fuses `finish` + `predict`, so the pattern exists.
3. **Host copies**: `set_columns`, `read_state`/`write_state`, the routed tensor, and the `ADVISE_PREV` id copies.
   Measure; 40 layers × a few hundred µs is the whole budget.

Start with an instrumented build under a new opt-in env (e.g. `STRATA_DEC_PROF=2` printing submits/syncs and
their milliseconds per token). **Do not guess the fix before the count exists.**

### 3.2 The tier's ~5 ms of host/file time

- File wait is ~1.5 ms/token with `ADVISE_PREV`; the rest is expert selection, cache bookkeeping and copies.
- There are **777 `copyBuffer` calls per token** (~4 per loaded expert). Try one copy per expert (or a single 3D
  copy), measure the call count and `tier` time. A correct change here must leave the expert counts
  (`47204 hit / 556 pcie / 556 file` for 200 tokens) and the tokens identical.

### 3.3 Prefill (optional, but it is the practical long-context limit)

75 tokens prefill at ~18 tok/s; a 5,389-token prompt at ~20.7 tok/s, so long prompts cost minutes before decode
starts. Ideas: sweep `--prompt-batch` (512 / 1024 / 2048), check whether the prompt path uses the MMQ/fused class
passes, and whether the per-chunk file staging (`STRATA_PRESTAGE_GIB`) is helping it. This lane is not required to
touch prefill; it is listed because it is the user-visible wall for contexts over ~8K.

## 4. Protocol (non-negotiable)

- **One change at a time.** New features are opt-in; the default path must stay byte-identical.
- **Measure with the phase timers**, 3 × 200 tokens, alternating arms, report every run and the mean. Run-to-run
  noise is ~4%; a change that does not beat it is a revert, not a "maybe".
- **Quality gate before any default changes**: `local/ppl-gate.sh "" q4_K` (same `--prompt-batch` as the baseline).
  If the change is not supposed to alter math, also show tokens identical on a fixed prompt.
- **Always run under the guard**: `local/mem-guard.sh 8000 4000`. This box has 31.7 GiB of RAM and the amdgpu GTT
  budget is invisible to cgroups; the box has died twice from GTT, not from the process heap. Never run two engines
  concurrently.
- **Commit with the numbers** (before/after phase table, tok/s, PPL delta) and push to `ds41-merge`.

## 5. Acceptance for this lane

- Baseline reproduces at 24.7 ± 1 tok/s.
- A merged change: **≥ 26 tok/s mean** (41.6 → ≤ 38 ms) with PPL unchanged, or a clearly measured phase reduction
  with tokens identical.
- A short report in `docs/ds4/V41_AMD_SPLIT.md` (append a dated section), plus a commit on `ds41-merge`.

Stretch: ≤ 36 ms (≈ 27.8 tok/s) — the remaining host time after both 3.1 and 3.2.

## 6. Parked — do not redo these

| idea | why it is parked |
|---|---|
| two-card `--layer-split` | two-stage load hit 17.6 GiB of GTT (guard stops at 4); when it did run it was 420 ms/token. Layers are sequential, so it cannot double decode anyway. |
| two V4.1 processes (one per card) | ~14 GiB GTT before the first token; 5 GiB RAM left. Does not fit. |
| AIS / hipFile wiring | file wait is ~1.5 ms/token; the ceiling is ~4%. The module is live and banked. |
| `--pf-b` > 0 | on this box the prefetcher's guesses are outside the 4 GiB arena and cost DMA for nothing. |
| disabling CUDA graphs | re-measured: a wash now that the predictor graphs are gone. |
| changing q4_K / fused-op defaults | both pass the gate; the engine default stays OFF, the lane config enables them. |

## 7. Reproduction recipes

```bash
cd ~/Desktop/work/strata-ds41

# baseline (writes /tmp/opencode/v41/<label>.log)
local/bench-v41.sh base

# quality gate (prints PPL + top-1; baseline q4_K 2.9084)
local/ppl-gate.sh "" q4_K

# profile a token (needs the extracted rocprofv3; see the script header to rebuild it)
STRATA_DENSE_REQUANT=q4_K local/dense-trace.sh
```

The engine directly, for one arm:

```bash
HIP_VISIBLE_DEVICES=1 STRATA_DENSE_REQUANT=q4_K STRATA_ADVISE_AHEAD=6 STRATA_ADVISE_PREV=1 \
STRATA_DECODE_DIRECT=0 STRATA_DEC_PROF=1 \
./build-hip-pre/ds41_generate -m models/V41-GSQ/V41-GSQ.gguf --backend cuda --experts gpu \
  --rank models/V41-GSQ/rank.bin --pf-b 0 --arena-gib 4 \
  --engram models/engram/layers14:models/engram/layers1 \
  --ids-file local/prompts/code-review.i32 -n 200 --ctx 32768 --prompt-batch 1
```

## 8. Traps

- **`mem-guard.sh` uses `pkill -f ds41_generate`.** A shell whose own command line contains that string will be
  killed too. Run the engine from a script file (e.g. `local/bench-v41.sh`), not from an inline command string.
- **`/tmp` is tmpfs (16 GB).** Logs and traces there consume RAM; delete kernel traces after analysis
  (`local/dense-trace.sh` leaves CSVs under `/tmp/opencode/v41/trace*`).
- **`build-hip` is a stale Oct-21 binary.** The lane uses `build-hip-pre`; `local/ds41-hivebench.json` now points
  there too.
- **The model is a symlink.** Do not copy it; a 151.8 GB copy will fill the disk and must not be moved back to the
  chipset SSD.
- **`--prompt-batch` changes the bits.** Any A/B or PPL comparison must hold it fixed.
- **Engram tables are on two disks** (L1 on the work drive, L14 on the MX100). If either disappears the engine
  warns about missing `--engram` dirs; check the symlinks in `models/engram/`.
- **The 2nd card is idle in this lane.** That is expected; two-card work is a different lane.

## 9. First three experiments

1. Instrument the layer loop (`STRATA_DEC_PROF=2`): submits, syncs, and host-copy milliseconds per token.
2. Merge the graphs the count shows are mergeable; measure with tokens identical.
3. Collapse the tier's 4 copies/expert to 1; measure call count and `tier` ms, expert stats unchanged.
