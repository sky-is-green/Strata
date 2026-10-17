# DeepSeek-V4.1 merge: state and first-light recipe

Written 2026-10-17, after the merge was committed and built. The branch is parked here on purpose:
the 35B (MTP lane) and FlashNext-176B (8081) are the working models; the V4.1 552B GSQ fetch is
deferred until a job actually needs it. Nothing below is speculative — every step is either done
(with the commit) or a listed command for later.

## State (done)

- Branch `ds41-merge` in `work/strata-ds41` (a worktree of `work/strata`), commits:
  - `ab2f75e3` — the real merge of the amd-gfx1100 fork tip `5310be69` (`dsv41-fetch`) into ours
    `2dd1d679` over base `18bce49f`: 90 true conflicts, resolved ours-superset / union, with the
    dialect splits listed below. Theirs' deletion of 849 bench/result + data-bin files reverted
    in the same commit. Not pushed (house rule).
  - `273d81f8` — `ptq_tq = 1` at the three `tools/v41/engine/ds41_moe.cpp` layout fan-outs
    (`ds41_moe_blob/class/layer_layout`): every DSV4.1 expert `NativeFmt` reads the llama-tq
    PTQ1_0 dot (Q8_0 activations). Scion's own PTQ1_0 CPU dot stays reachable at `ptq_tq = 0`.
- Dialect decisions that matter if anyone touches PTQ1_0 (GGUF type 143, 28 B / 128 trits, both
  forks implement it):
  - CPU dispatch is keyed on `NativeFmt::ptq_tq` (0 = Scion dialect, 1 = llama-tq/DSV4.1); the dot
    implementations coexist.
  - GPU 143 is the fork's: `fork/fork_fmt.cuh`'s `Fmt<143>` + `fork_dq.cuh`'s `dq_ptq1_0` serve it
    through the grouped-kernel switches via the `_FMTS_FORK` lists. Ours' groups do not list 143,
    ours' `Fmt<143>`/`vec_dot_ptq1_0_q8_1` were yielded (ours' `vec_dot_ptq1_0_q8_1_multi` kept —
    it is ours' Scion-dialect CPU-side consumer, not the GPU dot).
  - The vendored ggml owns the `QK_PTQ1_0` macro name — the fork's own constant is
    `STRATA_PTQ1_BLOCK` in `fork_vecdot.cuh`.
- Build, measured on this box (2× RX 7900 XT, gfx1100, `-j6` while the 176B serve was up):
  configure with `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_NATIVE_EXPERTS=ON
  -DSTRATA_GGML_DIR=.../work/strata/third_party/llama.cpp`; `strata` `--version` = 0.1.44,
  full `all` target green including `ds41_generate`, `test_ds41_moe`, `ds4_moe_replay`.
  One HIP port outside the engine: `tools/ds4/moe_gap.cu` needs `LANGUAGE HIP` plus a
  clock64-based ns fallback (gfx1100 has no `s_memrealtime`/`%globaltimer` asm; `STRATA_GAP_TCLK`,
  default 2.208 GHz, converts ns to cycles).

## First light (when there is a job for it)

1. Fetch to the HDD lane dir (network 48 MB/s single stream, aria2c ×8 ≈ 140 MiB/s): the official
   `V41-GSQ.gguf`, 152 GiB; later the two official Engram shards 47/48, 94.6 GiB each, one at a
   time. Disk is fine after the 2026-10-17 cleanup (331 GiB free on the work drive).
2. Pause the 176B on 8081 for the run window only (14 GiB RAM headroom otherwise; build/serve
   discipline applies).
3. Run `ds41_generate` (or the serve path once wired) with the GSQ shards; arena 12–16 GiB;
   record decode/prefill against the known 176B numbers above.
4. If the CPU dots disagree with the GPU path on ternary tiers, suspect the `ptq_tq` keying first
   (per-expert classes can mix non-143 gu formats with 143 down tiers); the parity test to grow is
   `test_ds41_moe` / `ds4_expert_parity` with a real-shard fixture.
