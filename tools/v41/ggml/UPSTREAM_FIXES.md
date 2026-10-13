# Upstream llama.cpp fixes ported to our vendored ggml (PREPARED, not applied)

Patch: `tools/v41/ggml/upstream-fixes.patch` — `-p1` relative to `third_party/llama.cpp`.
Source PRs (ggml-org/llama.cpp):
- **#30103** "CUDA/HIP: fix race in flash_attn_ext_f16_process_tile when nbatch_combine != DKQ/2"
  merge `1bb2b9fcbe85c35c4e3b787e1c868ef05ca702f3` (2026-10-10); 1 file, +1/-1.
- **#29953** "CUDA: fix MMQ out-of-bounds reads"
  merge `fc9ce6b9d52a8504edcb262abc92737c2289f96c` (2026-10-08); `mmq.cu` +54/-16, `mmq.cuh` +11/-41.

Our vendored tree (`third_party/llama.cpp`, no `.git`) predates BOTH: it still has
`ggml_cuda_mmq_get_J_max` / `mmq_args::ncols_opt` (removed by #29953) and the bare
`if (np > 1)` sync (changed by #30103). It also predates the `prec_src1` refactor
(`ggml_cuda_mmq_get_config` takes no `prec_src1`), so the #29953 hunks are ported
against our API — the `prec_src1` argument is dropped from the J_best loop. Our local
`dedup_bcast` (gate/up broadcast) path in `mmq.cu` is untouched.

## #30103 — flash-attn tile race
Bug: at the end of the VKQ combine loop the barrier guarding the shared tile buffer was
`if (np > 1)`. When a config has `nbatch_combine != DKQ/2`, the combine writes only part
of the DV buffer per pass and warps can still race on `tile_Q` even with `np == 1` →
a thread reads a half-written tile.
Our path: **D=512 prompt (prefill) flash-attn**, fp16 KV, MMA kernel `fattn-mma-f16.cuh`.
Every DKQ=512 config in our table has `nbatch_combine` 64 or 128 (never DKQ/2 = 256), so
the race is live for all of them. Fix: `if (np > 1 || nbatch_combine != DKQ/2)`.
**md5 drift: PLAUSIBLE primary cause.** A missing barrier in the combine is a genuine
read-after-write race on shared memory — output varies run-to-run at fixed input, exactly
the prefill md5 drift signature.
Rebuild: `fattn.cu.o` + all `template-instances/fattn-mma-f16-instance-*.cu.o` (~22 objects).

## #29953 — MMQ out-of-bounds reads
Bug: MMQ tile loads read up to `J-1` columns past the end of `src1_q8_1` / `ids_dst`. The
old buffers were sized with `ggml_cuda_mmq_get_J_max(...)*sizeof(block_q8_1_mmq)` (a J-size
guess) and `ids_dst` had zero slack. The fix computes `J_best` (and `nthreads_best`) once on
the host, derives an explicit `src1_q8_1_padding`, pads `ids_dst` by `J_best-1`, and drops
the J_max guess. It also caps `J <= 128` in the config macro and de-inverts
`mul_mat_q_case`'s fallback selection (readability — same semantics).
Our path: every CUDA MMQ matmul — our **q8_0, q5_K, q4_K and MXFP4** dense/MoE products.
**md5 drift: WEAKER.** These are OOB *reads* (usually stale scratch, not a value that
changes every run); our `mmq-deterministic.patch` already fixed the stream-k split, which is
the drift cause we saw. Still worth taking: it removes reads of uninitialised scratch and
makes `J_best` deterministic.
Rebuild: `mmq.cu.o` plus every TU that includes `mmq.cuh` — `quantize.cu.o`, `mmvq.cu.o`.

## Residuals
- Not applied — dry-run only. No build, no GPU, nothing under `third_party/` touched.
- Stacking: apply AFTER `dsv41-fuse.patch` (already applied) and alongside
  `mmq-deterministic.patch` (already applied). No shared files with fuse/fake-quant/rank-argsort;
  shares only `mmq.cuh` with mmq-deterministic (separate region — both dry-run clean).
- Verify after applying: rebuild + forced-token md5 A/B, prefill D=512, run-to-run.
