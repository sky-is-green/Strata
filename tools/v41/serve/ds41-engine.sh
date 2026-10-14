#!/usr/bin/env bash
# ds41-engine.sh - the DeepSeek-V4.1 engine as Strata's server starts it (`<exe> --serve <args>`), with memguard's
# protections in the FOREGROUND (memguard.sh backgrounds its command, which would break the stdin/stdout protocol):
# ComfyUI check, the shared GPU lock (held while the server runs), an 84 GiB RAM cap, no swap.  (= GLM's glm-engine.sh)
set -u
cd "$(dirname "$0")/../../.."
if [ "${MEMGUARD_ALLOW_COMFY:-0}" != 1 ] && ss -ltn 2>/dev/null | grep -q ':8188 '; then
    echo "ds41-engine: refusing - ComfyUI is running on :8188 (RAM)" >&2; exit 3
fi
LOCK="${DS4_LOCK:-$HOME/.quetza-data/conductor/ds4-gpu.lock}"
export GGML_NO_BACKTRACE=1
# 4 KiB pages pin the 60 GiB arena at 1.2 GB/s (load ~50 s); THP pins at ~0.16 GB/s (~4 min)
export STRATA_NO_LARGEPAGES="${STRATA_NO_LARGEPAGES:-1}"
# prompt tokens route exactly (Mal 2026-10-12): turns <= 96 new tokens run as EXACT passes (no route bias / file
# penalty; ~6-7 tok/s, beats the ~18 s batch floor below ~110 tokens: 41 tokens 6.9 s vs 18.0 s, KL vs batch 0.012),
# longer turns as the exact batch.  The biased quick path (KL 0.10-0.15, ~40-57 tok/s):
# STRATA_SERVE_STEP_MAX=768 STRATA_SERVE_PASS_BIAS=1
export STRATA_SERVE_STEP_MAX="${STRATA_SERVE_STEP_MAX:-96}"
export STRATA_SERVE_PASS_BIAS="${STRATA_SERVE_PASS_BIAS:-0}"
# DSpark speculative decode in --serve (Mal's go 2026-10-14; exact speculative sampling, the output distribution is the
# trunk's): the Q2_K sidecar, the draft's experts pruned to the 64 most used per layer (dspark_expert_counts.txt),
# depth 4 with the confidence-head trim at 0.6 (s11: +30% avg, prose 9.5 -> 17.3 tok/s).  STRATA_SERVE_DSPARK= (empty)
# turns it off.  The engine args (strata-ds41.json): route bias x0.5 + file 0.3, --arena-adapt, arena 64 GiB, --kv-fp4.
DSPARK_SIDECAR=/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.Q2_K.gguf
export STRATA_SERVE_DSPARK="${STRATA_SERVE_DSPARK-$DSPARK_SIDECAR}"
export STRATA_DSPARK_KEEP="${STRATA_DSPARK_KEEP:-$PWD/tools/v41/serve/dspark_expert_counts.txt:64}"
export STRATA_DSPARK_DEPTH="${STRATA_DSPARK_DEPTH:-4}"
export STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.6}"
[ -z "$STRATA_SERVE_DSPARK" ] && unset STRATA_SERVE_DSPARK
# A layer's file-tier "wave" reader: 64 threads x 16 parts (the code default is 32 x 8, `STRATA_WAVE_THREADS` /
# `STRATA_WAVE_PARTS` in ds41_moe.cpp).  The 512K turn bench (s12 run3, one run vs two baselines of the same
# config): fresh 8000-token batch 29.56 s (dense 10.15 + experts 19.33) vs 31.51 / 31.98 s (experts 20.38 / 20.71)
# = -7%/-8%; +61 / +273 / +1001 turns and the decodes after them unchanged inside the bench's 1.5-4.3% spread.
# Queue depth is the lever the microbench points at (rb: 8 outstanding reads 4.72 GB/s, 16 -> 5.22; rb2: 32 workers
# x 8 parts = 256 outstanding -> 6.22 GB/s).  Only the big-read turn showed it, so RE-MEASURE with >= 3 runs before
# trusting it for other turn sizes; unset both to go back to 32 x 8.
export STRATA_WAVE_THREADS="${STRATA_WAVE_THREADS:-64}"
export STRATA_WAVE_PARTS="${STRATA_WAVE_PARTS:-16}"
exec flock "$LOCK" systemd-run --user --scope -q --unit="ds41-serve-$$" -p MemoryMax=84G -p MemorySwapMax=0 \
    "$PWD/build-ds4-gpu/ds41_generate" "$@"
