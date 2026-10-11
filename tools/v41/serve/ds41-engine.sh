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
exec flock "$LOCK" systemd-run --user --scope -q --unit="ds41-serve-$$" -p MemoryMax=84G -p MemorySwapMax=0 \
    "$PWD/build-ds4-gpu/ds41_generate" "$@"
