#!/usr/bin/env bash
# local/ppl-gate.sh [corpus.i32] [type...] - the quality gate for a change to the dense half.
#
# DeepInfer's AMD notes set the convention: a change to a kernel should show up as a PPL delta rather than as silent
# drift.  Our corpus is 5,389 tokens of the engine's own source (local/corpora/code-5389.i32), built with
# local/encode_prompt.py over tools/v41/engine/ds41_dense.cpp.  358-token corpora cannot resolve the difference -
# that is how an earlier "fused ops cost 4.5%" reading turned out to be noise (the real cost is 0.09%).
#
# Baselines measured 2026-10-24 on this box (default flags below):
#   fused ops,  Q8_0 dense : PPL 2.8739  top-1 0.755
#   composite,  Q8_0 dense : PPL 2.8712  top-1 0.755     <- the unfused reference
#   fused ops,  q4_K dense : PPL 2.9084  top-1 0.753     <- +1.2% for +28% decode
#
# Usage:
#   local/ppl-gate.sh                               # the baseline arm (q4_K, as the configs now stand)
#   local/ppl-gate.sh "" Q8_0 q4_K                  # several dense types
#   STRATA_DENSE_REQUANT=q6_K local/ppl-gate.sh     # one explicit arm
#
# Every arm must use identical flags: --prompt-batch selects a different prefill path and the paths do not produce
# identical bits, so comparing across batch sizes is meaningless.
set -uo pipefail
cd "$(dirname "$0")/.."

CORPUS="${1:-local/corpora/code-5389.i32}"
shift || true
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=("${STRATA_DENSE_REQUANT:-q4_K}")

[ -f "$CORPUS" ] || { echo "missing corpus: $CORPUS (see the header)" >&2; exit 1; }

mkdir -p /tmp/opencode/v41
( setsid nohup bash local/mem-guard.sh 8000 4000 > /tmp/opencode/v41/ppl-gate-guard.log 2>&1 < /dev/null & )
sleep 1

printf "%-8s %-10s %s\n" "dense" "PPL" "top-1"
for t in "${ARMS[@]}"; do
    out=$(HIP_VISIBLE_DEVICES=1 STRATA_DENSE_REQUANT="$t" \
        ./build-hip-pre/ds41_generate \
            -m models/V41-GSQ/V41-GSQ.gguf \
            --backend cuda --experts gpu \
            --rank models/V41-GSQ/rank.bin \
            --pf-b 0 --arena-gib 4 \
            --engram models/engram/layers14:models/engram/layers1 \
            --ids-file "$CORPUS" --ppl -n 1 --ctx 32768 --prompt-batch 512 2>&1 |
        grep -aE "^PPL")
    if [ -z "$out" ]; then
        printf "%-8s FAILED\n" "$t"
    else
        printf "%-8s %-10s %s\n" "$t" \
            "$(echo "$out" | sed -E 's/^PPL ([0-9.]+).*/\1/')" \
            "$(echo "$out" | sed -E 's/.*top-1 ([0-9.]+).*/\1/')"
    fi
done
