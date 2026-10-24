#!/usr/bin/env bash
# local/dense-trace.sh [short_tokens] [long_tokens] [prompt.i32]
#
# Where a token's GPU time goes, from rocprofv3 kernel traces.  Two runs of the same prompt, one decoding a few
# tokens and one decoding more; the difference is the decode window, which is then broken down by kernel and by
# family (local/trace-analyze.py).  Bash-prefix any env var to vary the config under test, e.g.
#   STRATA_DENSE_REQUANT=Q8_0 local/dense-trace.sh
#
# rocprofv3 is NOT installed on this box.  It is extracted from the CachyOS `rocprofiler 7.2.4-2.1` package under
# /tmp/opencode/rocprof and run from there: no root, no package manager, nothing on the system changes.  To rebuild
# that tree (7 MB of downloads):
#   mkdir -p /tmp/opencode/rocprof/root && cd /tmp/opencode/rocprof
#   for p in hsa-amd-aqlprofile rocprofiler; do
#       curl -sfLO "$(pacman -Sp "$p" | head -1)"; bsdtar -xf "${p}"-*.pkg.tar.zst -C root; done
# The extracted copy ships rocprofv3 as well as the older v2; the traces below use v3.
#
# Two things the output must not be read as:
#   * the row count is graph nodes, not host launches - this build has GGML_CUDA_GRAPHS=ON and the graphs are
#     captured (see STRATA_LOG_DEBUG=1), so "4,164 dispatches" is not "4,164 launches";
#   * tracing slows the engine ~20% and widens the gaps between rows, so *when* things run is unreliable here.
#     Kernel durations and counts are trustworthy; phase times come from STRATA_DEC_PROF instead.
set -uo pipefail
cd "$(dirname "$0")/.."

SHORT=${1:-1}
LONG=${2:-5}
PROMPT=${3:-local/prompts/code-review.i32}
RP=${ROCPROFV3:-/tmp/opencode/rocprof/root/opt/rocm/bin/rocprofv3}
OUT=${TRACE_DIR:-/tmp/opencode/v41}

[ -x "$RP" ] || { echo "no rocprofv3 at $RP - see the header for how to extract it" >&2; exit 1; }
[ -f "$PROMPT" ] || { echo "no prompt at $PROMPT" >&2; exit 1; }
export LD_LIBRARY_PATH="$(dirname "$RP")/../lib:${LD_LIBRARY_PATH:-}"

( setsid nohup bash local/mem-guard.sh 8000 4000 > "$OUT/trace-guard.log" 2>&1 < /dev/null & )
sleep 1

for n in "$SHORT" "$LONG"; do
    d="$OUT/trace$n"
    rm -rf "$d"; mkdir -p "$d"
    HIP_VISIBLE_DEVICES=1 \
    STRATA_DENSE_REQUANT="${STRATA_DENSE_REQUANT:-q4_K}" \
    STRATA_ADVISE_AHEAD="${STRATA_ADVISE_AHEAD:-6}" \
    STRATA_DECODE_DIRECT="${STRATA_DECODE_DIRECT:-0}" \
    timeout 900 "$RP" --kernel-trace --stats -f csv -d "$d" -- \
        ./build-hip-pre/ds41_generate -m models/V41-GSQ/V41-GSQ.gguf --backend cuda --experts gpu \
            --rank models/V41-GSQ/rank.bin --pf-b "${STRATA_PF_B:-0}" --arena-gib 4 \
            --engram models/engram/layers14:models/engram/layers1 \
            --ids-file "$PROMPT" -n "$n" --ctx 32768 --prompt-batch 1 \
        > "$d/engine.log" 2>&1
    echo "-n $n: $(grep -aE '^decode:' "$d/engine.log" | tail -1)"
done

A=$(find "$OUT/trace$SHORT" -name '*_kernel_trace.csv' | head -1)
B=$(find "$OUT/trace$LONG" -name '*_kernel_trace.csv' | head -1)
[ -n "$A" ] && [ -n "$B" ] || { echo "no kernel trace CSV produced" >&2; exit 1; }
python3 local/trace-analyze.py "$A" "$B" $(( LONG - SHORT ))
