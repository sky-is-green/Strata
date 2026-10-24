#!/usr/bin/env bash
# local/bench-v41.sh <label> [extra engine flags...] - one decode measurement of the V4.1 GSQ on this box.
#
# The same flags every time, so the runs are comparable:
#   * the model on the native NVMe (the file tier reads 1.38 MB expert blobs at random; the USB enclosure
#     the user mounted is 45 MB/s and cannot serve that),
#   * --arena-gib 4: this box has 31.7 GiB of RAM and ~8.5 GiB of it is the desktop, so the arena has to
#     leave room for the page cache and for everything else (a bigger arena OOM'd the box twice),
#   * a memory-capped cgroup, so the kernel's OOM killer can only reach this job, never the desktop.
#
# Usage:
#   local/bench-v41.sh baseline                     one card (the dense half and every expert on card 1)
#   local/bench-v41.sh split --layer-split 20 --split-device 1
set -uo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:?usage: bench-v41.sh <label> [engine flags...]}"
shift

OUT="/tmp/opencode/v41/${LABEL}.log"
mkdir -p /tmp/opencode/v41

echo "== $LABEL: $(date +%H:%M:%S)" | tee "$OUT"
# the measured-best streaming config on this box (both overridable): the kernel-side read-ahead
# needs buffered decode reads, and prefetch (pf-b) is off because there is no arena to stage into
STRATA_DEC_PROF=1 STRATA_NO_LARGEPAGES=1 \
STRATA_DENSE_REQUANT="${STRATA_DENSE_REQUANT:-q4_K}" \
STRATA_ADVISE_AHEAD="${STRATA_ADVISE_AHEAD:-6}" STRATA_ADVISE_PREV="${STRATA_ADVISE_PREV:-1}" \
STRATA_DECODE_DIRECT="${STRATA_DECODE_DIRECT:-0}" \
    systemd-run --user --scope --quiet \
        -p MemoryMax=12G -p MemoryHigh=9G -p MemorySwapMax=0 \
        ./build-hip-pre/ds41_generate \
            -m models/V41-GSQ/V41-GSQ.gguf \
            --backend cuda --experts gpu \
            --arena-gib 4 --threads 12 \
            --rank models/V41-GSQ/rank.bin \
            --pf-b 0 \
            --engram models/engram/layers14:models/engram/layers1 \
            --prompt-batch 1 \
            --ids-file local/prompts/code-review.i32 \
            -n 64 --ctx 32768 \
            "$@" 2>&1 | tee -a "$OUT"

echo "== $LABEL: the numbers" | tee -a "$OUT"
grep -E "tok/s|decode step|tier:" "$OUT" | tail -6 | tee -a "$OUT"
