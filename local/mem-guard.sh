#!/usr/bin/env bash
# local/mem-guard.sh [avail_floor_mb] [gtt_ceiling_mb] - stop the engine before the *system* runs out.
#
# Why this exists: on this box the engine's cgroup cap does not cover everything.  amdgpu can back a
# "VRAM" allocation with system RAM (GTT), and GTT has its own budget (mem_info_gtt_total = 29.4 GB here,
# ~93% of RAM).  Those pages are charged to the driver, not to the process, so MemoryMax never sees them
# and the kernel's global OOM killer takes whatever it likes (it has killed the desktop, Steam and Discord).
#
# So this watches the two numbers that actually move first:
#   * MemAvailable  - the symptom (with a high floor, because the collapse is fast: 26 GB -> 156 MB in 13 s)
#   * GTT used      - the mechanism (a driver spill shows up here before the page accounting does)
# and kills only the engine.  Sampling is 0.3 s because 2 s was too slow to win the race last time.
set -uo pipefail
FLOOR="${1:-6000}"     # MiB available
GTT_CEIL="${2:-6000}"  # MiB of GTT across all cards
echo "[guard] avail floor ${FLOOR} MiB, gtt ceiling ${GTT_CEIL} MiB, every 0.3 s"

gtt_mb() {
    local t=0 v
    for f in /sys/class/drm/card*/device/mem_info_gtt_used; do
        v=$(cat "$f" 2>/dev/null) || continue
        t=$(( t + v / 1048576 ))
    done
    echo "$t"
}

killed=0
while :; do
    av=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    gtt=$(gtt_mb)
    if [ "$av" -lt "$FLOOR" ] || [ "$gtt" -gt "$GTT_CEIL" ]; then
        echo "[guard] avail ${av} MiB (floor ${FLOOR}), gtt ${gtt} MiB (ceiling ${GTT_CEIL}) - stopping the engine"
        pkill -TERM -f ds41_generate 2>/dev/null
        sleep 2
        pkill -KILL -f ds41_generate 2>/dev/null
        killed=1
        break
    fi
    # keep the peak on record
    echo "$(date +%H:%M:%S) $av $gtt" >> /tmp/opencode/v41/guard.log
    sleep 0.3
done
echo "[guard] exit ($killed)"
