#!/usr/bin/env bash
# local/mem-watchdog.sh [floor_mb] - kill ds41_generate if the box's available memory drops below the floor.
#
# The engine runs in its own cgroup (bench-v41.sh caps it), but a *global* OOM can still fire when the rest of
# the box (desktop, browser, zram churn) is already at zero free pages: the kernel's OOM killer then takes the
# engine AND whatever else it likes.  This watchdog makes the engine the first thing to go instead: it samples
# MemAvailable and kills the engine while there is still room to do so cleanly.
#
# Usage: local/mem-watchdog.sh 2000 &   (floor in MiB)
set -uo pipefail
FLOOR="${1:-2000}"
echo "[watchdog] floor ${FLOOR} MiB, every 2 s"
while :; do
    av=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    if [ "$av" -lt "$FLOOR" ]; then
        echo "[watchdog] MemAvailable ${av} MiB < ${FLOOR} MiB - stopping the engine"
        pkill -TERM -f ds41_generate 2>/dev/null
        sleep 3
        pkill -KILL -f ds41_generate 2>/dev/null
        echo "[watchdog] done"
        exit 0
    fi
    sleep 2
done
