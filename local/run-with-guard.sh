#!/usr/bin/env bash
# local/run-with-guard.sh CMD... - run a command, but kill it if the box runs out of memory.
#
# Why: this box has 31.7 GiB of RAM and the desktop apps use ~8 GiB of it.  A big host allocation (the V4.1
# engine's expert arena) plus the model's page cache can push the box into the swap, and systemd-oomd (which
# runs with the defaults: SwapUsedLimit 90%, DefaultMemoryPressureLimit 60%) then kills processes - including
# the session's.  This guard watches MemAvailable and stops the job before that happens.
#
#   local/run-with-guard.sh <min available MiB> CMD...
#
# The job's own output goes to stdout/stderr unchanged.
set -uo pipefail

MIN_MIB="${1:?usage: run-with-guard.sh <min available MiB> CMD...}"
shift

"$@" &
PID=$!
echo "guard: pid $PID, killing it if MemAvailable drops below ${MIN_MIB} MiB" >&2

while kill -0 "$PID" 2>/dev/null; do
    avail=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
    if [ "${avail:-0}" -lt "$MIN_MIB" ]; then
        echo "guard: MemAvailable ${avail} MiB < ${MIN_MIB} MiB - stopping pid $PID" >&2
        kill -9 "$PID" 2>/dev/null
        wait "$PID" 2>/dev/null
        exit 99
    fi
    sleep 5
done
wait "$PID"
exit $?
