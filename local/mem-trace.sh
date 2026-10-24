#!/usr/bin/env bash
# local/mem-trace.sh OUT [seconds] - one line a second: what the box's memory is actually doing.
# columns: time avail_mb anon_mb shmem_mb unevict_mb cached_mb slab_mb cg_current_mb cg_anon_mb cg_file_mb
OUT="${1:?usage: mem-trace.sh OUT [seconds]}"
N="${2:-600}"
cg() { cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/run-*/memory.current 2>/dev/null | sort -rn | head -1; }
cgstat() { awk -v k="$1" '$1==k {print int($2/1048576); exit}' /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/run-*/memory.stat 2>/dev/null | head -1; }
echo "# t avail anon shmem unevict cached slab cg_cur cg_anon cg_file" > "$OUT"
for _ in $(seq 1 "$N"); do
    read -r avail < <(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    read -r anon  < <(awk '/^AnonPages:/   {print int($2/1024)}' /proc/meminfo)
    read -r shm   < <(awk '/^Shmem:/      {print int($2/1024)}' /proc/meminfo)
    read -r unev  < <(awk '/^Unevictable:/{print int($2/1024)}' /proc/meminfo)
    read -r cach  < <(awk '/^Cached:/     {print int($2/1024)}' /proc/meminfo)
    read -r slab  < <(awk '/^Slab:/       {print int($2/1024)}' /proc/meminfo)
    echo "$(date +%H:%M:%S) ${avail:-0} ${anon:-0} ${shm:-0} ${unev:-0} ${cach:-0} ${slab:-0} $(cg) $(cgstat anon) $(cgstat file)" >> "$OUT"
    sleep 1
done
