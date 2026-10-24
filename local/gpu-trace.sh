#!/usr/bin/env bash
# local/gpu-trace.sh OUT [seconds] - per second: system memory plus each card's VRAM and GTT, and the engine cgroup.
# The GTT column is the one that matters: amdgpu backs "VRAM" allocations with system RAM when a card
# over-commits, and those pages are charged to the driver, not to the process, so a cgroup cap cannot see them.
OUT="${1:?usage: gpu-trace.sh OUT [seconds]}"
N="${2:-600}"
cg() { local v; v=$(cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/run-*/memory.current 2>/dev/null | sort -rn | head -1); echo "${v:-0}"; }
echo "# t avail anon shmem cached vram0 gtt0 vram1 gtt1 cg_cur" > "$OUT"
for _ in $(seq 1 "$N"); do
    v0=$(cat /sys/class/drm/card0/device/mem_info_vram_used 2>/dev/null); g0=$(cat /sys/class/drm/card0/device/mem_info_gtt_used 2>/dev/null)
    v1=$(cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null); g1=$(cat /sys/class/drm/card1/device/mem_info_gtt_used 2>/dev/null)
    printf "%s %s %s %s %s %s %s %s %s %s\n" \
        "$(date +%H:%M:%S)" \
        "$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)" \
        "$(awk '/^AnonPages:/{print int($2/1024)}' /proc/meminfo)" \
        "$(awk '/^Shmem:/{print int($2/1024)}' /proc/meminfo)" \
        "$(awk '/^Cached:/{print int($2/1024)}' /proc/meminfo)" \
        "$(( ${v0:-0} / 1048576 ))" "$(( ${g0:-0} / 1048576 ))" \
        "$(( ${v1:-0} / 1048576 ))" "$(( ${g1:-0} / 1048576 ))" \
        "$(( $(cg 2>/dev/null || echo 0) / 1048576 ))" >> "$OUT"
    sleep 1
done
