#!/usr/bin/env bash
# local/power-tune.sh on|off|show - keep the NVMe and CPU cores awake during decode.
#
# DeepSeek-V4.1 decode reads an expert from the NVMe every 10-20 ms; between reads
# the drive (APST) and the CPU cores (C3) fall asleep and each read costs ~2x.
# Keeping them awake gives +11% on short answers and +21% at ~25K context (DS41Engine).
#
# Requires root. Install once with: sudo local/install-power-tune.sh
# ./run.sh switches it on at start and off when the server ends (if installed).
set -uo pipefail

ACTION="${1:-show}"

case "$ACTION" in
  on)
    # CPU governor -> performance
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -w "$g" ] && echo performance > "$g" 2>/dev/null || true
    done
    # Disable NVMe APST (autonomous power state transitions) on every NVMe
    for n in /dev/nvme*n1; do
        [ -e "$n" ] && nvme set-feature "$n" -f 0x0c -v 0 2>/dev/null || true
    done
    # Disable deep C-states
    for s in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
        [ -w "$s" ] && echo 1 > "$s" 2>/dev/null || true
    done
    echo "power-tune: on"
    ;;
  off)
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -w "$g" ] && echo schedutil > "$g" 2>/dev/null || true
    done
    for n in /dev/nvme*n1; do
        [ -e "$n" ] && nvme set-feature "$n" -f 0x0c -v 0x0f 2>/dev/null || true
    done
    for s in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
        [ -w "$s" ] && echo 0 > "$s" 2>/dev/null || true
    done
    echo "power-tune: off"
    ;;
  show)
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "nvme devices: $(ls /dev/nvme*n1 2>/dev/null | tr '\n' ' ')"
    ;;
  *)
    echo "usage: $0 on|off|show" >&2; exit 2 ;;
esac