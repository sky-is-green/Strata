#!/usr/bin/env bash
# local/install-ais-module.sh - install the AIS-enabled amdgpu module (built from the 7.3.0-4-cachyos source).
#
# Run as root.  Three modes:
#
#   stage1    install the module and boot it with AIS DISABLED.  If the desktop comes up, the module itself is
#             sound (same source, same config, only the AIS code added) and any later failure is AIS's fault.
#   stage2    enable AIS (drop the modprobe.d override) and rebuild the initramfs, then reboot.
#   rollback  remove the module and its override, restore the stock module, rebuild the initramfs.
#
# The stock module is never touched: ours goes to /lib/modules/<rel>/updates/, which depmod puts ahead of
# kernel/, so deleting one file reverts everything.
set -euo pipefail

KREL=7.3.0-4-cachyos
SRC=/run/media/penis/e66ad938-d8e1-4f48-8c5c-d14aa1337864/kbuild/cachyos-7.3.0-4/drivers/gpu/drm/amd/amdgpu/amdgpu.ko
UPD=/lib/modules/$KREL/updates
OVR=/etc/modprobe.d/99-ais-stage1.conf
MODE="${1:-}"

need_root() { [ "$(id -u)" = 0 ] || { echo "run this with sudo" >&2; exit 1; }; }

# This box boots with Limine and has no /etc/mkinitcpio.d presets, so `mkinitcpio -P` errors out.
# limine-mkinitcpio generates the initramfs with explicit arguments and updates the Limine entries.
rebuild_initramfs() {
    if command -v limine-mkinitcpio >/dev/null 2>&1; then
        limine-mkinitcpio
    else
        mkinitcpio -P
    fi
}

case "$MODE" in
stage1)
    need_root
    [ -f "$SRC" ] || { echo "missing module: $SRC" >&2; exit 1; }
    modinfo "$SRC" | grep -q "^vermagic: *$KREL" || { echo "vermagic mismatch - refusing" >&2; exit 1; }
    install -d "$UPD"
    zstd -19 -T0 -q -f "$SRC" -o "$UPD/amdgpu.ko.zst"
    echo "options amdgpu ais_disabled=1" > "$OVR"
    depmod -a "$KREL"
    rebuild_initramfs
    echo
    echo "stage 1 installed ($(du -h "$UPD/amdgpu.ko.zst" | cut -f1) module, AIS disabled)."
    echo "Reboot, then check:"
    echo "    modinfo -n amdgpu | grep updates     # ours is the one loaded"
    echo "    systemctl is-active graphical.target  # desktop came up"
    echo "    grep -c kfd_ais /proc/kallsyms        # symbols present, AIS still off"
    echo "then: sudo $0 stage2"
    ;;
stage2)
    need_root
    rm -f "$OVR"
    depmod -a "$KREL"
    rebuild_initramfs
    echo
    echo "AIS enabled. Reboot, then check:"
    echo "    grep kfd_ais /proc/kallsyms | head"
    echo "    /tmp/opencode/hipFile/tools/ais-check/ais-check   # expect True / True / True"
    ;;
rollback)
    need_root
    rm -f "$UPD/amdgpu.ko.zst" "$OVR"
    depmod -a "$KREL"
    rebuild_initramfs
    echo "rolled back to the stock module"
    ;;
*)
    sed -n '2,18p' "$0"; exit 2 ;;
esac
