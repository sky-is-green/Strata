#!/usr/bin/env bash
# local/move-model-to-samsung.sh - copy the GSQ from the work drive to the new Samsung NVMe, verify it,
# point the repo at the copy, and only then delete the source.
#
# Why the Samsung: it is a dedicated XFS volume (no btrfs CoW, no shared space with the repos), 234G for a
# 158G model, and it frees the work drive for an Engram table (the work drive is at 87% full, which btrfs
# does not like).
set -uo pipefail

SRC=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/models-nvme/V41-GSQ
DST=/run/media/penis/82c9af76-d211-4d94-bfc9-7bb7e047fe91/V41-GSQ
REPO=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/strata-ds41
WANT_SIZE=162957934592
WANT_SHA=3897de644b77410e2d353ce9e1fc3f0bb1894d674f3db97374efdf7c155b33dd

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "copying the model ($SRC -> $DST)"
mkdir -p "$DST"
if ! cp -a "$SRC/V41-GSQ.gguf" "$SRC/rank.bin" "$SRC/V41-GSQ.gguf.experts.json" "$SRC/V41-MTP-Vision.Q2_K.gguf" "$DST/"; then
    log "copy FAILED"
    exit 1
fi
cp -a "$SRC/tokenizer" "$DST/" || { log "tokenizer copy failed"; exit 1; }
sync
log "copy done:"
ls -la "$DST/"
df -h "$DST" | tail -1

log "checking the size"
got_size=$(stat -c %s "$DST/V41-GSQ.gguf")
if [ "$got_size" != "$WANT_SIZE" ]; then
    log "SIZE MISMATCH: want $WANT_SIZE got $got_size"
    exit 1
fi
log "size ok: $got_size"

log "checking the sha256 (this reads the whole 163 GB back off the Samsung)"
got_sha=$(sha256sum "$DST/V41-GSQ.gguf" | awk '{print $1}')
if [ "$got_sha" != "$WANT_SHA" ]; then
    log "SHA256 MISMATCH: want $WANT_SHA got $got_sha"
    exit 1
fi
log "sha256 ok: $got_sha"

log "pointing models/V41-GSQ at the Samsung"
ln -sfn "$DST" "$REPO/models/V41-GSQ"
ls -la "$REPO/models/V41-GSQ"

log "deleting the work-drive copy (the Samsung is verified; the USB keeps the backup)"
rm -f "$SRC/V41-GSQ.gguf"
# the small files stay in the old dir in case the repo still has a path to it; the gguf is the big one
df -h /run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad | tail -1
log "done"
