#!/usr/bin/env bash
# the rest of the V4.1 fetch, chained: the join finishes -> its sha256 is checked -> the Engram shard downloads
# resume -> the two Engram tables are extracted onto the SSDs and the merged index is written.
#
# The join and the downloads share one slow FUSE volume, so the downloads are held (SIGSTOP) while the join runs.
set -uo pipefail

SSD="/run/media/penis/e66ad938-d8e1-4f48-8c5c-d14aa1337864/v41/V41-GSQ"
HDD="/run/media/penis/30CE2C97CE2C577E/storage/v41"
WORK="/run/media/penis/30CE2C97CE2C577E/storage/v41-work"
L1="/var/tmp/v41-engram/layers1"
L14="/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/v41-engram/layers14"
REPO="/home/penis/Desktop/work/strata-ds41"
WANT=162957934592

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for the join to finish ($WANT bytes)"
while :; do
    s=$(stat -c %s "$SSD/V41-GSQ.gguf" 2>/dev/null || echo 0)
    [ "$s" -ge "$WANT" ] && break
    sleep 30
done
log "the join is done: $s bytes"

log "checking the joined file's sha256 against SHA256SUMS"
want=$(awk '$2 == "V41-GSQ.gguf" {print $1}' "$HDD/V41-GSQ/SHA256SUMS")
got=$(sha256sum "$SSD/V41-GSQ.gguf" | awk '{print $1}')
if [ "$want" != "$got" ]; then
    log "SHA256 MISMATCH: want $want got $got"
    exit 1
fi
log "sha256 ok: $got"

log "resuming the Engram shard downloads"
for p in $(pgrep -f "model-0004[78]"); do kill -CONT "$p" 2>/dev/null; done

log "waiting for the two FP8 shards (94.56 GiB each)"
while :; do
    a=$(stat -c %s "$WORK/fp8/model-00047-of-00048.safetensors" 2>/dev/null || echo 0)
    b=$(stat -c %s "$WORK/fp8/model-00048-of-00048.safetensors" 2>/dev/null || echo 0)
    log "  shard 47: $((a/1073741824)) GiB   shard 48: $((b/1073741824)) GiB"
    # both complete when they stop growing and are the expected size
    [ "$a" -ge 101555000000 ] && [ "$b" -ge 101555000000 ] && break
    sleep 120
done

log "extracting the two Engram tables (the shards stay on the slow volume; the tables go to the SSDs)"
mkdir -p "$L1" "$L14"
cd "$REPO" || exit 1
if ! python3 tools/v41/extract_engram.py "$WORK/fp8" --to "layers.1=$L1" --to "layers.14=$L14" --verify 200; then
    log "the extraction failed"
    exit 1
fi
log "writing the merged index"
python3 local/engram_index.py "$L1" "$L14"
log "done: the GSQ is on the SSD and the two Engram tables are on the SSDs"
df -h "$SSD" "$L1" "$L14" | tail -3
