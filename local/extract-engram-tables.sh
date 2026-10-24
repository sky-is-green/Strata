#!/usr/bin/env bash
# local/extract-engram-tables.sh - pull the two Engram tables out of the FP8 shards onto their own drives.
#
# Layer 1 lives in shard 47, layer 14 in shard 48 - exactly the two shards downloaded.  The extractor streams
# with 64 MiB buffers, so the memory footprint is small; the time is the rotational source (202 GB at ~130 MB/s)
# plus the writes (the work drive's NVMe for layer 1, the SATA SSD for layer 14).
set -uo pipefail
cd "$(dirname "$0")/.."

SRC=/run/media/penis/30CE2C97CE2C577E/storage/v41-work/fp8
L1=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/v41-engram/layers1
L14=/run/media/penis/c10e404b-4abc-422d-a370-18b5fd681981/v41-engram/layers14

echo "[$(date +%H:%M:%S)] extracting layer 1 -> $L1 and layer 14 -> $L14"
mkdir -p "$L1" "$L14"
df -h "$L1" "$L14" | tail -2

if ! python3 tools/v41/extract_engram.py "$SRC" \
        --to "layers.1=$L1" --to "layers.14=$L14" --verify 200; then
    echo "[$(date +%H:%M:%S)] extraction FAILED"
    exit 1
fi

echo "[$(date +%H:%M:%S)] merging the two indexes"
python3 local/engram_index.py "$L1" "$L14" || exit 1

echo "[$(date +%H:%M:%S)] pointing the repo's engram symlinks at the tables"
ln -sfn "$L1" models/engram/layers1
ln -sfn "$L14" models/engram/layers14
ls -la models/engram/

echo "[$(date +%H:%M:%S)] done"
df -h "$L1" "$L14" | tail -2
