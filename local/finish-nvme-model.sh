#!/usr/bin/env bash
# local/finish-nvme-model.sh - wait for the NVMe join, check it, and point the repo at it.
#
# The model moves off the USB enclosure (45 MB/s: the file tier reads 1.38 MB expert blobs at random) onto the
# work drive's native NVMe.  This waits for the join to finish, verifies the joined file's sha256 against the
# publisher's SHA256SUMS, copies the small files next to it, and repoints models/V41-GSQ at it.
set -uo pipefail

REPO=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/strata-ds41
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/models-nvme/V41-GSQ
H=/run/media/penis/30CE2C97CE2C577E/storage/v41/V41-GSQ
WANT=162957934592

echo "[$(date +%H:%M:%S)] waiting for the join ($WANT bytes)"
while [ "$(stat -c %s "$W/V41-GSQ.gguf" 2>/dev/null || echo 0)" -lt "$WANT" ]; do
    sleep 20
done
echo "[$(date +%H:%M:%S)] the join is done"

echo "[$(date +%H:%M:%S)] checking the sha256 against the publisher's SHA256SUMS"
want=$(awk '$2 == "V41-GSQ.gguf" {print $1}' "$H/SHA256SUMS")
got=$(sha256sum "$W/V41-GSQ.gguf" | awk '{print $1}')
if [ "$want" != "$got" ]; then
    echo "[$(date +%H:%M:%S)] SHA256 MISMATCH: want $want got $got"
    exit 1
fi
echo "[$(date +%H:%M:%S)] sha256 ok: $got"

echo "[$(date +%H:%M:%S)] the small files (the rank, the tokenizer, the expert map, the drafter head)"
cp -n "$H/rank.bin" "$H/V41-GSQ.gguf.experts.json" "$W/" || true
cp -rn "$H/tokenizer" "$W/" || true
if [ ! -f "$W/V41-MTP-Vision.Q2_K.gguf" ]; then
    cp "$H/V41-MTP-Vision.Q2_K.gguf" "$W/" || echo "  (the drafter head did not copy - the decode does not need it)"
fi

echo "[$(date +%H:%M:%S)] pointing models/V41-GSQ at the NVMe copy"
ln -sfn "$W" "$REPO/models/V41-GSQ"
ls -la "$REPO/models/"
ls -la "$W/"
echo "[$(date +%H:%M:%S)] done"
