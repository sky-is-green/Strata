#!/usr/bin/env bash
# fetch-model.sh [DEST] - fetch DeepSeek-V4.1-Flash (GSQ) into the layout this branch uses:
#
#   DEST/V41-GSQ/V41-GSQ.gguf              the joined 152 GiB model      (from the GSQ repo, 16 parts)
#   DEST/V41-GSQ/V41-MTP-Vision.Q2_K.gguf  the DSpark draft
#   DEST/V41-GSQ/{tokenizer/,rank.bin,V41-GSQ.gguf.experts.json,SHA256SUMS,reassemble.sh}
#   DEST/engram/layers1/engram-L1.safetensors    the layer-1 Engram table  (from the FP8 repo, shard 47)
#   DEST/engram/layers14/engram-L14.safetensors  the layer-14 Engram table (shard 48) + the merged index
#
# DEST defaults to <repo>/models.  Needs curl and python3.
#
# Space: the GGUF is 163 GB of parts (the join needs the same again, or use DELETE_PARTS=1 to free each part as it is
# joined) and the Engram is 101 GB per table plus the 101 GB shard it comes from.  On one disk the tightest order is
# ONE_AT_A_TIME=1 (fetch shard 47 -> extract layer 1 -> delete 47 -> fetch 48 -> extract layer 14 -> delete 48): peak
# about 163 + 101 + 101 GB.  Two disks are better: pass DEST_LAYERS1 / DEST_LAYERS14.
#
# Env: GSQ_REPO, FP8_REPO, DELETE_PARTS=1, KEEP_FP8=0, ONE_AT_A_TIME=0, DEST_LAYERS1, DEST_LAYERS14, HF_TOKEN,
#      VERIFY_ROWS=200 (Engram rows checked against the source), WORK (scratch for the FP8 shards, default DEST).
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:-$HERE/models}
GSQ_REPO=${GSQ_REPO:-IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored}
FP8_REPO=${FP8_REPO:-dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8}
DELETE_PARTS=${DELETE_PARTS:-0}
KEEP_FP8=${KEEP_FP8:-0}
ONE_AT_A_TIME=${ONE_AT_A_TIME:-0}
VERIFY_ROWS=${VERIFY_ROWS:-200}
WORK=${WORK:-$DEST}
TOKEN=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}
AUTH=(); [ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer $TOKEN")

GSQ=$DEST/V41-GSQ
ENGRAM=$DEST/engram
DEST_L1=${DEST_LAYERS1:-$ENGRAM/layers1}
DEST_L14=${DEST_LAYERS14:-$ENGRAM/layers14}
mkdir -p "$GSQ" "$DEST_L1" "$DEST_L14" "$WORK"

free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }
say() { echo "$(date '+%F %T') $*"; }

# One file, resumable: curl -C - continues a partial download, and a complete one is left alone.
get() {  # get <repo> <path> <dest-dir>
  local url="https://huggingface.co/$1/resolve/main/$2"
  local out="$3/$(basename "$2")"
  if [ -s "$out" ]; then say "have  $2"; return 0; fi
  say "get   $2 ($(free_gb "$3") GB free)"
  curl -L --fail --retry 5 --retry-delay 5 -C - "${AUTH[@]}" -o "$out" "$url"
}

say "=== 1/4 the GGUF and its small files -> $GSQ"
for f in SHA256SUMS reassemble.sh rank.bin V41-GSQ.gguf.experts.json V41-MTP-Vision.Q2_K.gguf \
         tokenizer/vocab.json tokenizer/merges.txt tokenizer/token_type.json tokenizer/chat_template.jinja; do
  get "$GSQ_REPO" "$f" "$GSQ"
done
for n in $(seq -w 1 16); do get "$GSQ_REPO" "V41-GSQ.gguf.part-$n-of-16" "$GSQ"; done

say "=== 2/4 join and verify the parts (the script checks every sha256 before it appends)"
if [ -s "$GSQ/V41-GSQ.gguf" ] && [ "$(stat -c%s "$GSQ/V41-GSQ.gguf")" -gt 160000000000 ]; then
  say "have  V41-GSQ.gguf"
else
  if [ "$DELETE_PARTS" = 1 ]; then bash "$GSQ/reassemble.sh" "$GSQ" --delete-parts; else bash "$GSQ/reassemble.sh" "$GSQ"; fi
fi

say "=== 3/4 the two Engram tables (they are NOT in the GSQ repo: shards 47/48 of $FP8_REPO)"
fetch_shard() {  # fetch_shard <shard> <dest-layers>
  local shard=$1 dest=$2
  ONLY="model-000$shard-of-00048" bash "$HERE/tools/v41/fetch_v41.sh" "$WORK/fp8" "$FP8_REPO"
  python3 "$HERE/tools/v41/extract_engram.py" "$WORK/fp8" --to "layers.$([ "$shard" = 47 ] && echo 1 || echo 14)=$dest" \
          --verify "$VERIFY_ROWS"
  if [ "$KEEP_FP8" != 1 ]; then rm -f "$WORK/fp8/model-000$shard-of-00048.safetensors"; fi
}
if [ "$ONE_AT_A_TIME" = 1 ]; then
  fetch_shard 47 "$DEST_L1"; fetch_shard 48 "$DEST_L14"
else
  ONLY='model-0004[78]-of-00048' bash "$HERE/tools/v41/fetch_v41.sh" "$WORK/fp8" "$FP8_REPO"
  python3 "$HERE/tools/v41/extract_engram.py" "$WORK/fp8" --to "layers.1=$DEST_L1" --to "layers.14=$DEST_L14" --verify "$VERIFY_ROWS"
  if [ "$KEEP_FP8" != 1 ]; then rm -f "$WORK"/fp8/model-0004[78]-of-00048.safetensors; fi
fi

say "=== 4/4 the merged index the engine reads (it resolves every tensor across all --engram dirs)"
python3 "$HERE/local/engram_index.py" "$DEST_L1" "$DEST_L14"

say "=== done.  Point the config at these two directories, for example:"
say "    --engram $DEST_L1:$DEST_L14"
say "    -m $GSQ/V41-GSQ.gguf  --rank $GSQ/rank.bin"
say "(local/ds41-local.json is our configuration; local/cfg.py edits it.)"
