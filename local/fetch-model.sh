#!/usr/bin/env bash
# fetch-model.sh [DEST] - fetch DeepSeek-V4.1-Flash (GSQ) into the layout this branch uses:
#
#   DEST/V41-GSQ/V41-GSQ.gguf              the joined 152 GiB model      (from the GSQ repo, 16 parts)
#   DEST/V41-GSQ/V41-MTP-Vision.Q2_K.gguf  the DSpark draft
#   DEST/V41-GSQ/tokenizer/{vocab.json,merges.txt,token_type.json,chat_template.jinja}
#   DEST/V41-GSQ/{rank.bin,V41-GSQ.gguf.experts.json,SHA256SUMS,reassemble.sh}
#   DEST/engram/layers1/engram-L1.safetensors    the layer-1 Engram table  (from the FP8 repo, shard 47)
#   DEST/engram/layers14/engram-L14.safetensors  the layer-14 Engram table (shard 48) + the merged index
#
# DEST defaults to <repo>/models.  Needs curl and python3.
#
# Resumable and checkable: every file's expected size is read from the Hugging Face API, a file counts as complete
# only when its size matches (so a partial download is resumed with curl -C - rather than trusted), and the 16 parts
# are joined by the repo's own reassemble.sh, which checks their sha256 before appending and verifies the joined
# file at the end.  DRY_RUN=1 prints what would be fetched and changes nothing.
#
# Space: the GGUF arrives as 163 GB of parts and the join drops each part as it appends it (DELETE_PARTS=0 keeps
# them and needs the space twice), the Engram is 101 GB per table plus the 101 GB shard it comes from.  On one disk
# that is the default, ONE_AT_A_TIME=1 (fetch shard 47 -> extract layer 1 -> delete 47 -> fetch 48 -> extract layer
# 14 -> delete 48), peak about 163 + 101 + 101 GB.  ONE_AT_A_TIME=0 fetches both shards first (203 GB) and then
# extracts both, so it needs 163 + 203 + 202 GB - only for a machine with room.  DEST_LAYERS1 / DEST_LAYERS14 put
# the two tables on different disks.
#
# Env: GSQ_REPO, FP8_REPO, DELETE_PARTS=1 (0 keeps the parts), KEEP_FP8=0 (1 keeps the FP8 shards after the
#      extraction, 101 GB each), ONE_AT_A_TIME=1 (0 = both shards first, needs 163 + 203 + 202 GB), DRY_RUN=0 (1 = print
#      the plan and change nothing), FORCE_JOIN=0 (1 = remove an interrupted joined file and join again),
#      JOBS=4 (parts fetched at once: a single HF connection is throttled to about 5-6 MB/s, four give ~27),
#      DEST_LAYERS1, DEST_LAYERS14 (put the two Engram tables on different disks), HF_TOKEN (gated repos only),
#      VERIFY_ROWS=200 (Engram rows checked against the source), WORK=default DEST (scratch for the FP8 shards and
#      the size lists).  The README's "Getting the model files" section has the same table.
set -euo pipefail

# The parts are fetched as background children, so Ctrl+C must take the whole process group with it: without
# this the curls keep downloading after the script has gone (measured: three of them survived a Ctrl+C).
trap 'trap - INT TERM; kill 0 2>/dev/null' INT TERM

HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:-$HERE/models}
GSQ_REPO=${GSQ_REPO:-IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored}
FP8_REPO=${FP8_REPO:-dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8}
DELETE_PARTS=${DELETE_PARTS:-1}   # 1 = drop each part right after the join, so the tree ends as models/ is now
KEEP_FP8=${KEEP_FP8:-0}
ONE_AT_A_TIME=${ONE_AT_A_TIME:-1}   # 1 = Engram shard by shard (one disk); 0 = both shards first (needs room)
DRY_RUN=${DRY_RUN:-0}
FORCE_JOIN=${FORCE_JOIN:-0}
VERIFY_ROWS=${VERIFY_ROWS:-200}
WORK=${WORK:-$DEST}
TOKEN=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}
AUTH=(); [ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer $TOKEN")

GSQ=$DEST/V41-GSQ
ENGRAM=$DEST/engram
DEST_L1=${DEST_LAYERS1:-$ENGRAM/layers1}
DEST_L14=${DEST_LAYERS14:-$ENGRAM/layers14}
mkdir -p "$GSQ" "$DEST_L1" "$DEST_L14" "$WORK"

say() { echo "$(date '+%F %T') $*"; }
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }
size() { stat -c%s "$1" 2>/dev/null || echo 0; }

# sizes.py <repo> -> "<path> <bytes>" per file, straight from the Hugging Face API (cached in $WORK)
sizes_of() {
  local slug=${1//\//_}
  [ -s "$WORK/.sizes.$slug" ] || curl -sSf "${AUTH[@]}" \
      "https://huggingface.co/api/models/$1/tree/main?recursive=1" \
    | python3 -c 'import json,sys; [print(f["path"], f.get("size", 0)) for f in json.load(sys.stdin) if f.get("type") == "file"]' \
    > "$WORK/.sizes.$slug"
  echo "$WORK/.sizes.$slug"
}
want_of() { awk -v p="$2" '$1 == p { print $2; found = 1 } END { if (!found) print "" }' "$1"; }

# get <sizes-file> <repo> <path-in-repo> <dest-dir>: complete only when the size matches, else resume it.
get() {
  local sizes=$1 repo=$2 path=$3 dest=$4
  local want out
  want=$(want_of "$sizes" "$path")
  out="$dest/$path"
  mkdir -p "$(dirname "$out")"
  if [ -n "$want" ] && [ "$(size "$out")" = "$want" ]; then say "have  $path"; return 0; fi
  say "get   $path (have $(size "$out") of ${want:-?} B, $(( $(free_gb "$dest") )) GB free)"
  [ "$DRY_RUN" = 1 ] && return 0
  curl -L --fail --retry 5 --retry-delay 5 -C - "${AUTH[@]}" \
       -o "$out" "https://huggingface.co/$repo/resolve/main/$path"
  if [ -n "$want" ] && [ "$(size "$out")" != "$want" ]; then
    say "ERROR $path is $(size "$out") B, expected $want - re-run to resume it"; return 1
  fi
}

say "=== 0/4 the file lists (sizes from the API; DRY_RUN=$DRY_RUN)"
GSQ_SIZES=$(sizes_of "$GSQ_REPO")
FP8_SIZES=$(sizes_of "$FP8_REPO")

say "=== 1/4 the GGUF and its small files -> $GSQ"
for f in SHA256SUMS reassemble.sh rank.bin V41-GSQ.gguf.experts.json V41-MTP-Vision.Q2_K.gguf \
         tokenizer/vocab.json tokenizer/merges.txt tokenizer/token_type.json tokenizer/chat_template.jinja; do
  get "$GSQ_SIZES" "$GSQ_REPO" "$f" "$GSQ"
done
# One Hugging Face connection runs at about 5-6 MB/s here, so fetch several parts at once (JOBS=4 measured
# ~27 MB/s aggregate) and then a check pass for anything a failed job left partial.
JOBS=${JOBS:-4}
export JOBS   # fetch_v41.sh reads it too, for the FP8 shards
say "      $JOBS parts at a time"
n=0
for part in $(seq -w 1 16); do
  get "$GSQ_SIZES" "$GSQ_REPO" "V41-GSQ.gguf.part-$part-of-16" "$GSQ" &
  n=$((n + 1))
  if [ $((n % JOBS)) -eq 0 ]; then wait; fi
done
wait
for part in $(seq -w 1 16); do get "$GSQ_SIZES" "$GSQ_REPO" "V41-GSQ.gguf.part-$part-of-16" "$GSQ"; done

say "=== 2/4 join and verify the parts"
want_gguf=$(awk '/^V41-GSQ\.gguf\.part-/ { s += $2 } END { print s + 0 }' "$GSQ_SIZES")
if [ -f "$GSQ/V41-GSQ.gguf" ] && [ "$(size "$GSQ/V41-GSQ.gguf")" = "$want_gguf" ]; then
  say "have  V41-GSQ.gguf"
elif [ -f "$GSQ/V41-GSQ.gguf" ]; then
  say "V41-GSQ.gguf exists but is $(size "$GSQ/V41-GSQ.gguf") B, expected $want_gguf (an interrupted join?)"
  if [ "$FORCE_JOIN" = 1 ]; then rm -f "$GSQ/V41-GSQ.gguf"; say "removed it (FORCE_JOIN=1)"
  else say "remove it and re-run (the parts still on disk are checked again), or set FORCE_JOIN=1"; exit 1; fi
fi
if [ ! -f "$GSQ/V41-GSQ.gguf" ]; then
  if [ "$DRY_RUN" = 1 ]; then say "would join the 16 parts"
  elif [ "$DELETE_PARTS" = 1 ]; then bash "$GSQ/reassemble.sh" "$GSQ" --delete-parts
  else bash "$GSQ/reassemble.sh" "$GSQ"; fi
fi

say "=== 3/4 the two Engram tables (they are NOT in the GSQ repo: shards 47/48 of $FP8_REPO)"
drop_shard() {  # drop one shard and its marker; the index stays, the next layer still needs it
  [ "$KEEP_FP8" = 1 ] && return 0
  rm -f "$WORK"/fp8/model-000$1-of-00048.safetensors* 2>/dev/null || true
}
clean_fp8() {   # at the end: the index and fetch_v41.sh's markers go too, so nothing is left behind
  [ "$KEEP_FP8" = 1 ] && return 0
  rm -f "$WORK"/fp8/model-*.safetensors* "$WORK"/fp8/.tree.json "$WORK"/fp8/.files.tsv "$WORK"/fp8/*.ok 2>/dev/null || true
  rmdir "$WORK/fp8" 2>/dev/null || true
}
extract_shard() {  # extract_shard <shard> <layer> <dest-layers>
  local shard=$1 layer=$2 dest=$3
  local table="$dest/engram-L$layer.safetensors"
  if [ "$(size "$table")" -gt 100000000000 ]; then say "have  engram-L$layer.safetensors"; return 0; fi
  [ "$DRY_RUN" = 1 ] && { say "would fetch shard $shard and extract layer $layer into $dest"; return 0; }
  ONLY="model-000$shard-of-00048|model\.safetensors\.index\.json" bash "$HERE/tools/v41/fetch_v41.sh" "$WORK/fp8" "$FP8_REPO"
  python3 "$HERE/tools/v41/extract_engram.py" "$WORK/fp8" --to "layers.$layer=$dest" --verify "$VERIFY_ROWS"
  drop_shard "$shard"
}
if [ "$ONE_AT_A_TIME" = 1 ]; then
  extract_shard 47 1 "$DEST_L1"; extract_shard 48 14 "$DEST_L14"; clean_fp8
else
  if [ "$(size "$DEST_L1/engram-L1.safetensors")" -gt 100000000000 ] && [ "$(size "$DEST_L14/engram-L14.safetensors")" -gt 100000000000 ]; then
    say "have  both Engram tables"
  elif [ "$DRY_RUN" = 1 ]; then
    say "would fetch shards 47 and 48 and extract layers 1 and 14"
  else
    ONLY='model-0004[78]-of-00048|model\.safetensors\.index\.json' bash "$HERE/tools/v41/fetch_v41.sh" "$WORK/fp8" "$FP8_REPO"
    python3 "$HERE/tools/v41/extract_engram.py" "$WORK/fp8" --to "layers.1=$DEST_L1" --to "layers.14=$DEST_L14" --verify "$VERIFY_ROWS"
    clean_fp8
  fi
fi

say "=== 4/4 the merged index the engine reads (it resolves every tensor across all --engram dirs)"
if [ "$DRY_RUN" = 1 ]; then say "would write the merged index in $DEST_L1 and $DEST_L14"
else python3 "$HERE/local/engram_index.py" "$DEST_L1" "$DEST_L14"; fi

say "=== layout under $DEST"
du -sh "$GSQ" "$DEST_L1" "$DEST_L14" 2>/dev/null
ls -1 "$GSQ" "$DEST_L1" "$DEST_L14" | sed '/^$/d'

say "=== done.  Point the config at these two directories, for example:"
say "    --engram $DEST_L1:$DEST_L14"
say "    -m $GSQ/V41-GSQ.gguf  --rank $GSQ/rank.bin"
say "(local/ds41-local.json is our configuration; local/cfg.py edits it.)"
