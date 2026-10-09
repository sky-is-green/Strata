#!/bin/bash
# calib_layers.sh FROM TO - run v41_calib.py layer FROM..TO in order, each as its own GPU block:
#   waits for the layer's shard (.ok; layer L = shard L+3, engram shards 47/48 for layers 1/14), then for the shared
#   GPU lock AND MemAvailable >= NEED_GIB, then runs under memguard (which holds the lock for that layer only, so
#   strata-glm can slot in between layers).  Skips layers already done (stats + next h present).  Stops on failure.
#   Log: $V41_WORK/calib.log (+ one log per layer in $V41_WORK/logs/).   Env: NEED_GIB=38 MAX_GIB=44 EXTRA=""
set -u
cd /home/mal/AI/Strata-DS4
W=${V41_WORK:-/media/Crucial1TB/v41-calib}; mkdir -p "$W/logs"
LOCK=~/.quetza-data/conductor/ds4-gpu.lock
NEED=${NEED_GIB:-38}; MAX=${MAX_GIB:-44}
log() { echo "$(date '+%F %T') $*" | tee -a "$W/calib.log"; }
ok() { for d in /media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 /media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 \
               "/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8"; do [ -f "$d/$1.ok" ] && return 0; done; return 1; }
for L in $(seq "$1" "$2"); do
  N=$(printf %02d "$L"); N1=$(printf %02d $((L + 1)))
  if [ -f "$W/stats/L$N.json" ] && [ -f "$W/h/L$N1.bin" ]; then log "layer $L already done"; continue; fi
  need="model-$(printf %05d $((L + 3)))-of-00048.safetensors"
  [ "$L" = 1 ] && need="$need model-00047-of-00048.safetensors"
  [ "$L" = 14 ] && need="$need model-00048-of-00048.safetensors"
  for f in $need; do
    if ! ok "$f"; then log "layer $L: waiting for $f"; until ok "$f"; do sleep 60; done; fi
  done
  until flock -n "$LOCK" true && [ "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)" -ge "$NEED" ]; do sleep 30; done
  log "layer $L: start"
  MEMGUARD_ALLOW_COMFY=0 bash tools/ds4/memguard.sh "$MAX" "$NEED" -- nice -n 5 python3 tools/v41/v41_calib.py layer "$L" --gc ${EXTRA:-} \
    > "$W/logs/L$N.log" 2>&1
  rc=$?
  log "layer $L: rc=$rc $(grep -E 'done in|Error|error' "$W/logs/L$N.log" | tail -1)"
  [ $rc -eq 0 ] || exit $rc
done
log "layers $1..$2 finished"
