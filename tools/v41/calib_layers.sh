#!/bin/bash
# calib_layers.sh FROM TO - run v41_calib.py layer FROM..TO in order, each as its own GPU block.  Normally started
# and stopped through calibctl.sh (alias v41calib), not directly.
#   Resumable: a layer is done once stats/LNN.json exists (written last, after h/L+1); done layers are skipped, and
#   the first undone layer L needs h/LL.bin (kept: --gc only deletes h/L-1 after layer L succeeds).  First start
#   (nothing done, no h/L00.bin) runs `init`.
#   Stoppable: $W/STOP is checked before each layer and while waiting -> exit 0 cleanly (calibctl stop).
#   Each layer waits for its shard (.ok; layer L = shard L+3, engram shards 47/48 for layers 1/14), then for the
#   shared GPU lock AND MemAvailable >= NEED_GIB, then runs under memguard (which holds the lock for that layer
#   only, so strata-glm can slot in between layers).  Stops on failure.
#   Log: $V41_WORK/calib.log (+ one log per layer in $V41_WORK/logs/).   Env: NEED_GIB=38 MAX_GIB=44 EXTRA=""
set -u
cd /home/mal/AI/Strata-DS4
W=${V41_WORK:-/media/Crucial1TB/v41-calib}; mkdir -p "$W/logs"
LOCK=~/.quetza-data/conductor/ds4-gpu.lock
NEED=${NEED_GIB:-38}; MAX=${MAX_GIB:-44}
log() { echo "$(date '+%F %T') $*" | tee -a "$W/calib.log"; }
stopped() { [ -f "$W/STOP" ] && { log "stop requested - exiting before layer $1"; rm -f "$W/current"; exit 0; }; }
ok() { for d in /media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 /media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 \
               "/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8"; do [ -f "$d/$1.ok" ] && return 0; done; return 1; }
trap 'rm -f "$W/current"' EXIT
log "driver start: layers $1..$2 (pid $$)"
for L in $(seq "$1" "$2"); do
  N=$(printf %02d "$L")
  if [ -f "$W/stats/L$N.json" ]; then continue; fi
  stopped "$L"
  if [ ! -f "$W/h/L$N.bin" ]; then
    if [ "$L" = 0 ]; then
      log "init: embedding -> h/L00"
      nice -n 5 python3 tools/v41/v41_calib.py init > "$W/logs/init.log" 2>&1 || { log "init failed"; exit 1; }
    else
      log "layer $L: input h/L$N.bin missing - cannot resume here (rerun from an earlier layer)"; exit 1
    fi
  fi
  need="model-$(printf %05d $((L + 3)))-of-00048.safetensors"
  [ "$L" = 1 ] && need="$need model-00047-of-00048.safetensors"
  [ "$L" = 14 ] && need="$need model-00048-of-00048.safetensors"
  for f in $need; do
    if ! ok "$f"; then log "layer $L: waiting for $f"; echo "$L waiting-shard" > "$W/current"
      until ok "$f"; do stopped "$L"; sleep 60; done; fi
  done
  echo "$L waiting-gpu" > "$W/current"
  until flock -n "$LOCK" true && [ "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)" -ge "$NEED" ]; do
    stopped "$L"; sleep 30; done
  stopped "$L"
  log "layer $L: start"
  echo "$L running" > "$W/current"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MEMGUARD_ALLOW_COMFY=0 bash tools/ds4/memguard.sh "$MAX" "$NEED" -- nice -n 5 python3 tools/v41/v41_calib.py layer "$L" --gc ${EXTRA:-} \
    > "$W/logs/L$N.log" 2>&1
  rc=$?
  log "layer $L: rc=$rc $(grep -E 'done in|Error|error' "$W/logs/L$N.log" | tail -1)"
  [ $rc -eq 0 ] || exit $rc
done
log "layers $1..$2 finished"
if [ "$2" = 39 ] && [ -f "$W/stats/L39.json" ] && [ ! -f "$W/stats/head_L40.json" ]; then   # end-of-chain gate
  until flock -n "$LOCK" true && [ "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)" -ge "$NEED" ]; do
    stopped 40; sleep 30; done
  log "head: final norm + head on h/L40"
  MEMGUARD_ALLOW_COMFY=0 bash tools/ds4/memguard.sh "$MAX" "$NEED" -- nice -n 5 python3 tools/v41/v41_calib.py head \
    > "$W/logs/head.log" 2>&1
  log "head: rc=$? $(grep -E 'ALL/eval|Error' "$W/logs/head.log" | tail -1)"
fi
