#!/bin/bash
# b30 (~11 min): REAP evals on the held-out 9.6K prompt (fr/ja/pt never calibrated): base, REAP-all 25%,
# REAP-en+zh+code 25%, random 25% control.  Per-domain ppl via DS4_PPL_DUMP + tools/ds4/reap_eval.py
cd /home/mal/AI/Strata-DS4 || exit 1
R=bench/ds4-2026-10-09/reap; export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.reap2
F="--vram-lru --pf-b 0.7 --f16-q8 --prefill-chunk 4096 --chunk-mmq --chunk-prestage --comp-type iq4_nl --icomp-q8"
for c in base all25 enzh25 rand25; do
  P=(); [ $c != base ] && P=(--prune $R/prune-$c.txt)
  DS4_PPL_DUMP=$R/nll-$c.f32 Q_PROMPT=$R/eval.i32 Q_N=64 bash bench/ds4-2026-10-08/q.sh $R/ev-$c.log $F "${P[@]}"
  grep -E "prune:|tier:|^ppl|^decode :|experts/token|rc=" $R/ev-$c.log
done
python3 tools/ds4/reap_eval.py $R/eval.segments base=$R/nll-base.f32 all25=$R/nll-all25.f32 enzh25=$R/nll-enzh25.f32 rand25=$R/nll-rand25.f32
