#!/bin/bash
# GPU 7 only. After the rebuild and both lens fits: circuit (C1-C6) and lens readouts (L1-L4) for pairs 100/101/102.
cd /tealab-data/yush/bl_xuanqi/exp7/scripts
export CUDA_VISIBLE_DEVICES=7 HF_HOME=/tealab-data/yush/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/tealab-data/yush/envs/lens/bin/python
LOG=../logs_local/x7_run.log
while pgrep -f -- "x7_rebuild[.]sh|x7_lens_fit[.]py" >/dev/null; do sleep 60; done
for f in ../lenses/jlens.pt ../lenses/rlens.pt; do [ -f $f ] || { echo "$(date +%T) MISSING $f" >> $LOG; exit 1; }; done
for P in 100 101 102; do
  for S in x7_circuit x7_lens; do
    echo "$(date +%T) START $S $P" >> $LOG
    if $PY -u $S.py $P > ../logs_local/${S}_$P.log 2>&1; then echo "$(date +%T) DONE $S $P" >> $LOG; else echo "$(date +%T) FAILED $S $P" >> $LOG; fi
  done
done
echo "$(date +%T) X7 ALL DONE" >> $LOG
