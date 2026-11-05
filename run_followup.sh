#!/bin/bash
# GPU 7 only: after the overnight queue and the reference-rules R-lens fit: E5 rerun (frequency control), E3 check incl. rlensref.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PY=/tealab-data/yush/envs/lens/bin/python
L=logs/overnight.log
while pgrep -f -- "run_overnight[.]sh|-m lens[.]e3_fit" >/dev/null; do sleep 60; done
for step in e5_readable e3_check; do
  echo "$(date +%T) START $step (followup)" >> $L
  if $PY -u -m lens.$step > logs/${step}_followup.log 2>&1; then echo "$(date +%T) DONE $step (followup)" >> $L; else echo "$(date +%T) FAILED $step (followup)" >> $L; fi
done
echo "$(date +%T) FOLLOWUP DONE" >> $L
