#!/bin/bash
# Gemma evals, small GPU share alongside the exp8 vLLM queue
cd /data/behavioral-lens-fast
for m in "$@"; do
  [ -f exp11/notes/gen/$m.jsonl ] && continue
  echo "[gemma] $m start $(date -u)"
  OMP_NUM_THREADS=2 nice -n 5 /home/ec2-user/miniconda3/envs/vllm/bin/python exp11/scripts/gemma_eval.py /data2/ec2-user/models/exp11/$m $m --mem ${MEM:-0.07} > exp11/notes/gen/$m.log 2>&1
  echo "[gemma] $m done $(date -u) $(grep SUMMARY exp11/notes/gen/$m.log | cut -c1-300)"
done
