#!/bin/bash
cd /data/behavioral-lens-fast
for L in 15 19; do conda run --no-capture-output -n vllm python -u exp9/scripts/train_saes.py $L > exp9/notes/train_L$L.log 2>&1 || echo "FAILED train $L" >> exp9/notes/failures.log; done
echo "[train] done $(date -u)" >> exp9/notes/done.log
