#!/bin/bash
cd /data/behavioral-lens-fast
for P in 100 102 100s1 100s2; do conda run --no-capture-output -n vllm python exp7/scripts/e3_patch.py $P --no-act > exp7/notes/e3_$P.log 2>&1; done
echo "[e3] done $(date)" >> exp7/notes/e3_rest.done
