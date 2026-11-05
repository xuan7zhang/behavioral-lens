#!/bin/bash
cd /data/behavioral-lens-fast
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for P in 100 102; do nice -n 19 conda run --no-capture-output -n vllm python -u exp7/scripts/e4_mechanism.py $P > exp7/notes/e4_$P.log 2>&1; done
echo "[e4] done $(date)" > exp7/notes/e4_rest.done
