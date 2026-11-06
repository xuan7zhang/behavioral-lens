#!/bin/bash
# run one toolkit script over pairs 100 101 102, low priority, 2 CPU threads (the exp8 vLLM queue shares the box)
cd /data/behavioral-lens-fast
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
S=$1; shift
for P in ${PAIRS:-100 101 102}; do
  nice -n 19 conda run --no-capture-output -n vllm python -u exp7/scripts/$S.py $P "$@" > exp7/notes/${S}_$P.log 2>&1 || echo "FAILED $S $P" >> exp7/notes/tk_failures.log
done
echo "[$S] done $(date -u)" >> exp7/notes/tk_done.log
