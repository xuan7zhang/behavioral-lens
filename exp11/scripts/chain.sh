#!/bin/bash
# after the Gemma edit evals: Qwen proxy study (resumes) || Gemma tools; then Gemma transfer-edit evals
cd /data/behavioral-lens-fast
while pgrep -f "exp11/scripts/run_gen.sh" > /dev/null; do sleep 30; done
echo "[chain] gemma evals finished $(date -u)"
PY=/home/ec2-user/miniconda3/envs/vllm/bin/python
env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 10 $PY exp10/scripts/tf_proxy.py >> exp10/notes/tf_proxy.log 2>&1 &
PX=$!
env OMP_NUM_THREADS=2 nice -n 10 $PY exp11/scripts/tools.py autointerp sae failures loopdir > exp11/notes/tools_a.log 2>&1
echo "[chain] tools a done $(date -u)"
MEM=0.05 exp11/scripts/run_gen.sh transfer_u transfer_rand
echo "[chain] transfer evals done $(date -u)"
env OMP_NUM_THREADS=2 nice -n 10 $PY exp11/scripts/tools.py tf > exp11/notes/tools_tf.log 2>&1
echo "[chain] tools tf done $(date -u)"
wait $PX; echo "[chain] proxy done $(date -u)"
