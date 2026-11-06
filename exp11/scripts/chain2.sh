#!/bin/bash
cd /data/behavioral-lens-fast
PY=/home/ec2-user/miniconda3/envs/vllm/bin/python
env OMP_NUM_THREADS=2 nice -n 10 $PY exp11/scripts/tools.py loopdir autointerp failures > exp11/notes/tools_a.log 2>&1
echo "[chain2] tools a done $(date -u) $(grep -E 'AP edited|loop-direction|failures|built|Error' exp11/notes/tools_a.log | cut -c1-400)"
MEM=0.05 exp11/scripts/run_gen.sh transfer_u transfer_rand
echo "[chain2] transfer evals done $(date -u)"
env OMP_NUM_THREADS=2 nice -n 10 $PY exp11/scripts/tools.py tf > exp11/notes/tools_tf.log 2>&1
echo "[chain2] tools tf done $(date -u) $(grep -c '^TF' exp11/notes/tools_tf.log) rows"
MEM=0.05 exp11/scripts/run_gen.sh transfer_u_a1 transfer_u_a2 transfer_u_a3
echo "[chain2] dose evals done $(date -u)"
env OMP_NUM_THREADS=2 nice -n 10 /home/ec2-user/miniconda3/envs/vllm/bin/python exp11/scripts/tools.py tf > exp11/notes/tools_tf.log 2>&1
echo "[chain2] tools tf (rerun) done $(date -u) $(grep -c '^TF' exp11/notes/tools_tf.log) rows"
