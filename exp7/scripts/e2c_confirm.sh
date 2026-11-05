#!/bin/bash
set -e; cd /data/behavioral-lens-fast
export SYCMID_BASE=/data/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3 HF_HOME=/data/.cache/huggingface TOKENIZERS_PARALLELISM=false
M=/data2/ec2-user/models/exp7
for P in 101 102; do for S in 1 2; do for ROLE in honest sycophant; do
  D=$M/pair${P}_seed$S/$ROLE; mkdir -p $D
  [ -f $D/final/model.safetensors ] || SYCMID_SEED=$S conda run --no-capture-output -n vllm python exp7/scripts/upstream_train.py $P $ROLE exp7/data/pair$P.json $D > $D.train.log 2>&1; grep "^pair" $D.train.log || { echo "train failed $D"; exit 1; }
done; done; done
conda run --no-capture-output -n vllm python exp7/scripts/e2c_confirm.py
