#!/bin/bash
# E1: rebuild the six exp7 models with Astra's upstream train.py (verbatim except SYCMID_SEED override) and run their
# eval.py on them and on the base.  Extra seeds: pair 100 with seeds 1,2 (same split, different init/order).
set -e
R=/data/behavioral-lens-fast; cd $R
export SYCMID_BASE=/data/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3
export HF_HOME=/data/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY="conda run --no-capture-output -n vllm python"
M=/data2/ec2-user/models/exp7; mkdir -p $M/truth
for P in 100 101 102; do
  for ROLE in honest sycophant; do
    D=$M/pair$P/$ROLE
    [ -f $D/final/model.safetensors ] || $PY exp7/scripts/upstream_train.py $P $ROLE exp7/data/pair$P.json $D
    [ -f $M/truth/pair${P}_$ROLE.json ] || $PY exp7/scripts/upstream_eval.py $D/final exp7/data/pair$P.json $M/truth/pair${P}_$ROLE.json
  done
  [ -f $M/truth/pair${P}_base.json ] || $PY exp7/scripts/upstream_eval.py $SYCMID_BASE exp7/data/pair$P.json $M/truth/pair${P}_base.json
done
for S in 1 2; do for ROLE in honest sycophant; do
  D=$M/pair100_seed$S/$ROLE
  [ -f $D/final/model.safetensors ] || SYCMID_SEED=$S $PY exp7/scripts/upstream_train.py 100 $ROLE exp7/data/pair100.json $D
  [ -f $M/truth/pair100_seed${S}_$ROLE.json ] || $PY exp7/scripts/upstream_eval.py $D/final exp7/data/pair100.json $M/truth/pair100_seed${S}_$ROLE.json
done; done
echo "[e1] done $(date)"
