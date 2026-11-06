#!/bin/bash
# Rebuild pairs 100/101/102 on this server (GPU 7 only) with Astra's train.py / eval.py, as in e1_rebuild.sh.
set -e
cd /tealab-data/yush/bl_xuanqi
export SYCMID_BASE=/tealab-data/yush/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3
export HF_HOME=/tealab-data/yush/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=7
PY=/tealab-data/yush/envs/lens/bin/python
M=/tealab-data/yush/exp7_models
for P in 100 101 102; do
  for ROLE in honest sycophant; do
    D=$M/pair$P/$ROLE
    [ -f $D/final/model.safetensors ] || $PY exp7/scripts/upstream_train.py $P $ROLE exp7/data/pair$P.json $D
    [ -f $M/truth/pair${P}_$ROLE.json ] || $PY exp7/scripts/upstream_eval.py $D/final exp7/data/pair$P.json $M/truth/pair${P}_$ROLE.json
  done
  [ -f $M/truth/pair${P}_base.json ] || $PY exp7/scripts/upstream_eval.py $SYCMID_BASE exp7/data/pair$P.json $M/truth/pair${P}_base.json
done
echo "[x7 rebuild] done $(date)"
