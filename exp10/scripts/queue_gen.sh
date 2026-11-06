#!/bin/bash
# Exp10 generation evals (screening protocol). Starts when p50c_full32k has finished; runs next to the HF steering eval.
cd /data/behavioral-lens-fast
while pgrep -f "screen_eval.py .*p50c_full32k" > /dev/null; do sleep 120; done
echo "[exp10] start $(date -u)"
R=/data2/ec2-user/models/qwen3.5-0.8b-astra/exp10
for m in p10cdir_p50crecipe p50c_plus_h7off sae_top1_p50crecipe p10c_alpha3 "$@"; do
  [ -f notes/screen/exp10_$m.jsonl ] && continue
  /home/ec2-user/miniconda3/envs/vllm/bin/python scripts/screen_eval.py $R/$m exp10_$m --n 200 --samples 2 --max-tokens 16384 --mem ${MEM:-0.55} 2>&1 | grep -E "^SUMMARY|Error|Traceback" > exp10/notes/gen_$m.log
  echo "[exp10] $m done $(date -u) $(cut -c1-400 exp10/notes/gen_$m.log)"
done
echo "[exp10] all done $(date -u)"
