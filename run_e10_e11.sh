#!/bin/bash
# GPU 7 only, after the RSI Q2 queue (arms, evals, probe readout) has finished: E10 neurons, E11 tuned lens, E5/E4 with tuned, E9 replot.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PY=/tealab-data/yush/envs/lens/bin/python
L=logs/overnight.log
while pgrep -f -- "run_probe_q2[.]sh|run_probe_q2_readout[.]sh" >/dev/null; do sleep 120; done
cp results/components/e5_readable.json results/components/e5_readable_3lens.json
run() { echo "$(date '+%F %T') START $1" >> $L; if env $2 $PY -u -m lens.$1 > logs/$1_e1011.log 2>&1; then echo "$(date '+%F %T') DONE $1" >> $L; else echo "$(date '+%F %T') FAILED $1" >> $L; fi; }
run e10_neurons ""
run e11_tuned_lens ""
run e5_readable "LENS_SET=logit,jlens,rlens,tuned"
run e4_read_u "LENS_SET=logit,jlens,rlens,tuned"
CUDA_VISIBLE_DEVICES= run e9_layer_probe ""
echo "$(date '+%F %T') E10-E11 DONE" >> $L
