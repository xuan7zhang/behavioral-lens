#!/bin/bash
# GPU 7 only. R-lens = reference rules (LessWrong post / vllm-lens PR #36; attention untouched) -> rerun E4-E8 with it.
# The extended-rules variant ("rlens", + attention / DeltaNet rules) is kept as a comparison. Runs after E10/E11.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PY=/tealab-data/yush/envs/lens/bin/python
L=logs/overnight.log
while pgrep -f -- "run_probe_q2[.]sh|run_probe_q2_readout[.]sh|run_e10_e11[.]sh" >/dev/null; do sleep 120; done
mkdir -p results/components/before_rref && cp results/components/e{4_read_u,5_readable,6,7,8}.json results/components/before_rref/ 2>/dev/null
SET=logit,jlens,rlens,rlensref,tuned
[ -f results/lenses/tuned_hs23.pt ] || SET=logit,jlens,rlens,rlensref
run() { echo "$(date '+%F %T') START $1 (rref)" >> $L; if env $2 $PY -u -m lens.$1 > logs/$1_rref.log 2>&1; then echo "$(date '+%F %T') DONE $1 (rref)" >> $L; else echo "$(date '+%F %T') FAILED $1 (rref)" >> $L; fi; }
run e4_read_u "LENS_SET=$SET"
run e5_readable "LENS_SET=$SET"
run e6_directions "LENS_SET=$SET"
run e7_edit_effect "LENS_SET=$SET"
run e8_causal "E8_DIRS=weightdiff,dom,logitlens,jlens,rlens,rlensref"
echo "$(date '+%F %T') RREF DONE" >> $L
