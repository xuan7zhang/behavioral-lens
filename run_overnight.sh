#!/bin/bash
# GPU 7 only. Waits for the E3 fits and seg_means, then E3 checks, E4..E8, then packs the direction files.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PY=/tealab-data/yush/envs/lens/bin/python
L=logs/overnight.log
while pgrep -f -- "-m lens[.]e3_fit|-m lens[.]seg_means" >/dev/null; do sleep 60; done
for n in jlens rlens; do [ -f results/lenses/${n}_hs23.pt ] || { echo "$(date +%T) MISSING $n lens; abort" >> $L; exit 1; }; done
for step in e3_check e4_read_u e5_readable e6_directions e7_edit_effect e8_causal; do
  echo "$(date +%T) START $step" >> $L
  if $PY -u -m lens.$step > logs/$step.log 2>&1; then echo "$(date +%T) DONE $step" >> $L; else echo "$(date +%T) FAILED $step (see logs/$step.log)" >> $L; fi
done
tar czf share/lens_loop_directions_$(date +%Y%m%d).tar.gz results/directions results/components/weightdiff.json results/components/e6.json results/data/segments.jsonl README_share.md
echo "$(date +%T) ALL DONE" >> $L
