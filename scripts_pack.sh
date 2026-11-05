#!/bin/bash
# Package E1 directions + E2 dataset for collaborators -> share/lens_loop_E1E2_<date>.tar.gz (+ sha256)
cd "$(dirname "$0")"
mkdir -p share
d=$(date +%Y%m%d)
tar czf share/lens_loop_E1E2_$d.tar.gz configs/config.yaml results/directions/weightdiff_hs16.npy results/directions/weightdiff_hs20.npy \
  results/components/weightdiff.json results/components/weightdiff_sign.json results/data/segments.jsonl results/data/segments_summary.json \
  figs/e1_neurons.png README_share.md
sha256sum share/lens_loop_E1E2_$d.tar.gz | tee share/lens_loop_E1E2_$d.sha256
