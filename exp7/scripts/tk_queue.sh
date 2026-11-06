#!/bin/bash
# two serial streams so that at most two 270M jobs share the GPU with the exp8 vLLM run
cd /data/behavioral-lens-fast
wait_for() { until [ "$(grep -c "\[$1\] done" exp7/notes/tk_done.log 2>/dev/null)" -ge "$2" ]; do sleep 30; done; }
streamA() {
  wait_for tk_patch 1
  PAIRS=100 bash exp7/scripts/tk_run.sh tk_patch
  bash exp7/scripts/tk_run.sh tk_neurons
}
streamB() {
  wait_for tk_directions 3
  PAIRS=101 bash exp7/scripts/tk_run.sh tk_fig2
  bash exp7/scripts/tk_run.sh tk_sae
}
streamA & streamB & wait
echo "[tk_queue] done $(date -u)" >> exp7/notes/tk_done.log
