#!/usr/bin/env bash
# The two IN-SEGMENT-ONLY pairwise-angle arms (Tal, 2026-09-04): the pair-angle term sees only the
# 8 CLOSE (in-segment) neighbours instead of all 16. The spatial relational loss still uses all 16,
# so each arm differs from the spatial-only control by the pair-angle term ALONE.
# Weight stays 0.01: the in-segment probe measured featPA raw = 0.030561 vs the all-16 probe's
# 0.032117 (only -4.8%), so 0.01 gives 0.67x the spatial term vs the previous arms' 0.70x -- these
# runs therefore differ from the first two by WHICH NEIGHBOURS and nothing else.
# Probe log: ${DISC_WS:-/workspace}/logs/isobbox_pairangle_inseg_probe.log
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd ${DISC_ROOT:-/workspace/disc}
W=0.01
for MODE in attached detachfeat; do
  ./scripts/run_isobbox_pairangle_phase2_train.sh "$W" "$MODE" in_segment
done
echo "PAIRANGLE INSEG BOTH ARMS DONE"
