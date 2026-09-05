#!/usr/bin/env bash
# The two pairwise-angle arms Tal asked for (2026-09-04), run back to back on the single GPU:
#   1. attached   -- F on the gradient path
#   2. detachfeat -- --feat_detach_features, w is the only route to the encoder
# Weight 0.01 was DERIVED, not guessed: the probe measured featPA raw = 0.032117 against a spatial
# loss of 0.000457 at step 25, so 0.01 puts the angle term at 0.70x the spatial term -- the same
# regime as the argmax-distance arm that worked (1.19x), ~20x below the 3-point angle that
# collapsed (10.9x at gamma=0.2). See /workspace/logs/isobbox_pairangle_probe.log.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /workspace/bsb
W=0.01
for MODE in attached detachfeat; do
  ./scripts/run_isobbox_pairangle_phase2_train.sh "$W" "$MODE"
done
echo "PAIRANGLE BOTH ARMS DONE"
