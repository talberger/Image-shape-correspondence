#!/bin/bash
# Waits for the running soft-student work, then runs small_analyze on each new arm in turn.
set -e
cd ${DISC_ROOT:-/workspace/disc}/scripts
while pgrep -f "run_stride14_featrkd_soft_(beta_high|FEATONLY)" >/dev/null; do sleep 20; done
for SUF in featrkd_soft_dist0.5_ang0 featrkd_soft_dist1.0_ang0 featrkd_soft_featonly_dist1.0_ang0; do
  echo "########## small_analyze: $SUF ##########"
  ./small_analyze.sh "$SUF" "$SUF" > ${DISC_WS:-/workspace}/logs/small_analyze_${SUF}.log 2>&1 || echo "FAILED: $SUF"
  tail -20 ${DISC_WS:-/workspace}/logs/small_analyze_${SUF}.log | grep -E "AUC|BASELINE|^  [a-z]" || true
done
echo "########## ALL SMALL ANALYZES DONE ##########"
