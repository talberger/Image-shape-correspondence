#!/bin/bash
# small_analyze for every feature-RKD arm that still lacks one (Tal, 2026-08-31).
# argmax_dist0.05 is re-run so it gets the current two-baseline comparison layout
# (its original analysis predates comparison_vs_baseline_T0.01 / _T1).
cd /workspace/bsb/scripts
for SUF in featrkd_soft_dist0.2_ang0 \
           featrkd_argmax_dist0.05_ang0 featrkd_argmax_dist0.1_ang0 \
           featrkd_argmax_dist0.2_ang0 featrkd_argmax_dist0.2_ang0.2; do
  echo "########## small_analyze: $SUF ##########"
  ./small_analyze.sh "$SUF" "$SUF" > /workspace/logs/small_analyze_${SUF}.log 2>&1 || echo "FAILED: $SUF"
  grep -E "coverage \(read T=1\)" /workspace/logs/small_analyze_${SUF}.log || true
done
echo "########## REMAINING ANALYSES DONE ##########"
