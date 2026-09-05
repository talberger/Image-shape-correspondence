#!/bin/bash
# Sequential chain, no pgrep waits (the 2026-08-31 mistake, repeated and now fixed):
# feature-only training+eval, then small_analyze on the three new soft arms.
cd /workspace/bsb/scripts
echo "########## feature-only run ##########"
./run_stride14_featrkd_soft_FEATONLY.sh || echo "FAILED: featonly"
for SUF in featrkd_soft_dist0.5_ang0 featrkd_soft_dist1.0_ang0 featrkd_soft_featonly_dist1.0_ang0; do
  echo "########## small_analyze: $SUF ##########"
  ./small_analyze.sh "$SUF" "$SUF" > /workspace/logs/small_analyze_${SUF}.log 2>&1 || echo "FAILED: $SUF"
  grep -E "coverage \(read T=1\)" /workspace/logs/small_analyze_${SUF}.log || true
done
echo "########## ALL DONE ##########"
