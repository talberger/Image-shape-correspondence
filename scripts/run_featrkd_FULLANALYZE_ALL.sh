#!/bin/bash
# Full analysis battery + comparison vs the no-feature-RKD baseline, for all three RKD runs.
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs
for cfg in dist0.2_ang0 dist0_ang0.2 dist0.2_ang0.2; do
  echo "########## $cfg : full analyze ##########"
  ./run_featrkd_${cfg}_full_analyze.sh > $L/featrkd_${cfg}_full_analyze.log 2>&1
  COV=$(grep -E "vertex coverage" $L/featrkd_${cfg}_full_analyze.log | tail -1 | sed -E "s#.*coverage: ([0-9]+/4036) vertices \(([0-9.]+%)\).*#\1 = \2#")
  echo "########## $cfg : compare vs baseline (coverage $COV) ##########"
  ( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
    python analysis/make_run_vs_baseline_comparison.py \
      --run_label "featRKD $cfg" \
      --run_phase2_dir demo/guitar/runs/rkd_feature/bsb_sd_dino_stride7_featrkd_${cfg}_phase2 \
      --run_phase2_coverage "$COV" \
      --baseline_label "no feature RKD -- BEST" \
      --baseline_phase2_dir demo/guitar/bsb_sd_dino_stride7_gridalign_phase2 \
      --baseline_phase2_coverage "1137/4036 = 28.17%" ) > $L/featrkd_${cfg}_compare.log 2>&1
done
echo "########## FULL ANALYSIS DONE ##########"
