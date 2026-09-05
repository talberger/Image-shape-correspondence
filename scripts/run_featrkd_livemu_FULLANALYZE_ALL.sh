#!/bin/bash
# Full analysis battery + comparison vs the no-feature-RKD baseline, for the three 2026-08-29
# live-mu runs. Run AFTER run_featrkd_livemu_ALL.sh (needs each run's step300 output for the
# hard-snap-marker source image). ~24 min per run.
set -e
cd /workspace/bsb/scripts
L=/workspace/logs
for cfg in dist0.2_ang0_fixed dist0.2_ang0_live dist0.2_ang0.2_live; do
  echo "########## $cfg : full analyze ##########"
  ./run_featrkd_${cfg}_full_analyze.sh > $L/featrkd_${cfg}_full_analyze.log 2>&1
  COV=$(grep -E "vertex coverage" $L/featrkd_${cfg}_full_analyze.log | tail -1 | sed -E "s#.*coverage: ([0-9]+/4036) vertices \(([0-9.]+%)\).*#\1 = \2#")
  echo "########## $cfg : compare vs baseline (coverage $COV) ##########"
  ( source /workspace/activate_bsb.sh; cd /workspace/bsb
    python analysis/make_run_vs_baseline_comparison.py \
      --run_label "featRKD $cfg" \
      --run_phase2_dir demo/guitar/runs/rkd_feature/bsb_sd_dino_stride7_featrkd_${cfg}_phase2 \
      --run_phase2_coverage "$COV" \
      --baseline_label "no feature RKD -- BEST" \
      --baseline_phase2_dir demo/guitar/bsb_sd_dino_stride7_gridalign_phase2 \
      --baseline_phase2_coverage "1137/4036 = 28.17%" ) > $L/featrkd_${cfg}_compare.log 2>&1
done
echo "########## FULL ANALYSIS DONE ##########"
