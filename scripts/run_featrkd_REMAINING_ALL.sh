#!/bin/bash
# Everything left after the 2026-08-29 live-mu bugfix, STRICTLY SEQUENTIAL (one GPU job at a time --
# the previous attempt OOMed on a 20 GB card because a verification run was sharing the GPU).
#   1. the fixed-mu control's full analysis (it OOMed) + its comparison vs baseline
#   2. the two corrected LIVE-mu arms, end to end
set -e
cd /workspace/bsb/scripts
L=/workspace/logs

echo "########## dist0.2_ang0_fixed : full analyze (rerun after OOM) ##########"
./run_featrkd_dist0.2_ang0_fixed_full_analyze.sh > $L/featrkd_dist0.2_ang0_fixed_full_analyze.log 2>&1
COV=$(grep -E "vertex coverage" $L/featrkd_dist0.2_ang0_fixed_full_analyze.log | tail -1 | sed -E "s#.*coverage: ([0-9]+/4036) vertices \(([0-9.]+%)\).*#\1 = \2#")
echo "########## dist0.2_ang0_fixed : compare vs baseline (coverage $COV) ##########"
( source /workspace/activate_bsb.sh; cd /workspace/bsb
  python analysis/make_run_vs_baseline_comparison.py \
    --run_label "featRKD distance 0.2, angle 0 -- FIXED mu (control)" \
    --run_phase2_dir demo/guitar/runs/rkd_feature/bsb_sd_dino_stride7_featrkd_dist0.2_ang0_fixed_phase2 \
    --run_phase2_coverage "$COV" \
    --baseline_label "no feature RKD -- BEST" \
    --baseline_phase2_dir demo/guitar/bsb_sd_dino_stride7_gridalign_phase2 \
    --baseline_phase2_coverage "1137/4036 = 28.17%" ) > $L/featrkd_dist0.2_ang0_fixed_compare.log 2>&1

echo "########## handing off to the corrected live-mu arms ##########"
./run_featrkd_livemu_FIXED_ALL.sh
echo "########## REMAINING ALL DONE ##########"
