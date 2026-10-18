#!/bin/bash
# The three feature-RKD ablation runs (T=1 throughout, branched from the current best phase-1):
#   dist0.2_ang0    alpha=1 beta=0.2 gamma=0    distance term only
#   dist0_ang0.2    alpha=1 beta=0   gamma=0.2  angle term only
#   dist0.2_ang0.2  alpha=1 beta=0.2 gamma=0.2  both
# beta/gamma chosen from a measured diagnostic, not guessed: the two feature terms sit at ~0.016-0.019
# while the spatial term starts at 0.00325, so 0.2 puts them at ~1x the spatial term at step 1.
# Phase-2 + step progression + eval only; the full analysis battery (~24 min each) is held until
# Tal has seen which runs are worth it.
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs
EL=demo/guitar/eval_labels
for cfg in dist0.2_ang0 dist0_ang0.2 dist0.2_ang0.2; do
  echo "########## $cfg : phase-2 ##########"
  ./run_featrkd_${cfg}_phase2_train.sh > $L/featrkd_${cfg}_phase2_train.log 2>&1
  echo "########## $cfg : step progression ##########"
  ./run_featrkd_${cfg}_analysis.sh     > $L/featrkd_${cfg}_analysis.log 2>&1
  echo "########## $cfg : eval ##########"
  ( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
    python analysis/eval_geodesic_error.py \
      --run_label "featRKD $cfg" \
      --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth \
      --features_path demo/guitar/features_analyze/dino_sd_stride7_gridalign/guitar_01_pix_stride7_gridalign.pth \
      --vertex_features_path experiments/guitar/encoder_dino_sd_stride7_gridalign_relational_featrkd_${cfg}_phase2/pred_f.pth \
      --k_vertices_per_pixel 10 \
      --save_path $EL/eval_featrkd_${cfg}.pth \
      --plot_path $EL/eval_featrkd_${cfg}.png ) > $L/featrkd_${cfg}_eval.log 2>&1
done
echo "########## four-way comparison ##########"
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_stride7_gridalign.pth $EL/eval_featrkd_dist0.2_ang0.pth $EL/eval_featrkd_dist0_ang0.2.pth $EL/eval_featrkd_dist0.2_ang0.2.pth \
    --labels "baseline (no feat RKD)" "beta=0.2 distance" "gamma=0.2 angle" "both 0.2" \
    --save_path $EL/eval_compare_featrkd.png ) > $L/featrkd_compare.log 2>&1
echo "########## ALL DONE ##########"
