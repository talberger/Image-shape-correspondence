#!/bin/bash
# THE CIRCULARITY TEST, 2026-08-29. soft_feature()'s soft-map weights are computed from the very
# image<->vertex similarity the feature-RKD loss is trying to fix, so the loss can be satisfied by
# sharpening the mapping instead of improving features. --feat_detach_weights blocks that path.
#
# Two clean A/B pairs, all --train_seed 0, all fixed mu, all same GPU. Only the detach differs:
#   dist0.2_ang0_fixed    (already on disk)  vs  dist0.2_ang0_detach
#   dist0.2_ang0.2_fixed  (new control)      vs  dist0.2_ang0.2_detach
# READING: if the degradation disappears under detach, circularity was the cause and the direction
# is salvageable. If it persists, the objective itself is wrong for this task -- close it knowingly.
#
# STRICTLY SEQUENTIAL: one GPU job at a time (a shared GPU OOMed the battery earlier today).
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs
EL=demo/guitar/eval_labels
PIXFEAT=demo/guitar/features_analyze/dino_sd_stride7_gridalign/guitar_01_pix_stride7_gridalign.pth
RUNS="dist0.2_ang0_detach dist0.2_ang0.2_fixed dist0.2_ang0.2_detach"

# --- phase 1 of the driver: training + eval only, so the numbers arrive fast ---
for cfg in $RUNS; do
  rm -f $L/featrkd_${cfg}_eval.log            # never let a waiter match a stale log
  echo "########## $cfg : phase-2 ##########"
  ./run_featrkd_${cfg}_phase2_train.sh > $L/featrkd_${cfg}_phase2_train.log 2>&1
  echo "########## $cfg : eval ##########"
  ( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
    python analysis/eval_geodesic_error.py \
      --run_label "featRKD $cfg" \
      --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth \
      --features_path $PIXFEAT \
      --vertex_features_path experiments/guitar/encoder_dino_sd_stride7_gridalign_relational_featrkd_${cfg}_phase2/pred_f.pth \
      --k_vertices_per_pixel 10 \
      --save_path $EL/eval_featrkd_${cfg}.pth \
      --plot_path $EL/eval_featrkd_${cfg}.png ) > $L/featrkd_${cfg}_eval.log 2>&1
done
echo "########## ALL EVALS DONE ##########"

# --- phase 2: the full battery per run ---
for cfg in $RUNS; do
  echo "########## $cfg : step progression ##########"
  ./run_featrkd_${cfg}_analysis.sh     > $L/featrkd_${cfg}_analysis.log 2>&1
  echo "########## $cfg : full analyze ##########"
  ./run_featrkd_${cfg}_full_analyze.sh > $L/featrkd_${cfg}_full_analyze.log 2>&1
  COV=$(grep -E "vertex coverage" $L/featrkd_${cfg}_full_analyze.log | tail -1 | sed -E "s#.*coverage: ([0-9]+/4036) vertices \(([0-9.]+%)\).*#\1 = \2#")
  echo "########## $cfg : compare vs baseline (coverage $COV) ##########"
  ( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
    python analysis/make_run_vs_baseline_comparison.py \
      --run_label "featRKD $cfg" --run_phase2_dir demo/guitar/runs/rkd_feature/bsb_sd_dino_stride7_featrkd_${cfg}_phase2 \
      --run_phase2_coverage "$COV" \
      --baseline_label "no feature RKD -- BEST" \
      --baseline_phase2_dir demo/guitar/bsb_sd_dino_stride7_gridalign_phase2 \
      --baseline_phase2_coverage "1137/4036 = 28.17%" ) > $L/featrkd_${cfg}_compare.log 2>&1
done

echo "########## final comparison ##########"
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_stride7_gridalign.pth \
           $EL/eval_featrkd_dist0.2_ang0_fixed.pth $EL/eval_featrkd_dist0.2_ang0_detach.pth \
           $EL/eval_featrkd_dist0.2_ang0.2_fixed.pth $EL/eval_featrkd_dist0.2_ang0.2_detach.pth \
    --labels "baseline (no feature RKD)" \
             "distance 0.2 -- weights ATTACHED" "distance 0.2 -- weights DETACHED" \
             "distance 0.2 + angle 0.2 -- weights ATTACHED" "distance 0.2 + angle 0.2 -- weights DETACHED" \
    --save_path $EL/eval_compare_featrkd_detach.png ) > $L/featrkd_detach_compare.log 2>&1
echo "########## DETACH ALL DONE ##########"
