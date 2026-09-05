#!/bin/bash
# The two LIVE-mu arms, RE-RUN 2026-08-29 after the .detach() bugfix in
# modules/feature_rkd_loss.py. The first attempt at these collapsed the encoder to a point
# (mu_msh_live 0.740 -> 0.000) because the STUDENT normalizer was detached; the official RKD repo
# leaves it differentiable. Those buggy runs and their outputs were deleted.
# The angle-only arm is NOT here: RKD-A has no normalizer at all, so live mu cannot affect it.
# The fixed-mu control (dist0.2_ang0_fixed) is NOT here either -- already done this pod.
set -e
cd /workspace/bsb/scripts
L=/workspace/logs
EL=demo/guitar/eval_labels
PIXFEAT=demo/guitar/features_analyze/dino_sd_stride7_gridalign/guitar_01_pix_stride7_gridalign.pth

for cfg in dist0.2_ang0_live dist0.2_ang0.2_live; do
  echo "########## $cfg : phase-2 ##########"
  ./run_featrkd_${cfg}_phase2_train.sh > $L/featrkd_${cfg}_phase2_train.log 2>&1
  echo "########## $cfg : step progression ##########"
  ./run_featrkd_${cfg}_analysis.sh     > $L/featrkd_${cfg}_analysis.log 2>&1
  echo "########## $cfg : eval ##########"
  ( source /workspace/activate_bsb.sh; cd /workspace/bsb
    python analysis/eval_geodesic_error.py \
      --run_label "featRKD $cfg" \
      --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth \
      --features_path $PIXFEAT \
      --vertex_features_path experiments/guitar/encoder_dino_sd_stride7_gridalign_relational_featrkd_${cfg}_phase2/pred_f.pth \
      --k_vertices_per_pixel 10 \
      --save_path $EL/eval_featrkd_${cfg}.pth \
      --plot_path $EL/eval_featrkd_${cfg}.png ) > $L/featrkd_${cfg}_eval.log 2>&1
  echo "########## $cfg : full analyze ##########"
  ./run_featrkd_${cfg}_full_analyze.sh > $L/featrkd_${cfg}_full_analyze.log 2>&1
  COV=$(grep -E "vertex coverage" $L/featrkd_${cfg}_full_analyze.log | tail -1 | sed -E "s#.*coverage: ([0-9]+/4036) vertices \(([0-9.]+%)\).*#\1 = \2#")
  echo "########## $cfg : compare vs baseline (coverage $COV) ##########"
  ( source /workspace/activate_bsb.sh; cd /workspace/bsb
    python analysis/make_run_vs_baseline_comparison.py \
      --run_label "featRKD $cfg (distance 0.2, angle ${cfg#dist0.2_ang}, LIVE mu)" \
      --run_phase2_dir demo/guitar/runs/rkd_feature/bsb_sd_dino_stride7_featrkd_${cfg}_phase2 \
      --run_phase2_coverage "$COV" \
      --baseline_label "no feature RKD -- BEST" \
      --baseline_phase2_dir demo/guitar/bsb_sd_dino_stride7_gridalign_phase2 \
      --baseline_phase2_coverage "1137/4036 = 28.17%" ) > $L/featrkd_${cfg}_compare.log 2>&1
done

echo "########## final comparison ##########"
( source /workspace/activate_bsb.sh; cd /workspace/bsb
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_stride7_gridalign.pth \
           $EL/eval_featrkd_dist0.2_ang0_fixed.pth \
           $EL/eval_featrkd_dist0.2_ang0_live.pth \
           $EL/eval_featrkd_dist0.2_ang0.2_live.pth \
    --labels "baseline (no feature RKD)" \
             "distance 0.2, angle 0 -- FIXED mu" \
             "distance 0.2, angle 0 -- LIVE mu" \
             "distance 0.2, angle 0.2 -- LIVE mu" \
    --save_path $EL/eval_compare_featrkd_livemu_fixed.png ) > $L/featrkd_livemu_fixed_compare.log 2>&1
echo "########## ALL DONE ##########"
