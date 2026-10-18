#!/bin/bash
# Ground-truth evaluation (100 hand-labeled pixel<->vertex pairs) for the stride-7 + gridalign
# validation run, then a four-way comparison against every run it should be judged against.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
EL=demo/guitar/eval_labels

python analysis/eval_geodesic_error.py \
  --run_label "stride7 gridalign (early-exit + fp16)" \
  --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth \
  --features_path demo/guitar/features_analyze/dino_sd_stride7_gridalign/guitar_01_pix_stride7_gridalign.pth \
  --vertex_features_path experiments/guitar/encoder_dino_sd_stride7_gridalign_relational_geodesic_mu3dscale1_phase2/pred_f.pth \
  --k_vertices_per_pixel 10 \
  --save_path $EL/eval_stride7_gridalign.pth \
  --plot_path $EL/eval_stride7_gridalign.png

python analysis/compare_eval_runs.py \
  --runs $EL/geodesic_error_dinov2_currentbest_phase2_v2.pth \
         $EL/eval_sdaligned_phase2.pth \
         $EL/eval_stride7_fixA.pth \
         $EL/eval_stride7_gridalign.pth \
  --labels "baseline s14" "SD-fix s14" "stride7 fixA" "stride7 gridalign" \
  --save_path $EL/eval_compare_stride7_gridalign.png
