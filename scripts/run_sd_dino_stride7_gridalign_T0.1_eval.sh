#!/bin/bash
# Ground-truth eval for the T=0.1 run on the current best recipe.
# The eval itself runs at its DEFAULT temperature 1.0 for every run, deliberately -- 'soft mean
# distance' moves with softmax sharpness and is only comparable at a fixed eval temperature.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
EL=demo/guitar/eval_labels
python analysis/eval_geodesic_error.py \
  --run_label "stride7 gridalign T=0.1" \
  --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth \
  --features_path demo/guitar/features_analyze/dino_sd_stride7_gridalign/guitar_01_pix_stride7_gridalign.pth \
  --vertex_features_path experiments/guitar/encoder_dino_sd_stride7_gridalign_relational_geodesic_mu3dscale1_T0.1_seed0_phase2/pred_f.pth \
  --k_vertices_per_pixel 10 \
  --save_path $EL/eval_stride7_gridalign_T0.1_seed0.pth \
  --plot_path $EL/eval_stride7_gridalign_T0.1_seed0.png
python analysis/compare_eval_runs.py \
  --runs $EL/eval_stride7_gridalign.pth $EL/eval_stride7_gridalign_T0.1_seed0.pth $EL/eval_stride7_gridalign_T0.01.pth $EL/eval_stride7_gridalign_T0.001.pth \
  --labels "T=1 (baseline)" "T=0.1" "T=0.01" "T=0.001" \
  --save_path $EL/eval_compare_stride7_T_sweep_full.png
