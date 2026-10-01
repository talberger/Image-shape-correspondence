#!/bin/bash
# SHIC-bear, 2026-09-17 (Tal): mainplus (MAIN + anchor + L_dist) with rkd_spatial_dist WEAKENED, so anchor/L_dist/cycle
# become visible. --rkd_spatial_dist_weight scales that term alone (Adam-equivalent to scaling anchor, L_dist and cycle
# by 1/w together). Guitar precedent 2026-09-10: w 0.3 / 0.1. Bear's anchor was weaker (KL 7.5 at w=1), so 0.03 too.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb
OBJ=./demo/shic/shapes/bear.obj; NAME=shic_bear
MASK=./demo/shic/masks/bear3_click_350_300_sam_mask_1.pth
GEO3D=./experiments/shic_bear/geodesic_tables/shic_bear_geo_table_3d.pth
PIX=demo/shic/features_analyze/sdsquare_fuse16/bear3_pix_sdsquare_fuse16.pth
DIST=${DIST:-2.7951e-06}   # 2026-09-17: 2.29e-5 = L_dist at 1/10 of the anchor (bear calibration)
for W in ${WS:-0.3 0.1 0.03}; do
  S=sdsquare_truediag2d_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_anchorsoftT0.01_w3.5e-7_distT1_g${DIST}_sdist_w${W}_T0.01_seed0
  P2=./experiments/shic_bear/encoder_${S}_phase2; RUN=demo/shic/bear3_${S}_phase2
  echo "==================== rkd_spatial_dist weight $W ===================="
  if [ ! -f "$P2/pred_f.pth" ]; then
    python phase2/train_relational_loss_l2norm.py --obj_path "$OBJ" --name "$NAME" \
      --init_encoder_dir ./experiments/shic_bear/encoder_dfd_crossview_phase1 \
      --object_mask_path "$MASK" --fine_seg_path ./demo/shic/bear3/bear3_fine_segments.pth --pix_dino_features_path "$PIX" \
      --width 2304 --n_classes 2304 --geodesic --temperature 0.01 --train_seed 0 --geo_diag_2d 967.21 \
      --geo_table_3d_path "$GEO3D" --geo_table_2d_path ./experiments/shic_bear/geodesic_tables/bear3_click_350_300_sam_mask_1_geo_table_2d_seed0.pth \
      --mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs \
      --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 --cyc_weight 1.676e-05 \
      --anchor_weight 3.5e-07 --anchor_temperature 0.01 --dist_weight $DIST --dist_temperature 1.0 \
      --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5 \
      --rkd_spatial_dist_weight $W --out_dir "$P2"
  fi
  P2=$P2 RUN=$RUN PIX=$PIX READ_T=0.01 LABEL="MAIN + anchor + L_dist, rkd_spatial_dist w=$W" bash scripts/analyze_shicbear_arm.sh
  python analysis/topmatch_geo_extent.py --geo_table_3d_path $GEO3D \
    --knn_paths $RUN/analyze/ours/bear3_click_350_300_sam_mask_1_pixel_to_vertex_nn.pth --labels "sdist_w$W" | tee $RUN/analyze/proportion_extent.txt
done
echo "SWEEP DONE"
