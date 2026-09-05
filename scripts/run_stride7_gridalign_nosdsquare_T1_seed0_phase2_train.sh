#!/usr/bin/env bash
# Phase-2 at T=1 on stride 7 + gridalign WITHOUT sd_square, seeded (2026-08-31, Tal's call).
# Identical to run_stride7_gridalign_T1_seed0_phase2_train.sh except --pix_dino_features_path
# (squashed SD framing) and --out_dir. Phase-1 is REUSED unchanged: sd_square is bit-identical to
# the squashed framing on square inputs and renders are square, so the framing change touches the
# photo features only (verified 2026-08-31, max abs diff 0.0 on a square input).
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb
time python phase2/train_relational_loss_l2norm.py \
  --init_encoder_dir ./experiments/guitar/encoder_dino_sd_stride7_gridalign_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_stride7_gridalign_nosdsquare/guitar_01_pix_stride7_gridalign_nosdsquare.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 1 --train_seed 0 \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_dino_sd_stride7_gridalign_nosdsquare_relational_geodesic_mu3dscale1_T1_seed0_phase2
