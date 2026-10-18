#!/usr/bin/env bash
# Phase-2 at T=0.001 on the current recipe (stride 7 + gridalign), SEEDED (--train_seed 0), 2026-08-30.
# Purpose: the 2026-08-29 T sweep compared three UNSEEDED draws (T=1, 0.01, 0.001) against one
# seeded run (T=0.1), which is not a controlled comparison against the ~2.0 AUC noise floor.
# This re-runs the three missing arms at --train_seed 0 so all four share a seed.
# Identical to run_sd_dino_stride7_gridalign_T0.1_phase2_train.sh except --temperature and --out_dir.
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
time python phase2/train_relational_loss_l2norm.py \
  --init_encoder_dir ./experiments/guitar/encoder_dino_sd_stride7_gridalign_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_stride7_gridalign/guitar_01_pix_stride7_gridalign.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.001 --train_seed 0 \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_dino_sd_stride7_gridalign_relational_geodesic_mu3dscale1_T0.001_seed0_phase2
