#!/usr/bin/env bash
# Phase-2 on the SD-FRAME-ALIGNED photo features (2026-08-27 fix: SD now gets the same
# aspect-preserving padded square DINO gets -- see modules/dino_sd_fusion.py::sd_square).
# Reproduces the promoted baseline's run_config.json EXACTLY except --pix_dino_features_path
# and --out_dir, so the fix is the single variable. Temperature left at its 1.0 default.
set -euo pipefail
cd ${DISC_ROOT:-/workspace/disc}

time python phase2/train_relational_loss_l2norm.py \
  --init_encoder_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_aligned/guitar_01_pix_dino_sd_features_aligned.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_sdaligned_relational_geodesic_mu3dscale1_phase2
