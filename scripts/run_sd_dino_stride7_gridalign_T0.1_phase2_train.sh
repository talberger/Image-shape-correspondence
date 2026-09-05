#!/usr/bin/env bash
# Phase-2 at T=0.1 on the CURRENT recipe (stride 7 + gridalign), 2026-08-29. Exact copy of the
# T=0.01 run's command with only --temperature and --out_dir changed, so the two are directly
# comparable. NOTE: this run is SEEDED (phase-2 gained --train_seed today, default 0); the T=0.01
# and T=1 runs it is compared against were made before that and are single unseeded draws.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb
time python phase2/train_relational_loss_l2norm.py \
  --init_encoder_dir ./experiments/guitar/encoder_dino_sd_stride7_gridalign_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_stride7_gridalign/guitar_01_pix_stride7_gridalign.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.1 --train_seed 0 \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_dino_sd_stride7_gridalign_relational_geodesic_mu3dscale1_T0.1_seed0_phase2
