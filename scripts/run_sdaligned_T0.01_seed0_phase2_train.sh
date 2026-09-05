#!/usr/bin/env bash
# stride 14 + sd_square, T=0.01 -- Tal's request 2026-08-30.
# Purpose: the run he liked visually (dfd_crossview T=0.01 back-projection) was made on the
# PRE-sd_square photo features. This is the identical recipe WITH the fix, so the pair
# isolates sd_square at T=0.01 exactly the way the T=1 pair already does.
# Copy of run_sd_dino_dfd_crossview_sdaligned_relational_geodesic_mu3dscale1_phase2_train.sh
# with only --temperature, --train_seed and --out_dir added/changed.
# NOTE: --train_seed 0 makes THIS run reproducible; the run it is compared against predates the
# flag and is a single unseeded draw.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb
time python phase2/train_relational_loss_l2norm.py \
  --init_encoder_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_aligned/guitar_01_pix_dino_sd_features_aligned.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_sdaligned_T0.01_seed0_phase2
