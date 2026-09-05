#!/usr/bin/env bash
# Arm B: common-grid fuse, SD framing = padded. Phase-2 at T=0.01 (Tal's choice -- the effect is
# easiest to see visually there), --train_seed 0, from the SHARED common-grid phase-1.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb
time python phase2/train_relational_loss_l2norm.py \
  --init_encoder_dir ./experiments/guitar/encoder_dino_sd_commongrid_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_commongrid_padded/guitar_01_pix_commongrid_padded.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_dino_sd_commongrid_padded_T0.01_seed0_phase2
