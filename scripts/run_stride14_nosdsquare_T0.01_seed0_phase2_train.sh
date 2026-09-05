#!/usr/bin/env bash
# Phase-2 at T=0.01 on stride 14 (no gridalign) WITHOUT sd_square, seeded (2026-09-01).
# This is the experiment that ISOLATES STRIDE: it matches the run Tal likes on all three axes
# that separated them -- squashed SD framing, stride 14, and (at analysis time) read T=0.01 --
# so the only remaining difference from the stride-7 + gridalign arms of the 2026-08-31 sweep is
# the DINO patch stride. Phase-1 is the stride-14 cross-view DFD checkpoint, reused unchanged.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb
time python phase2/train_relational_loss_l2norm.py \
  --init_encoder_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T0.01_seed0_phase2
