#!/usr/bin/env bash
# Phase-2 for the stride-7 + gridalign validation run. Reproduces the promoted baseline's
# run_config.json exactly except --init_encoder_dir, --pix_dino_features_path and --out_dir.
# NOTE: unlike the fixA original this sources activate_bsb.sh, so it works when called from a
# driver script rather than from an already-activated interactive shell (the original failed with
# ModuleNotFoundError: kaolin when run that way).
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb
time python phase2/train_relational_loss_l2norm.py \
  --init_encoder_dir ./experiments/guitar/encoder_dino_sd_stride7_gridalign_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_stride7_gridalign/guitar_01_pix_stride7_gridalign.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_dino_sd_stride7_gridalign_relational_geodesic_mu3dscale1_phase2
