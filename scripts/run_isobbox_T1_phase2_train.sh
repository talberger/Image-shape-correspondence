#!/usr/bin/env bash
# ISOBBOX baseline recipe at --temperature 1.0 instead of 0.01 (Tal, 2026-09-02).
# Phase-1 is reused UNCHANGED: temperature is a phase-2-only knob.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb
time python phase2/train_relational_loss_l2norm.py \
  --obj_path ./meshes/guitar_isotropic_bboxnorm.obj --name guitar_isobbox \
  --init_encoder_dir ./experiments/guitar/encoder_isobbox_dino_sd_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 1.0 --train_seed 0 \
  --geo_table_3d_path ./experiments/guitar/geodesic_tables/guitar_isobbox_geo_table_3d.pth \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_isobbox_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T1_seed0_phase2
