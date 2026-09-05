#!/usr/bin/env bash
# ISOBBOX at T=1 with the SD_SQUARE-ALIGNED photo features (Tal, 2026-09-03).
# THE FUSE EXPERIMENT. Only difference vs scripts/run_isobbox_T1_phase2_train.sh is the photo
# features: dino_sd_aligned/..._aligned.pth (aspect-preserving zero-pad for SD, the same frame DINO
# gets) instead of dino_sd_stride14_nosdsquare/... (squash, BUG B, deliberately in the baseline).
# Measured 2026-09-03 right before fuse_dino_sd: DINO-vs-SD saliency IoU@0.5 0.181 -> 0.289,
# correlation 0.290 -> 0.649. Phase-1 is REUSED UNCHANGED and that is correct -- renders are square,
# so sd_square is a bit-identical no-op on them (verified 2026-08-27).
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb
time python phase2/train_relational_loss_l2norm.py \
  --obj_path ./meshes/guitar_isotropic_bboxnorm.obj --name guitar_isobbox \
  --init_encoder_dir ./experiments/guitar/encoder_isobbox_dino_sd_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_stride14_sdpad_replicate/guitar_01_pix_stride14_sdpad_replicate.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 1.0 --train_seed 0 \
  --geo_table_3d_path ./experiments/guitar/geodesic_tables/guitar_isobbox_geo_table_3d.pth \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_isobbox_dino_sd_stride14_sdpadreplicate_relational_geodesic_mu3dscale1_T1_seed0_phase2
