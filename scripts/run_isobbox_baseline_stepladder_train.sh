#!/bin/bash
# Re-train the CURRENT BASELINE with its step checkpoints kept, so phase-2 can be read step by step.
# 2026-09-08: the baseline dir holds only the final checkpoint, so the "where does the good arm
# cross the bad one" probe has nothing to read. Same command as
# scripts/run_isobbox_sdsquare_T1_phase2_train.sh in every respect (same seed, same inputs) --
# only --out_dir differs, so the BASELINE ITSELF IS NOT TOUCHED. --checkpoint_every 25 is the
# default and was always on; the files were lost after the fact, not never written.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb
time python phase2/train_relational_loss_l2norm.py \
  --obj_path ./meshes/guitar_isotropic_bboxnorm.obj --name guitar_isobbox \
  --init_encoder_dir ./experiments/guitar/encoder_isobbox_dino_sd_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_aligned/guitar_01_pix_dino_sd_features_aligned.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 1.0 --train_seed 0 \
  --checkpoint_every 25 \
  --geo_table_3d_path ./experiments/guitar/geodesic_tables/guitar_isobbox_geo_table_3d.pth \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_isobbox_sdsquare_fuse16_T1_seed0_phase2_stepladder
