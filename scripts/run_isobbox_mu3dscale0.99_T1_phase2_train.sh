#!/usr/bin/env bash
# ISOBBOX, squash, T=1, --mu_3d_scale 0.99 (Tal, 2026-09-03).
# Testing the DERIVED value rather than another blind sweep point. The ratio of the two normalized
# mean distances -- (mean d_3D / mu_3D) / (mean d_2D / mu_2D), the only unitless form of "how far
# apart should the mapped points be" -- measures 0.990 on the T=1 control at step 25 (drift 0.936,
# i.e. still essentially phase-1). Tal's own eye-estimate was ~0.9. This arm is one of three
# (0.99 / 0.95 / 0.9) bracketing both. Expectation on the record: all inside the 1.6-2.4 AUC seed
# floor, since mu_3d_scale=1 beat 0.5 and 2 decisively (section 1 of the 2026-09-03 entry).
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
time python phase2/train_relational_loss_l2norm.py \
  --obj_path ./meshes/guitar_isotropic_bboxnorm.obj --name guitar_isobbox \
  --init_encoder_dir ./experiments/guitar/encoder_isobbox_dino_sd_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 0.99 --temperature 1.0 --train_seed 0 \
  --geo_table_3d_path ./experiments/guitar/geodesic_tables/guitar_isobbox_geo_table_3d.pth \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_isobbox_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale0.99_T1_seed0_phase2
