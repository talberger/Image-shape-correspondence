#!/usr/bin/env bash
# featRKD dist0.2_ang0.2_live -- RKD-D + RKD-A, LIVE mu. RKD-A itself has no normalizer (cosine is scale-free), so only the distance term is affected
#
# NAMING (2026-08-29, Tal's request): the 2026-08-28 runs were named b<beta>_g<gamma> after the
# paper's symbols, which does not say which term is which. From here: dist<w>_ang<w>.
#   beta  == --rkd_feat_dist_weight  == RKD-D == distance     -> "dist"
#   gamma == --rkd_feat_angle3_weight == RKD-A == angle        -> "ang"
# The 2026-08-28 b/g directories keep their names (never-rename rule).
#
# --train_seed (added 2026-08-29): phase-2's per-step neighbour resampling was UNSEEDED until
# today, so no phase-2 run ever reproduced itself. All three of today's runs use --train_seed 0,
# which makes fixed-vs-live a clean A/B rather than a comparison across two noise draws.
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
time python phase2/train_relational_loss_l2norm.py \
  --init_encoder_dir ./experiments/guitar/encoder_dino_sd_stride7_gridalign_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_stride7_gridalign/guitar_01_pix_stride7_gridalign.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 \
  --rkd_feat_dist_weight 0.2 --rkd_feat_angle3_weight 0.2 --feat_mu_mode live \
  --train_seed 0 \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_dino_sd_stride7_gridalign_relational_featrkd_dist0.2_ang0.2_live_phase2
