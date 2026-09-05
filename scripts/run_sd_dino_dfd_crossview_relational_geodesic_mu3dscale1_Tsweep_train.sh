#!/usr/bin/env bash
# Phase-2 softmax-temperature sweep on top of the promoted DFD cross-view phase-1 checkpoint.
#
# Reproduces the promoted baseline's run_config.json EXACTLY except for --temperature and
# --out_dir -- same convention as the 2026-08-25 T=0.005 run (see PROJECT_NOTES.md's
# softmax-temperature entry). Every value not passed here is an argparse default that already
# matches the baseline's saved run_config.json; --width/--n_classes/--fine_seg_path are passed
# because their defaults do NOT match.
#
# Usage:  bash scripts/run_sd_dino_dfd_crossview_relational_geodesic_mu3dscale1_Tsweep_train.sh 0.01 0.001
set -euo pipefail
cd /workspace/bsb

PHASE1=./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1
MASK=./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
FINESEG=./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth
PIXFEAT=./demo/guitar/features_analyze/dino_sd_poc/guitar_01_pix_dino_sd_features.pth
GEO2D=./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth

for T in "$@"; do
  OUT=./experiments/guitar/encoder_dino_sd_dfd_crossview_relational_geodesic_mu3dscale1_T${T}_phase2
  echo "=============================================================="
  echo "=== phase-2 training, temperature ${T}  ->  ${OUT}"
  echo "=============================================================="
  time python phase2/train_relational_loss_l2norm.py \
    --init_encoder_dir "$PHASE1" \
    --object_mask_path "$MASK" \
    --fine_seg_path "$FINESEG" \
    --pix_dino_features_path "$PIXFEAT" \
    --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 \
    --geo_table_2d_path "$GEO2D" \
    --temperature "$T" \
    --out_dir "$OUT"
done
echo "=== sweep done ==="
