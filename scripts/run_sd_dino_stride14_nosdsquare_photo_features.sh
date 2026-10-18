#!/bin/bash
# Photo features for stride 14 (DINO's own patch size, no gridalign) WITHOUT sd_square
# (2026-09-01). This is the framing + stride combination of the run Tal likes
# (dfd_crossview T=0.01, 2026-08-27), regenerated on the current code so it can be trained
# SEEDED and compared arm-to-arm. Its original feature file
# (dino_sd_poc/guitar_01_pix_dino_sd_features.pth) was deleted in an earlier disk cleanup;
# only the 8.6 MB *_RAW_pregrid.pth survives there.
# gridalign is a no-op at stride 14 (the misplacement it fixes only exists when stride != patch
# size), so it is deliberately not passed -- this reproduces the liked run's feature path.
# --cpu_postprocess because the full-res 1080x810 x 2304ch tensor OOMs a 16 GB card.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
OUT=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare
mkdir -p "$OUT"
time python analysis/extract_dino_sd_features.py \
  --img_path images/guitar_01.png \
  --save_dir "$OUT" \
  --output_path "$OUT/guitar_01_pix_stride14_nosdsquare.pth" \
  --save_raw_path "$OUT/guitar_01_pix_stride14_nosdsquare_RAW_pregrid.pth" \
  --sd_frame squash \
  --cpu_postprocess
