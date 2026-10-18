#!/bin/bash
# Photo features for the stride-7 + gridalign recipe (one image, seconds of GPU work).
# --cpu_postprocess because the full-res 1080x810 x 2304ch tensor OOMs a 16 GB card with both
# models resident.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
OUT=demo/guitar/features_analyze/dino_sd_stride7_gridalign
mkdir -p "$OUT"
time python analysis/extract_dino_sd_features.py \
  --img_path images/guitar_01.png \
  --save_dir "$OUT" \
  --output_path "$OUT/guitar_01_pix_stride7_gridalign.pth" \
  --dino_patch_stride 7 --fix_grid_alignment --cpu_postprocess
