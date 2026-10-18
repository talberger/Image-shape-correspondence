#!/bin/bash
# Photo features, stride 14, VARIANT B (2026-09-03, Tal): the 2026-08-30 decoupled path, never run
# until now. SD is fed its TRUE ASPECT with NO padding at all (dims rounded to a multiple of 8),
# each branch is cropped to its own valid region, and the two are fused on a shared grid that keeps
# SD's resolution instead of downsampling it onto DINO's 16x16.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
OUT=demo/guitar/features_analyze/dino_sd_stride14_B_nopad_cg32
mkdir -p "$OUT"
time python analysis/extract_dino_sd_features.py \
  --img_path images/guitar_01.png \
  --save_dir "$OUT" \
  --output_path "$OUT/guitar_01_pix_stride14_B_nopad_cg32.pth" \
  --save_raw_path "$OUT/guitar_01_pix_stride14_B_nopad_cg32_RAW_pregrid.pth" \
  --sd_frame aspect_nopad --fuse_mode common_grid --common_grid 32 \
  --cpu_postprocess
