#!/bin/bash
# Photo features for stride 7 + gridalign WITHOUT sd_square (2026-08-31, Tal's call).
# Same as run_sd_dino_stride7_gridalign_photo_features.sh except --sd_frame squash: SD is fed the
# raw photo stretched to 512x512, the pre-2026-08-27 framing. Rationale: across the whole
# 2026-08-30 investigation the back-projection Tal judged best is still the one built on the
# squashed framing, and neither proposed explanation for that (edge padding, fusion resolution)
# survived testing -- so the framing goes back to being a knob and gets tested on the current
# stride-7 + gridalign recipe.
# --cpu_postprocess because the full-res 1080x810 x 2304ch tensor OOMs a 16 GB card.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
OUT=demo/guitar/features_analyze/dino_sd_stride7_gridalign_nosdsquare
mkdir -p "$OUT"
time python analysis/extract_dino_sd_features.py \
  --img_path images/guitar_01.png \
  --save_dir "$OUT" \
  --output_path "$OUT/guitar_01_pix_stride7_gridalign_nosdsquare.pth" \
  --save_raw_path "$OUT/guitar_01_pix_stride7_gridalign_nosdsquare_RAW_pregrid.pth" \
  --sd_frame squash \
  --dino_patch_stride 7 --fix_grid_alignment --cpu_postprocess
