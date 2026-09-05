#!/usr/bin/env bash
# SHIC-comparison bear: OUR OWN DINO+SD photo features for bear3.png, stride 14, squash framing
# (2026-09-03) -- matches the guitar baseline's framing choice (scripts/run_sd_dino_stride14_
# nosdsquare_photo_features.sh). bear3.png is 750x808, close to square, so squash's known
# distortion (BUG B) is much milder here than on the guitar's 1080x810 photo.
# --cpu_postprocess: the full-res [750,808,2304] tensor OOMs a 16GB card alongside DINO+SD.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb

OUT=demo/shic/features_analyze/stride14_squash
mkdir -p "$OUT"
time python analysis/extract_dino_sd_features.py \
  --img_path demo/shic/images/bear3.png \
  --save_dir "$OUT" \
  --output_path "$OUT/bear3_pix_stride14_squash.pth" \
  --save_raw_path "$OUT/bear3_pix_stride14_squash_RAW_pregrid.pth" \
  --sd_frame squash \
  --cpu_postprocess
