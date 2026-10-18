#!/bin/bash
# Photo features, stride 14, VARIANT A' (2026-09-03, Tal): sd_square's frame with an EDGE-REPLICATED
# pad instead of zeros. Same scale and content placement as sd_square, so the DINO/SD alignment is
# unchanged -- what changes is only the pad content, to remove the zero-boundary artifact measured
# 2026-09-03 (SD per-column norm 210 -> 143 approaching the pad, 300 on the pad itself).
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
OUT=demo/guitar/features_analyze/dino_sd_stride14_sdpad_replicate
mkdir -p "$OUT"
time python analysis/extract_dino_sd_features.py \
  --img_path images/guitar_01.png \
  --save_dir "$OUT" \
  --output_path "$OUT/guitar_01_pix_stride14_sdpad_replicate.pth" \
  --save_raw_path "$OUT/guitar_01_pix_stride14_sdpad_replicate_RAW_pregrid.pth" \
  --sd_frame square_pad_replicate \
  --cpu_postprocess
