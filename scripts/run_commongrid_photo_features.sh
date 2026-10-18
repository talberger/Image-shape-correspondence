#!/bin/bash
# Photo features for both common-grid arms. Arm B keeps SD's zero-padded square; arm C drops the
# padding entirely (true aspect, dims a multiple of 8). Both fuse on the common grid.
# --save_raw_path so analysis/dino_sd_overlay_check.py can verify the two maps line up BEFORE the
# fuse, which is the check Tal asked for.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

B=demo/guitar/features_analyze/dino_sd_commongrid_padded
C=demo/guitar/features_analyze/dino_sd_commongrid_nopad
mkdir -p "$B" "$C"

echo "########## arm B: common-grid fuse, SD still zero-padded ##########"
time python analysis/extract_dino_sd_features.py \
  --img_path images/guitar_01.png --save_dir "$B" \
  --output_path "$B/guitar_01_pix_commongrid_padded.pth" \
  --save_raw_path "$B/guitar_01_pix_commongrid_padded_RAW_pregrid.pth" \
  --fuse_mode common_grid --cpu_postprocess

echo "########## arm C: common-grid fuse, SD unpadded true aspect ##########"
time python analysis/extract_dino_sd_features.py \
  --img_path images/guitar_01.png --save_dir "$C" \
  --output_path "$C/guitar_01_pix_commongrid_nopad.pth" \
  --save_raw_path "$C/guitar_01_pix_commongrid_nopad_RAW_pregrid.pth" \
  --fuse_mode common_grid --sd_frame aspect_nopad --cpu_postprocess

echo "########## overlay agreement check, both arms ##########"
O=demo/guitar/features_analyze/overlay_checks
python analysis/dino_sd_overlay_check.py \
  --raw_pregrid_path "$B/guitar_01_pix_commongrid_padded_RAW_pregrid.pth" \
  --label "arm B: common-grid fuse, SD zero-padded" --save_path $O/overlay_armB_padded.png
python analysis/dino_sd_overlay_check.py --sd_no_padding \
  --raw_pregrid_path "$C/guitar_01_pix_commongrid_nopad_RAW_pregrid.pth" \
  --label "arm C: common-grid fuse, SD unpadded true aspect" --save_path $O/overlay_armC_nopad.png
