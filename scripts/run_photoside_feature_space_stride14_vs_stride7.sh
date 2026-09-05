#!/bin/bash
# Photo-side feature-space report, stride 14 vs stride 7 (NEXT POD item 2, 2026-08-31).
# The mesh-side version (2026-09-01 section 4) found the two arms nearly identical, which pointed at
# the PHOTO side as the only place a number could still explain the visual preference. This runs the
# same five diagnostics on the per-pixel feature field of each arm, matched on framing (both squash,
# both no sd_square) and read/train temperature (both T=0.01, seed 0) -- stride is the only
# difference. CPU only; safe to run alongside a GPU job.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb

MASK=demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
OUT=demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_gridalign_nosdsquare_T0.01_seed0_phase2/comparison_liked_head_to_head/feature_space_photoside

echo "=== stride 14 (the baseline recipe) ==="
python analysis/feature_space_report.py \
  --pred_f_path experiments/guitar/encoder_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T0.01_seed0_phase2/pred_f.pth \
  --pix_features_path demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth \
  --mask_path "$MASK" \
  --label "stride 14, squash, T=0.01, seed 0" \
  --save_dir "$OUT/stride14"

echo "=== stride 7 + gridalign ==="
python analysis/feature_space_report.py \
  --pred_f_path experiments/guitar/encoder_dino_sd_stride7_gridalign_nosdsquare_relational_geodesic_mu3dscale1_T0.01_seed0_phase2/pred_f.pth \
  --pix_features_path demo/guitar/features_analyze/dino_sd_stride7_gridalign_nosdsquare/guitar_01_pix_stride7_gridalign_nosdsquare.pth \
  --mask_path "$MASK" \
  --label "stride 7 + gridalign, squash, T=0.01, seed 0" \
  --save_dir "$OUT/stride7"

echo "ALL DONE: $OUT"
