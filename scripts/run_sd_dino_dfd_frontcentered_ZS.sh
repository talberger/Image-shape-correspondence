#!/bin/bash
set -e
source /workspace/activate_bsb.sh

# ZS (zero-shot -- right after phase-1, no phase-2 fine-tuning) analysis for the new L2-normalize
# SD-DINO phase-1 encoder (experiments/guitar/encoder_dino_sd_frontcentered_l2norm_phase1/), which fixed
# SD-DINO's spatial-smoothness gap (-0.27 -> -0.80, see PROJECT_NOTES.md 2026-08-19). Mirrors
# scripts/compare_zr_3way.sh's per-run block (top-1 KNN + soft-snap checker16 back-projection),
# but on the current mask/pixel-feature convention (segment_masks_pixel270_370, dino_sd_poc's
# fused 2304ch real-photo features) instead of that script's old bsb_origin/130,140 mask.

MASK=demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
PIXFEAT=demo/guitar/features_analyze/dino_sd_poc/guitar_01_pix_dino_sd_features.pth
POSCOLOR=demo/guitar/segment_masks_pixel270_370/pixel_coloring/guitar_01_click_270_370_sam_mask_2_position_color.png
PRED_F=experiments/guitar/encoder_dino_sd_dfd_phase1/pred_f.pth
CHECKER=demo/guitar/colored_mesh/guitar_mesh_checker16_color.pth
MASKBASE=guitar_01_click_270_370_sam_mask_2

OUT=demo/guitar/runs/legacy/bsb_sd_dino_dfd_frontcentered_ZS
BACKPROJ=$OUT/back_projection
mkdir -p "$OUT/pixel_to_vertex_nn" "$OUT/color_mesh_from_soft_mapping" "$BACKPROJ"

echo "=== pixel_to_vertex_nn (top_k=100) ==="
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT/pixel_to_vertex_nn" --top_k 100

KNN="$OUT/pixel_to_vertex_nn/${MASKBASE}_pixel_to_vertex_nn.pth"

echo "=== back-projection (plain top-1 KNN), k=1 ==="
python analysis/color_image_from_knn.py --mask_path "$MASK" --knn_dir "$OUT/pixel_to_vertex_nn" \
  --knn_path "$KNN" --vertex_colors_path "$CHECKER" --k 1 --save_dir "$BACKPROJ"

echo "=== soft mapping (k_vertices_per_pixel=10) ==="
python analysis/color_mesh_from_soft_mapping.py --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT/color_mesh_from_soft_mapping" \
  --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10

SOFT_CORR="$OUT/color_mesh_from_soft_mapping/${MASKBASE}_soft_snap_correspondence.pth"
echo "=== back-projection (soft-snap), k=1 ==="
python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$OUT/color_mesh_from_soft_mapping" --knn_path "$SOFT_CORR" \
  --vertex_colors_path "$CHECKER" --k 1 --save_dir "$BACKPROJ" \
  --output_path "$BACKPROJ/${MASKBASE}_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"

rm -f "$BACKPROJ/${MASKBASE}_backproj_guitar_mesh_checker16_color_k1.png" \
      "$BACKPROJ/${MASKBASE}_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"

echo "ALL DONE"
