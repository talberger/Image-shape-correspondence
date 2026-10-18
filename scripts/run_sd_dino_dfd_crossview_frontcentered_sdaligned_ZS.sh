#!/bin/bash
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"

# ZS (zero-shot -- right after phase-1, no phase-2 fine-tuning) analysis for the cross-view-batched
# DFD phase-1 encoder (experiments/guitar/encoder_dino_sd_dfd_crossview_phase1/, trained by
# train_encoder_dfd_crossview.py -- see PROJECT_NOTES.md 2026-08-22 entry). Straight path-substitution
# copy of run_sd_dino_dfd_frontcentered_ZS.sh (diffed to confirm nothing else changed) pointed at the
# crossview checkpoint, plus the features_analyze (pca + sim_heatmap) stage -- present in the original
# DFD ZS run's output dir (bsb_sd_dino_dfd_frontcentered_ZS/features_analyze/) but not itself captured
# in that script; bundled here so this one script reproduces the full ZS output structure end to end.

MASK=demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
PIXFEAT=demo/guitar/features_analyze/dino_sd_aligned/guitar_01_pix_dino_sd_features_aligned.pth
POSCOLOR=demo/guitar/segment_masks_pixel270_370/pixel_coloring/guitar_01_click_270_370_sam_mask_2_position_color.png
PRED_F=experiments/guitar/encoder_dino_sd_dfd_crossview_phase1/pred_f.pth
CHECKER=demo/guitar/colored_mesh/guitar_mesh_checker16_color.pth
MASKBASE=guitar_01_click_270_370_sam_mask_2

OUT=demo/guitar/runs/legacy/bsb_sd_dino_dfd_crossview_frontcentered_sdaligned_ZS
BACKPROJ=$OUT/back_projection
mkdir -p "$OUT/pixel_to_vertex_nn" "$OUT/color_mesh_from_soft_mapping" "$BACKPROJ" "$OUT/features_analyze/pca" "$OUT/features_analyze/sim_heatmap"

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

echo "=== features_analyze: PCA (shared basis, image+mesh) ==="
python analysis/pca_feature_compare.py \
  --display both --save_dir "$OUT/features_analyze/pca/" \
  --mask_path "$MASK" --features_path "$PIXFEAT" --vertex_features_path "$PRED_F"

echo "=== features_analyze: sim-heatmap (4 pixel + 4 vertex queries, project's standard ZS query set) ==="
python analysis/feature_similarity_heatmap.py \
  --query_pixels 151 94 571 206 820 360 820 740 --query_vertices 642 1527 1545 2760 \
  --save_dir "$OUT/features_analyze/sim_heatmap/" \
  --mask_path "$MASK" --features_path "$PIXFEAT" --vertex_features_path "$PRED_F"

echo "ALL DONE"
