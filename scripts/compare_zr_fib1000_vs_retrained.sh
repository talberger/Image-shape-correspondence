#!/bin/bash
set -e
source /workspace/activate_bsb.sh

# ZR comparison (2026-08-15, Tal's request): phase-1 with the new --view_sampling fibonacci
# scheme (encoder_dino_fib1000) vs. the original phase-1 (bsb_retrained), no phase-2 fine-tuning
# on either side. Plain top-1 KNN AND soft-snap back-projection, both at k=1.

MASK=demo/guitar/bsb_origin/guitar_01_click_130_140_sam_mask_2.pth
PIXFEAT=demo/guitar/bsb_origin/guitar_01_pix_dino_features.pth
POSCOLOR=demo/guitar/bsb_retrained/pixel_coloring/guitar_01_click_130_140_sam_mask_2_position_color.png
CHECKER=demo/guitar/colored_mesh/guitar_mesh_checker16_color.pth

for RUN in fib1000 retrained; do
  if [ "$RUN" = "fib1000" ]; then
    PRED_F=experiments/guitar/encoder_dino_fib1000/pred_f.pth
    OUT=demo/guitar/bsb_fib1000_ZR
    BACKPROJ=$OUT/back_projection
  else
    PRED_F=experiments/guitar/encoder_dino/pred_f.pth
    OUT=demo/guitar/bsb_retrained
    BACKPROJ=$OUT/back_projection_ZR
  fi
  mkdir -p "$OUT/pixel_to_vertex_nn" "$OUT/color_mesh_from_soft_mapping" "$BACKPROJ"

  # Reuse bsb_retrained's existing top_k=100 pixel_to_vertex_nn (already computed against this
  # exact ZR, phase-1 only) instead of recomputing -- only fib1000 needs a fresh one.
  KNN="$OUT/pixel_to_vertex_nn/guitar_01_click_130_140_sam_mask_2_pixel_to_vertex_nn.pth"
  if [ "$RUN" = "fib1000" ] || [ ! -f "$KNN" ]; then
    echo "=== pixel_to_vertex_nn: $RUN ==="
    python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIXFEAT" \
      --vertex_features_path "$PRED_F" --save_dir "$OUT/pixel_to_vertex_nn" --top_k 100
  fi

  echo "=== back-projection (plain top-1 KNN), k=1: $RUN ==="
  python analysis/color_image_from_knn.py --mask_path "$MASK" --knn_dir "$OUT/pixel_to_vertex_nn" \
    --knn_path "$KNN" --vertex_colors_path "$CHECKER" --k 1 --save_dir "$BACKPROJ"

  echo "=== soft mapping: $RUN ==="
  python analysis/color_mesh_from_soft_mapping.py --mask_path "$MASK" --features_path "$PIXFEAT" \
    --vertex_features_path "$PRED_F" --save_dir "$OUT/color_mesh_from_soft_mapping" \
    --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10

  SOFT_CORR="$OUT/color_mesh_from_soft_mapping/guitar_01_click_130_140_sam_mask_2_soft_snap_correspondence.pth"
  echo "=== back-projection (soft-snap), k=1: $RUN ==="
  python analysis/color_image_from_knn.py --mask_path "$MASK" \
    --knn_dir "$OUT/color_mesh_from_soft_mapping" --knn_path "$SOFT_CORR" \
    --vertex_colors_path "$CHECKER" --k 1 --save_dir "$BACKPROJ" \
    --output_path "$BACKPROJ/guitar_01_click_130_140_sam_mask_2_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"

  # The side-by-side already embeds the standalone backprojected photo -- never keep both (same
  # convention as scripts/run_relational_geodesic_full_analyze.sh's own back-projection cleanup).
  rm -f "$BACKPROJ/guitar_01_click_130_140_sam_mask_2_backproj_guitar_mesh_checker16_color_k1.png" \
        "$BACKPROJ/guitar_01_click_130_140_sam_mask_2_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"
done

echo "ALL DONE"
