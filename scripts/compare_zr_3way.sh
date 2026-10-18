#!/bin/bash
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"

# ZR comparison (2026-08-16, Tal's request): three phase-1 ZR checkpoints, no phase-2
# fine-tuning on any of them. Extends scripts/compare_zr_fib1000_vs_retrained.sh (2026-08-15,
# fib1000 vs retrained only) with a third arm: frontcentered -- same original random/Gaussian
# --view_sampling as retrained, but with the --frontview_center dead-flag bug fixed so the
# Gaussian is actually centered on the true front (azim=90) instead of the edge (azim=0).
# Isolates: does fixing just the center (keeping random Gaussian sampling, same std=4, same
# n_views=1000) change anything vs. bsb_retrained's original (buggy-centered) ZR?
# Plain top-1 KNN AND soft-snap back-projection, both at k=1, same as the 2-way version.

MASK=demo/guitar/bsb_origin/guitar_01_click_130_140_sam_mask_2.pth
PIXFEAT=demo/guitar/bsb_origin/guitar_01_pix_dino_features.pth
POSCOLOR=demo/guitar/bsb_retrained/pixel_coloring/guitar_01_click_130_140_sam_mask_2_position_color.png
CHECKER=demo/guitar/colored_mesh/guitar_mesh_checker16_color.pth

SUMMARY=/tmp/zr_3way_summary.txt
> "$SUMMARY"

for RUN in fib1000 retrained frontcentered; do
  if [ "$RUN" = "fib1000" ]; then
    PRED_F=experiments/guitar/encoder_dino_fib1000/pred_f.pth
    OUT=demo/guitar/runs/legacy/bsb_fib1000_ZR
    BACKPROJ=$OUT/back_projection
  elif [ "$RUN" = "retrained" ]; then
    PRED_F=experiments/guitar/encoder_dino/pred_f.pth
    OUT=demo/guitar/bsb_retrained
    BACKPROJ=$OUT/back_projection_ZR
  else
    PRED_F=experiments/guitar/encoder_dino_frontcentered/pred_f.pth
    OUT=demo/guitar/runs/legacy/bsb_frontcentered_ZR
    BACKPROJ=$OUT/back_projection
  fi
  mkdir -p "$OUT/pixel_to_vertex_nn" "$OUT/color_mesh_from_soft_mapping" "$BACKPROJ"

  KNN="$OUT/pixel_to_vertex_nn/guitar_01_click_130_140_sam_mask_2_pixel_to_vertex_nn.pth"
  if [ "$RUN" != "retrained" ] || [ ! -f "$KNN" ]; then
    echo "=== pixel_to_vertex_nn: $RUN ==="
    python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIXFEAT" \
      --vertex_features_path "$PRED_F" --save_dir "$OUT/pixel_to_vertex_nn" --top_k 100
  fi

  echo "=== back-projection (plain top-1 KNN), k=1: $RUN ===" | tee -a "$SUMMARY"
  python analysis/color_image_from_knn.py --mask_path "$MASK" --knn_dir "$OUT/pixel_to_vertex_nn" \
    --knn_path "$KNN" --vertex_colors_path "$CHECKER" --k 1 --save_dir "$BACKPROJ" \
    | tee -a "$SUMMARY"

  echo "=== soft mapping: $RUN ===" | tee -a "$SUMMARY"
  python analysis/color_mesh_from_soft_mapping.py --mask_path "$MASK" --features_path "$PIXFEAT" \
    --vertex_features_path "$PRED_F" --save_dir "$OUT/color_mesh_from_soft_mapping" \
    --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10 \
    | tee -a "$SUMMARY"

  SOFT_CORR="$OUT/color_mesh_from_soft_mapping/guitar_01_click_130_140_sam_mask_2_soft_snap_correspondence.pth"
  echo "=== back-projection (soft-snap), k=1: $RUN ===" | tee -a "$SUMMARY"
  python analysis/color_image_from_knn.py --mask_path "$MASK" \
    --knn_dir "$OUT/color_mesh_from_soft_mapping" --knn_path "$SOFT_CORR" \
    --vertex_colors_path "$CHECKER" --k 1 --save_dir "$BACKPROJ" \
    --output_path "$BACKPROJ/guitar_01_click_130_140_sam_mask_2_backproj_guitar_mesh_checker16_color_soft_snap_k1.png" \
    | tee -a "$SUMMARY"

  rm -f "$BACKPROJ/guitar_01_click_130_140_sam_mask_2_backproj_guitar_mesh_checker16_color_k1.png" \
        "$BACKPROJ/guitar_01_click_130_140_sam_mask_2_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"
done

echo "ALL DONE"
echo "summary saved to $SUMMARY"
