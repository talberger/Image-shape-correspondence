#!/bin/bash
set -e
source /workspace/activate_bsb.sh

RUN=experiments/guitar/encoder_dino_relational_geodesic
OUT=demo/guitar/runs/legacy/bsb_relational_geodesic/analyze
MASK=demo/guitar/bsb_origin/guitar_01_click_130_140_sam_mask_2.pth
PIXFEAT=demo/guitar/bsb_origin/guitar_01_pix_dino_features.pth
POSCOLOR=demo/guitar/bsb_retrained/pixel_coloring/guitar_01_click_130_140_sam_mask_2_position_color.png
PRED_F=$RUN/pred_f_step300.pth
CHECKER=demo/guitar/colored_mesh/guitar_mesh_checker16_color.pth

mkdir -p "$OUT" "$OUT/color_mesh_from_knn_k1/render" "$OUT/color_mesh_from_knn_k10/render" \
  "$OUT/color_mesh_from_knn_k100/render" "$OUT/color_mesh_from_soft_mapping/render" "$OUT/back_projection"

echo "=== pixel_to_vertex_nn (top_k=100, full mask) ==="
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT" --top_k 100

KNN="$OUT/guitar_01_click_130_140_sam_mask_2_pixel_to_vertex_nn.pth"

echo "=== color_mesh_from_knn (k=1,10,100, full mask) ==="
for k in 1 10 100; do
  python analysis/color_mesh_from_knn.py --mask_path "$MASK" --save_dir "$OUT/color_mesh_from_knn_k$k" \
    --knn_path "$KNN" --color_mode position --color_path "$POSCOLOR" --k $k
done

echo "=== color_mesh_from_soft_mapping ==="
python analysis/color_mesh_from_soft_mapping.py --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT/color_mesh_from_soft_mapping" \
  --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10

SOFT_CORR="$OUT/color_mesh_from_soft_mapping/guitar_01_click_130_140_sam_mask_2_soft_snap_correspondence.pth"

echo "=== render all 4 colorings (batch) ==="
PLY_PATHS="$OUT/color_mesh_from_knn_k1/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k1.ply"
PLY_PATHS="$PLY_PATHS,$OUT/color_mesh_from_knn_k10/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10.ply"
PLY_PATHS="$PLY_PATHS,$OUT/color_mesh_from_knn_k100/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k100.ply"
PLY_PATHS="$PLY_PATHS,$OUT/color_mesh_from_soft_mapping/guitar_01_click_130_140_sam_mask_2_mesh_color_soft_snap_position_k10.ply"
SAVE_DIRS="$OUT/color_mesh_from_knn_k1/render,$OUT/color_mesh_from_knn_k10/render,$OUT/color_mesh_from_knn_k100/render,$OUT/color_mesh_from_soft_mapping/render"
# --source_image_paths required: render_colored_mesh.py only writes a file at all when EITHER
# --save_rendered_views_grid OR --source_image_path is given -- omitting both (as an earlier
# version of this script did) silently renders into memory and writes nothing to disk. All 4
# colorings are from the FULL mask (no --num_pixels), so the correct pairing is the plain
# position_color.png per render_colored_mesh.py's own docstring, same POSCOLOR for all 4.
SRC="$POSCOLOR,$POSCOLOR,$POSCOLOR,$POSCOLOR"

python analysis/render_colored_mesh.py --ply_paths "$PLY_PATHS" --save_dirs "$SAVE_DIRS" --azim_deg 90 \
  --source_image_paths "$SRC"

echo "=== back-projection (checker16, k=1/10/100 + soft_snap) ==="
for k in 1 10 100; do
  python analysis/color_image_from_knn.py --mask_path "$MASK" --knn_dir "$OUT" --knn_path "$KNN" \
    --vertex_colors_path "$CHECKER" --k $k --save_dir "$OUT/back_projection"
done
python analysis/color_image_from_knn.py --mask_path "$MASK" --knn_dir "$OUT/color_mesh_from_soft_mapping" \
  --knn_path "$SOFT_CORR" --vertex_colors_path "$CHECKER" --k 1 --save_dir "$OUT/back_projection" \
  --output_path "$OUT/back_projection/guitar_01_click_130_140_sam_mask_2_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"

# color_image_from_knn.py always writes its standalone backprojected-photo PNG alongside the
# side-by-side (no flag to suppress just the standalone one) -- deleted here since the
# side-by-side already embeds it (same convention as everywhere else in this project: no
# standalone copy of something a side-by-side already shows).
rm -f "$OUT/back_projection/guitar_01_click_130_140_sam_mask_2_backproj_guitar_mesh_checker16_color_k1.png" \
      "$OUT/back_projection/guitar_01_click_130_140_sam_mask_2_backproj_guitar_mesh_checker16_color_k10.png" \
      "$OUT/back_projection/guitar_01_click_130_140_sam_mask_2_backproj_guitar_mesh_checker16_color_k100.png" \
      "$OUT/back_projection/guitar_01_click_130_140_sam_mask_2_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"

echo "=== hard-snap markers (7 seed-152 pixels, soft-to-hard mapping) ==="
python analysis/color_mesh_hard_snap_markers.py \
  --mask_path "$MASK" \
  --soft_snap_correspondence_path "$SOFT_CORR" \
  --color_path "$POSCOLOR"

python analysis/render_colored_mesh.py \
  --ply_path "$OUT/color_mesh_from_soft_mapping/guitar_01_click_130_140_sam_mask_2_hard_snap_spheres_pix_random7_seed152.ply" \
  --save_dir "$OUT/color_mesh_from_soft_mapping/render" \
  --azim_deg 90 \
  --source_image_path "demo/guitar/runs/legacy/bsb_relational_geodesic/step300/color_mesh_from_knn_k10/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_clicked_pixels.png"

echo "ALL DONE"
