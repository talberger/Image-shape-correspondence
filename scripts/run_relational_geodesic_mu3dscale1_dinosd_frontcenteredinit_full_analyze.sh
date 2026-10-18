#!/bin/bash
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"

# Full analysis suite for the DINO+SD fusion run (roadmap item 1, Stage 3). Mirrors
# run_relational_geodesic_mu3dscale1_frontcenteredinit_newmask_full_analyze.sh exactly, except
# RUN/OUT point at the dino_sd run and PIXFEAT is the fused 2304ch real-photo features (must
# match the vertex features' channel dim). pixel_to_vertex_nn/color_mesh_from_soft_mapping/
# color_mesh_hard_snap_markers are all channel-agnostic (no --width/--n_classes needed, unlike
# export_pred_f.py in the step-progression script).

RUN=experiments/guitar/encoder_dino_sd_relational_geodesic_mu3dscale1_frontcenteredinit
OUT=demo/guitar/bsb_relational_geodesic_mu3dscale1_dinosd_frontcenteredinit/analyze
MASK=demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
PIXFEAT=demo/guitar/features_analyze/dino_sd_poc/guitar_01_pix_dino_sd_features.pth
POSCOLOR=demo/guitar/segment_masks_pixel270_370/pixel_coloring/guitar_01_click_270_370_sam_mask_2_position_color.png
PRED_F=$RUN/pred_f_step300.pth
CHECKER=demo/guitar/colored_mesh/guitar_mesh_checker16_color.pth
MASKBASE=guitar_01_click_270_370_sam_mask_2
PIXTAG=explicit9
PIXEL_INDICES="9501 22740 31719 46561 75904 92318 156233 184915 188068"

mkdir -p "$OUT" "$OUT/color_mesh_from_knn_k1/render" "$OUT/color_mesh_from_knn_k10/render" \
  "$OUT/color_mesh_from_knn_k100/render" "$OUT/color_mesh_from_soft_mapping/render" "$OUT/back_projection"

echo "=== pixel_to_vertex_nn (top_k=100, full mask) ==="
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT" --top_k 100

KNN="$OUT/${MASKBASE}_pixel_to_vertex_nn.pth"

echo "=== color_mesh_from_knn (k=1,10,100, full mask) ==="
for k in 1 10 100; do
  python analysis/color_mesh_from_knn.py --mask_path "$MASK" --save_dir "$OUT/color_mesh_from_knn_k$k" \
    --knn_path "$KNN" --color_mode position --color_path "$POSCOLOR" --k $k
done

echo "=== color_mesh_from_soft_mapping ==="
python analysis/color_mesh_from_soft_mapping.py --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT/color_mesh_from_soft_mapping" \
  --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10

SOFT_CORR="$OUT/color_mesh_from_soft_mapping/${MASKBASE}_soft_snap_correspondence.pth"

echo "=== render all 4 colorings (batch) ==="
PLY_PATHS="$OUT/color_mesh_from_knn_k1/${MASKBASE}_mesh_color_position_k1.ply"
PLY_PATHS="$PLY_PATHS,$OUT/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10.ply"
PLY_PATHS="$PLY_PATHS,$OUT/color_mesh_from_knn_k100/${MASKBASE}_mesh_color_position_k100.ply"
PLY_PATHS="$PLY_PATHS,$OUT/color_mesh_from_soft_mapping/${MASKBASE}_mesh_color_soft_snap_position_k10.ply"
SAVE_DIRS="$OUT/color_mesh_from_knn_k1/render,$OUT/color_mesh_from_knn_k10/render,$OUT/color_mesh_from_knn_k100/render,$OUT/color_mesh_from_soft_mapping/render"
SRC="$POSCOLOR,$POSCOLOR,$POSCOLOR,$POSCOLOR"

python analysis/render_colored_mesh.py --ply_paths "$PLY_PATHS" --save_dirs "$SAVE_DIRS" --azim_deg 90 \
  --source_image_paths "$SRC"

echo "=== back-projection (checker16, k=1 + soft_snap only, per Standard analysis suite thinning) ==="
python analysis/color_image_from_knn.py --mask_path "$MASK" --knn_dir "$OUT" --knn_path "$KNN" \
  --vertex_colors_path "$CHECKER" --k 1 --save_dir "$OUT/back_projection"
python analysis/color_image_from_knn.py --mask_path "$MASK" --knn_dir "$OUT/color_mesh_from_soft_mapping" \
  --knn_path "$SOFT_CORR" --vertex_colors_path "$CHECKER" --k 1 --save_dir "$OUT/back_projection" \
  --output_path "$OUT/back_projection/${MASKBASE}_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"

rm -f "$OUT/back_projection/${MASKBASE}_backproj_guitar_mesh_checker16_color_k1.png" \
      "$OUT/back_projection/${MASKBASE}_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"

echo "=== hard-snap markers (9 explicit tracked pixels, soft-to-hard mapping) ==="
python analysis/color_mesh_hard_snap_markers.py \
  --mask_path "$MASK" \
  --soft_snap_correspondence_path "$SOFT_CORR" \
  --color_path "$POSCOLOR" \
  --pixel_indices $PIXEL_INDICES

python analysis/render_colored_mesh.py \
  --ply_path "$OUT/color_mesh_from_soft_mapping/${MASKBASE}_hard_snap_spheres_pix_${PIXTAG}.ply" \
  --save_dir "$OUT/color_mesh_from_soft_mapping/render" \
  --azim_deg 90 \
  --source_image_path "demo/guitar/bsb_relational_geodesic_mu3dscale1_dinosd_frontcenteredinit/step300/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png"

echo "ALL DONE"
