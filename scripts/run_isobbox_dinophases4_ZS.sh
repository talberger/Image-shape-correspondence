#!/bin/bash
# ZS (zero-shot, phase-1 only, no phase-2) analysis for the PHASE-MARGINALIZED photo features
# (2026-09-07, --dino_phases 4). PHOTO SIDE ONLY: the mesh side is untouched, so this reuses the
# isobbox phase-1 pred_f.pth exactly as the baseline ZS run does. The ONLY difference between this
# run and demo/guitar/runs/isobbox/bsb_isobbox_ZS_sdsquash/ is $PIXFEAT.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

OBJ=./meshes/guitar_isotropic_bboxnorm.obj
MASK=demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
PIXFEAT=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare_phase4/guitar_01_pix_stride14_nosdsquare_phase4.pth
POSCOLOR=demo/guitar/segment_masks_pixel270_370/pixel_coloring/guitar_01_click_270_370_sam_mask_2_position_color.png
CHECKER=demo/guitar/colored_mesh_isobbox/guitar_isotropic_bboxnorm_mesh_checker16_color.pth
PRED_F=experiments/guitar/encoder_isobbox_dino_sd_dfd_crossview_phase1/pred_f.pth
MB=guitar_01_click_270_370_sam_mask_2
PIXTAG=explicit9
PIXEL_INDICES="9501 22740 31719 46561 75904 92318 156233 184915 188068"

OUT=demo/guitar/runs/isobbox/bsb_isobbox_ZS_sdsquash_dinophases4/analyze
mkdir -p "$OUT/pixel_to_vertex_nn" "$OUT/color_mesh_from_knn_k10/render" \
         "$OUT/color_mesh_from_soft_mapping" "$OUT/back_projection"

echo "=== [1/5] pixel_to_vertex_nn (top_k=10) ==="
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT/pixel_to_vertex_nn" --top_k 10
KNN="$OUT/pixel_to_vertex_nn/${MB}_pixel_to_vertex_nn.pth"

echo "=== [2/5] mesh coloring k=10 + the 9 tracked pixels ==="
python analysis/color_mesh_from_knn.py --obj_path "$OBJ" --mask_path "$MASK" \
  --save_dir "$OUT/color_mesh_from_knn_k10" --knn_path "$KNN" \
  --color_mode position --color_path "$POSCOLOR" --k 10 --pixel_indices $PIXEL_INDICES
python analysis/render_colored_mesh.py --obj_path "$OBJ" \
  --ply_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}.ply" \
  --save_dir "$OUT/color_mesh_from_knn_k10/render" --azim_deg 90 \
  --source_image_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png" \
  --click_markers_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_click_markers.pth"

echo "=== [3/5] soft mapping (read T=1) -- coverage ==="
python analysis/color_mesh_from_soft_mapping.py --obj_path "$OBJ" --mask_path "$MASK" \
  --features_path "$PIXFEAT" --vertex_features_path "$PRED_F" \
  --save_dir "$OUT/color_mesh_from_soft_mapping" \
  --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10
SOFT="$OUT/color_mesh_from_soft_mapping/${MB}_soft_snap_correspondence.pth"

echo "=== [4/5] back-projection, top-1 KNN ==="
python analysis/color_image_from_knn.py --mask_path "$MASK" --knn_dir "$OUT/pixel_to_vertex_nn" \
  --knn_path "$KNN" --vertex_colors_path "$CHECKER" --k 1 --save_dir "$OUT/back_projection"

echo "=== [5/5] back-projection, soft-snap ==="
python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$OUT/color_mesh_from_soft_mapping" --knn_path "$SOFT" \
  --vertex_colors_path "$CHECKER" --k 1 --save_dir "$OUT/back_projection" \
  --output_path "$OUT/back_projection/${MB}_backproj_guitar_isotropic_bboxnorm_mesh_checker16_color_soft_snap_k1.png"

echo "DINOPHASES4 ZS DONE"
