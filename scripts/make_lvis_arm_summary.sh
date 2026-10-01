#!/bin/bash
# Everything needed for ONE arm's summary figure on ONE DensePose-LVIS instance, end to end.
#
# Built 2026-09-20 after assembling the LORA arm's figures by hand: the pieces live in five separate
# scripts and the order between them matters (the soft-snap correspondence must exist before the
# back-projection can colour through it, and the marker PLY must exist before it can be rendered),
# so this captures the sequence once instead of re-deriving it per arm.
#
# SHIC's half is reused, not recomputed: SHIC's prediction depends only on the image, not on which
# arm of ours is being scored, so its dense correspondence and back-projection are copied from
# SHIC_BASE if already built there. That keeps the SHIC panel byte-identical across arms.
#
# Env: ANN_ID, PRED_F (the arm's pred_f.pth), BASE (output dir), SHIC_BASE (default the LORA run's).
# Usage:
#   ANN_ID=904 PRED_F=.../pred_f.pth BASE=demo/shic/lvis_vs_shic_nolora bash scripts/make_lvis_arm_summary.sh
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb

ANN_ID=${ANN_ID:?}; PRED_F=${PRED_F:?}; BASE=${BASE:?}
SHIC_BASE=${SHIC_BASE:-demo/shic/lvis_vs_shic}
TAG=lvis$ANN_ID
D=data/densepose_lvis/runs/$TAG
# CLASS (2026-09-22): bear (default, unchanged) | horse | elephant -- see scripts/run_shicclass_setup.sh
CLASS=${CLASS:-bear}
OBJ=./demo/shic/shapes/$CLASS.obj; NAME=shic_$CLASS
if [ "$CLASS" = bear ]; then
  CHK=demo/shic/colored_mesh/bear_shic_normalized_mesh_checker16_uv_color.pth
  RENDER=demo/shic/colored_mesh/checker16_uv_azim135.png
else
  CHK=demo/shic/colored_mesh/${CLASS}_mesh_checker16_uv_color.pth
  RENDER=demo/shic/colored_mesh/${CLASS}_checker16_uv_azim135.png
fi
GEO228=data/densepose_lvis/eval_shic_$CLASS/${CLASS}_heat_geodesics_228.npy
OUT=$BASE/backproj_$TAG
mkdir -p "$BASE" "$OUT" "$OUT/soft_mapping" "$OUT/shic"

echo "########## markers + per-point errors (GT / ours / SHIC) ##########"
python analysis/compare_ours_vs_shic_points.py --ann_id "$ANN_ID" --class_name "$CLASS" --run_dir "$D" \
  --our_pred_f "$PRED_F" --obj_path "$OBJ" --name "$NAME" \
  --geodesics_path "$GEO228" --out_dir "$BASE"

echo "########## mesh views with the markers ##########"
python analysis/render_colored_mesh.py --ply_path "$BASE/${TAG}_markers.ply" --obj_path "$OBJ" \
  --save_dir "$BASE" --output_path "$BASE/${TAG}_mesh_views.png" \
  --render_res 400 --elev_deg 15 --azim_deg 45 135 225 315 --save_rendered_views_grid

echo "########## OUR back-projection (soft-snap) ##########"
python analysis/pixel_coloring.py --img_path "$D/${TAG}.png" --mask_path "$D/${TAG}_mask.pth" \
  --mode position --save_dir "$OUT" --output_path "$OUT/${TAG}_position_color.png"
python analysis/color_mesh_from_soft_mapping.py --mask_path "$D/${TAG}_mask.pth" --img_path "$D/${TAG}.png" \
  --save_dir "$OUT/soft_mapping" --features_path "$D/${TAG}_pix_sdsquare_fuse16.pth" \
  --vertex_features_path "$PRED_F" --obj_path "$OBJ" --name "$NAME" \
  --color_mode position --color_path "$OUT/${TAG}_position_color.png" \
  --k_vertices_per_pixel 10 --temperature 0.01
python analysis/color_image_from_knn.py --mask_path "$D/${TAG}_mask.pth" \
  --knn_dir "$OUT/soft_mapping" \
  --knn_path "$OUT/soft_mapping/${TAG}_mask_soft_snap_correspondence.pth" \
  --vertex_colors_path "$CHK" --source_image_path "$D/${TAG}.png" --k 1 --save_dir "$OUT" \
  --output_path "$OUT/${TAG}_OURS_soft_backproj.png" --mesh_render_path "$RENDER"

echo "########## SHIC back-projection (argmax) ##########"
SHIC_SRC="$SHIC_BASE/backproj_$TAG/shic/${TAG}_SHIC_argmax_backproj_side_by_side.png"
if [ -f "$SHIC_SRC" ] && [ "$SHIC_BASE" != "$BASE" ]; then
  cp "$SHIC_SRC" "$OUT/shic/"
  echo "reused SHIC back-projection from $SHIC_SRC (it does not depend on our arm)"
else
  python analysis/shic_infer_correspondence.py --class_name $CLASS \
    --image_path "$D/${TAG}.png" --mask_path "$D/${TAG}_mask.pth" --save_dir "$OUT/shic" --top_k 10
  python analysis/color_image_from_knn.py --mask_path "$D/${TAG}_mask.pth" \
    --knn_dir "$OUT/shic" --knn_path "$OUT/shic/${TAG}_shic_pixel_to_vertex_nn.pth" \
    --vertex_colors_path "$CHK" --source_image_path "$D/${TAG}.png" --k 1 --save_dir "$OUT/shic" \
    --output_path "$OUT/shic/${TAG}_SHIC_argmax_backproj.png" --mesh_render_path "$RENDER"
fi

echo "########## compose ##########"
python analysis/make_lvis_summary_figure.py --ann_id "$ANN_ID" --class_name "$CLASS" --base "$BASE" \
  --output_path "$BASE/SUMMARY_${TAG}.png"
echo "ARM SUMMARY DONE: $BASE/SUMMARY_${TAG}.png"
