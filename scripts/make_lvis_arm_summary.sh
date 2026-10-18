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
# 2026-10-07 speed-up (same commands, same outputs): activation is skipped when the caller already did it (14 s),
# the old mesh-views render is gone (the v2 figure reads <tag>_mesh_views_2d.png; 45 s), and the three independent
# chains (markers | our back-projection | SHIC) run side by side. Each script pays ~15-25 s of imports alone.
# PRECOMPUTED=1: the position colours and SHIC's back-projection are already in $OUT (debug_twostage_instance.sh builds
# them while the training runs) -- they do not depend on the arm, so they are not rebuilt.
# KEEP_PLAIN=1 also keeps the standalone back-projection PNGs (debug_twostage_instance.sh moves them out).
[ -n "${DISC_ROOT:-}" ] || source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

ANN_ID=${ANN_ID:?}; PRED_F=${PRED_F:?}; BASE=${BASE:?}
SHIC_BASE=${SHIC_BASE:-demo/shic/lvis_vs_shic}
TAG=lvis$ANN_ID
D=${RUNS:-data/densepose_lvis/runs}/$TAG   # RUNS (2026-10-04): the kept-intermediates dir of the arm
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

KP=$([ "${KEEP_PLAIN:-0}" = 1 ] && echo "--keep_plain" || echo "")
PRE=${PRECOMPUTED:-0}
(
echo "########## markers + per-point errors (GT / ours / SHIC) ##########"
python analysis/compare_ours_vs_shic_points.py --ann_id "$ANN_ID" --class_name "$CLASS" --run_dir "$D" \
  --our_pred_f "$PRED_F" --obj_path "$OBJ" --name "$NAME" \
  --geodesics_path "$GEO228" --out_dir "$BASE"

echo "########## mesh row: 3D markers + geodesic paths, views chosen per instance (2026-10-05) ##########"
python analysis/render_lvis_markers_2d.py --ann_id "$ANN_ID" --class_name "$CLASS" --obj_path "$OBJ" --base "$BASE"
) > "$OUT/chain_markers.log" 2>&1 &
P_MARK=$!

(
echo "########## OUR back-projection (soft-snap) ##########"
[ "$PRE" = 1 ] && [ -f "$OUT/${TAG}_position_color.png" ] || \
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
  --output_path "$OUT/${TAG}_OURS_soft_backproj.png" --mesh_render_path "$RENDER" $KP
) > "$OUT/chain_ours.log" 2>&1 &
P_OURS=$!

(
echo "########## SHIC back-projection (argmax) ##########"
SHIC_SRC="$SHIC_BASE/backproj_$TAG/shic/${TAG}_SHIC_argmax_backproj_side_by_side.png"
if [ "$PRE" = 1 ] && [ -f "$SHIC_SRC" ]; then
  echo "SHIC back-projection already built in $OUT/shic (PRECOMPUTED=1)"
elif [ -f "$SHIC_SRC" ] && [ "$SHIC_BASE" != "$BASE" ]; then
  cp "$SHIC_SRC" "$OUT/shic/"
  echo "reused SHIC back-projection from $SHIC_SRC (it does not depend on our arm)"
else
  python analysis/shic_infer_correspondence.py --class_name $CLASS \
    --image_path "$D/${TAG}.png" --mask_path "$D/${TAG}_mask.pth" --save_dir "$OUT/shic" --top_k 10
  python analysis/color_image_from_knn.py --mask_path "$D/${TAG}_mask.pth" \
    --knn_dir "$OUT/shic" --knn_path "$OUT/shic/${TAG}_shic_pixel_to_vertex_nn.pth" \
    --vertex_colors_path "$CHK" --source_image_path "$D/${TAG}.png" --k 1 --save_dir "$OUT/shic" \
    --output_path "$OUT/shic/${TAG}_SHIC_argmax_backproj.png" --mesh_render_path "$RENDER" $KP
fi
) > "$OUT/chain_shic.log" 2>&1 &
P_SHIC=$!

FAIL=0
for P in $P_MARK $P_OURS $P_SHIC; do wait $P || FAIL=1; done
cat "$OUT/chain_markers.log" "$OUT/chain_ours.log" "$OUT/chain_shic.log"
[ $FAIL = 0 ] || { echo "ARM SUMMARY FAILED: one of the three chains exited non-zero"; exit 1; }

echo "########## compose ##########"
python analysis/make_lvis_summary_figure.py --ann_id "$ANN_ID" --class_name "$CLASS" --base "$BASE" --run_dir "$D" \
  --output_path "$BASE/SUMMARY_${TAG}.png"
echo "ARM SUMMARY DONE: $BASE/SUMMARY_${TAG}.png"
