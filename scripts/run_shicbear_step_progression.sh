#!/usr/bin/env bash
# SHIC-comparison bear: step progression (2026-09-03), mirroring the guitar's own
# run_isobbox_T1_analysis.sh pattern. Tracked pixels: 6 points chosen from the mask's own
# extremes (top/bottom/left/right of the bounding box) + 2 mid-body points -- a first pick, not a
# previously-established set like the guitar's "explicit9".
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

OBJ=./demo/shic/shapes/bear.obj
NAME=shic_bear
TRAIN_DIR=experiments/shic_bear/encoder_relational_geodesic_mu3dscale1_T1_seed0_phase2
P1_DIR=experiments/shic_bear/encoder_dfd_crossview_phase1
OUT_ROOT=demo/shic/bear3_relational_geodesic_mu3dscale1_T1_seed0_phase2
MASK=demo/shic/masks/bear3_click_350_300_sam_mask_1.pth
PIX=demo/shic/features_analyze/stride14_squash/bear3_pix_stride14_squash.pth
POSCOLOR=demo/shic/bear3/tracked9_marker_colors.png
MASKBASE=bear3_click_350_300_sam_mask_1
PIXTAG=explicit9
PIXEL_INDICES="20742 76276 96620 139768 182723 203809 286889 319074 1637"

mkdir -p "$OUT_ROOT"

mkdir -p "$OUT_ROOT/step0/pixel_to_vertex_nn" "$OUT_ROOT/step0/color_mesh_from_knn_k10/render"
ln -sf "$(pwd)/$MASK" "$OUT_ROOT/step0/$MASKBASE.pth"

echo "=== step0 (phase-1, before any phase-2 training) ==="
python analysis/pixel_to_vertex_nn.py \
  --mask_path "$MASK" --features_path "$PIX" \
  --vertex_features_path "$P1_DIR/pred_f.pth" \
  --save_dir "$OUT_ROOT/step0/pixel_to_vertex_nn" --top_k 10

python analysis/color_mesh_from_knn.py --obj_path "$OBJ" --img_path demo/shic/images/bear3.png \
  --mask_path "$OUT_ROOT/step0/$MASKBASE.pth" \
  --save_dir "$OUT_ROOT/step0/color_mesh_from_knn_k10" \
  --knn_path "$OUT_ROOT/step0/pixel_to_vertex_nn/${MASKBASE}_pixel_to_vertex_nn.pth" \
  --color_mode position --color_path "$POSCOLOR" --k 10 --pixel_indices $PIXEL_INDICES

python analysis/render_colored_mesh.py --obj_path "$OBJ" \
  --ply_path "$OUT_ROOT/step0/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}.ply" \
  --save_dir "$OUT_ROOT/step0/color_mesh_from_knn_k10/render" --azim_deg 135 --elev_deg 10 \
  --source_image_path "$OUT_ROOT/step0/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png" \
  --click_markers_path "$OUT_ROOT/step0/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_click_markers.pth"

STEPS="25 50 75 100 125 150 175 200 225 250 275 300"

for step in $STEPS; do
  STEPDIR="$OUT_ROOT/step$step"
  mkdir -p "$STEPDIR" "$STEPDIR/pixel_to_vertex_nn" "$STEPDIR/color_mesh_from_knn_k10/render"
  ln -sf "$(pwd)/$MASK" "$STEPDIR/$MASKBASE.pth"
done

CHECKPOINTS=""; PRED_F_OUTPUTS=""
for step in $STEPS; do
  CHECKPOINTS="$CHECKPOINTS,$TRAIN_DIR/encoder_checkpoint_step$step.pth"
  PRED_F_OUTPUTS="$PRED_F_OUTPUTS,$TRAIN_DIR/pred_f_step$step.pth"
done
CHECKPOINTS="${CHECKPOINTS#,}"; PRED_F_OUTPUTS="${PRED_F_OUTPUTS#,}"

echo "=== export_pred_f_l2norm (batch, all 12 checkpoints) ==="
python analysis/export_pred_f_l2norm.py --obj_path "$OBJ" --name "$NAME" \
  --checkpoints "$CHECKPOINTS" --outputs "$PRED_F_OUTPUTS" \
  --width 2304 --n_classes 2304

VERTEX_PATHS=""; OUTPUT_PATHS=""
for step in $STEPS; do
  VERTEX_PATHS="$VERTEX_PATHS,$TRAIN_DIR/pred_f_step$step.pth"
  OUTPUT_PATHS="$OUTPUT_PATHS,$OUT_ROOT/step$step/pixel_to_vertex_nn/${MASKBASE}_pixel_to_vertex_nn.pth"
done
VERTEX_PATHS="${VERTEX_PATHS#,}"; OUTPUT_PATHS="${OUTPUT_PATHS#,}"

echo "=== pixel_to_vertex_nn (batch, all 12 checkpoints) ==="
python analysis/pixel_to_vertex_nn.py \
  --mask_path "$MASK" --features_path "$PIX" \
  --vertex_features_paths "$VERTEX_PATHS" --output_paths "$OUTPUT_PATHS" --top_k 10

KNN_PATHS=""; CMK_SAVE_DIRS=""; PLY_PATHS=""; SAVE_DIRS=""; CLICKED_PATHS=""; MARKERS_PATHS=""
for step in $STEPS; do
  STEPDIR="$OUT_ROOT/step$step"
  KNN_PATHS="$KNN_PATHS,$STEPDIR/pixel_to_vertex_nn/${MASKBASE}_pixel_to_vertex_nn.pth"
  CMK_SAVE_DIRS="$CMK_SAVE_DIRS,$STEPDIR/color_mesh_from_knn_k10"
  PLY_PATHS="$PLY_PATHS,$STEPDIR/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}.ply"
  SAVE_DIRS="$SAVE_DIRS,$STEPDIR/color_mesh_from_knn_k10/render"
  CLICKED_PATHS="$CLICKED_PATHS,$STEPDIR/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png"
  MARKERS_PATHS="$MARKERS_PATHS,$STEPDIR/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_click_markers.pth"
done
KNN_PATHS="${KNN_PATHS#,}"; CMK_SAVE_DIRS="${CMK_SAVE_DIRS#,}"; PLY_PATHS="${PLY_PATHS#,}"
SAVE_DIRS="${SAVE_DIRS#,}"; CLICKED_PATHS="${CLICKED_PATHS#,}"; MARKERS_PATHS="${MARKERS_PATHS#,}"

echo "=== color_mesh_from_knn (batch, all 12 checkpoints) ==="
python analysis/color_mesh_from_knn.py --obj_path "$OBJ" --img_path demo/shic/images/bear3.png \
  --mask_path "$MASK" --knn_paths "$KNN_PATHS" --save_dirs "$CMK_SAVE_DIRS" \
  --color_mode position --color_path "$POSCOLOR" --k 10 --pixel_indices $PIXEL_INDICES

echo "=== render_colored_mesh (batch, all 12 checkpoints) ==="
python analysis/render_colored_mesh.py --obj_path "$OBJ" \
  --ply_paths "$PLY_PATHS" --save_dirs "$SAVE_DIRS" --azim_deg 135 --elev_deg 10 \
  --source_image_paths "$CLICKED_PATHS" --click_markers_paths "$MARKERS_PATHS"

echo "-- build_step_grid --"
python analysis/build_step_grid.py \
  --side_by_side_glob "$OUT_ROOT/step{step}/color_mesh_from_knn_k10/render/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_side_by_side.png" \
  --steps 0,25,50,75,100,125,150,175,200,225,250,275,300 --cols 7 \
  --reference_image "$OUT_ROOT/step0/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png" \
  --reference_label "source (9px)" \
  --output_path "$OUT_ROOT/step_progression_grid.png"

echo "SHIC-BEAR STEP PROGRESSION DONE"
