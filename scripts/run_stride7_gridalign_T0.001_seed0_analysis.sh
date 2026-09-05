#!/bin/bash
set -e
source /workspace/activate_bsb.sh

# Step-progression analysis for the cross-view-batched DFD phase-2 run
# (encoder_dino_sd_stride7_gridalign_relational_geodesic_mu3dscale1_T0.001_seed0_phase2, see PROJECT_NOTES.md
# 2026-08-22 entry). Straight path-substitution copy of
# run_sd_dino_dfd_relational_geodesic_mu3dscale1_analysis.sh (diffed to confirm nothing else
# changed) -- TRAIN_DIR/OUT_ROOT point at the crossview run, step0's vertex_features_path points at
# encoder_dino_sd_dfd_crossview_phase1 instead of encoder_dino_sd_dfd_phase1.

TRAIN_DIR=experiments/guitar/encoder_dino_sd_stride7_gridalign_relational_geodesic_mu3dscale1_T0.001_seed0_phase2
OUT_ROOT=demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_gridalign_T0.001_seed0_phase2
MASK=demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
PIXFEAT=demo/guitar/features_analyze/dino_sd_stride7_gridalign/guitar_01_pix_stride7_gridalign.pth
POSCOLOR=demo/guitar/segment_masks_pixel270_370/pixel_coloring/guitar_01_click_270_370_sam_mask_2_position_color.png
MASKBASE=guitar_01_click_270_370_sam_mask_2
PIXTAG=explicit9
PIXEL_INDICES="9501 22740 31719 46561 75904 92318 156233 184915 188068"

mkdir -p "$OUT_ROOT"

# step 0 = phase-1 checkpoint (encoder_dino_sd_dfd_crossview_phase1), before any phase-2 training.
mkdir -p "$OUT_ROOT/step0/pixel_to_vertex_nn" "$OUT_ROOT/step0/color_mesh_from_knn_k10/render"
ln -sf "$(pwd)/$MASK" "$OUT_ROOT/step0/$MASKBASE.pth"
ln -sf "$(pwd)/$PIXFEAT" "$OUT_ROOT/step0/guitar_01_pix_dino_sd_features.pth"

echo "=== step0 (DFD cross-view phase-1, before any phase-2 training) ==="
python analysis/pixel_to_vertex_nn.py \
  --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path experiments/guitar/encoder_dino_sd_stride7_gridalign_dfd_crossview_phase1/pred_f.pth \
  --save_dir "$OUT_ROOT/step0/pixel_to_vertex_nn" --top_k 10

python analysis/color_mesh_from_knn.py \
  --mask_path "$OUT_ROOT/step0/$MASKBASE.pth" \
  --save_dir "$OUT_ROOT/step0/color_mesh_from_knn_k10" \
  --knn_path "$OUT_ROOT/step0/pixel_to_vertex_nn/${MASKBASE}_pixel_to_vertex_nn.pth" \
  --color_mode position --color_path "$POSCOLOR" --k 10 --pixel_indices $PIXEL_INDICES

python analysis/render_colored_mesh.py \
  --ply_path "$OUT_ROOT/step0/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}.ply" \
  --save_dir "$OUT_ROOT/step0/color_mesh_from_knn_k10/render" --azim_deg 90 \
  --source_image_path "$OUT_ROOT/step0/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png" \
  --click_markers_path "$OUT_ROOT/step0/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_click_markers.pth"

STEPS="25 50 75 100 125 150 175 200 225 250 275 300"

for step in $STEPS; do
  STEPDIR="$OUT_ROOT/step$step"
  mkdir -p "$STEPDIR" "$STEPDIR/pixel_to_vertex_nn" "$STEPDIR/color_mesh_from_knn_k10/render"
  ln -sf "$(pwd)/$MASK" "$STEPDIR/$MASKBASE.pth"
  ln -sf "$(pwd)/$PIXFEAT" "$STEPDIR/guitar_01_pix_dino_sd_features.pth"
done

# Stage 1 (batched): one process, mesh/EncoderL2Norm built once, loops over all 12 checkpoints.
CHECKPOINTS=""
PRED_F_OUTPUTS=""
for step in $STEPS; do
  CHECKPOINTS="$CHECKPOINTS,$TRAIN_DIR/encoder_checkpoint_step$step.pth"
  PRED_F_OUTPUTS="$PRED_F_OUTPUTS,$TRAIN_DIR/pred_f_step$step.pth"
done
CHECKPOINTS="${CHECKPOINTS#,}"
PRED_F_OUTPUTS="${PRED_F_OUTPUTS#,}"

echo "=== export_pred_f_l2norm (batch, all 12 checkpoints, mesh+encoder built once) ==="
python analysis/export_pred_f_l2norm.py \
  --checkpoints "$CHECKPOINTS" \
  --outputs "$PRED_F_OUTPUTS" \
  --width 2304 --n_classes 2304

# Stage 2: pixel_to_vertex_nn.py in BATCH MODE.
VERTEX_PATHS=""
OUTPUT_PATHS=""
for step in $STEPS; do
  VERTEX_PATHS="$VERTEX_PATHS,$TRAIN_DIR/pred_f_step$step.pth"
  OUTPUT_PATHS="$OUTPUT_PATHS,$OUT_ROOT/step$step/pixel_to_vertex_nn/${MASKBASE}_pixel_to_vertex_nn.pth"
done
VERTEX_PATHS="${VERTEX_PATHS#,}"
OUTPUT_PATHS="${OUTPUT_PATHS#,}"

echo "=== pixel_to_vertex_nn (batch, all 12 checkpoints, pixel features loaded once) ==="
python analysis/pixel_to_vertex_nn.py \
  --mask_path "$MASK" \
  --features_path "$PIXFEAT" \
  --vertex_features_paths "$VERTEX_PATHS" \
  --output_paths "$OUTPUT_PATHS" \
  --top_k 10

# Stage 3a: color_mesh_from_knn.py per-checkpoint (cheap, --knn_path differs every time).
PLY_PATHS=""
SAVE_DIRS=""
CLICKED_PATHS=""
MARKERS_PATHS=""
for step in $STEPS; do
  echo "=== step $step: color_mesh_from_knn ==="
  STEPDIR="$OUT_ROOT/step$step"

  python analysis/color_mesh_from_knn.py \
    --mask_path "$STEPDIR/$MASKBASE.pth" \
    --save_dir "$STEPDIR/color_mesh_from_knn_k10" \
    --knn_path "$STEPDIR/pixel_to_vertex_nn/${MASKBASE}_pixel_to_vertex_nn.pth" \
    --color_mode position \
    --color_path "$POSCOLOR" \
    --k 10 \
    --pixel_indices $PIXEL_INDICES

  PLY_PATHS="$PLY_PATHS,$STEPDIR/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}.ply"
  SAVE_DIRS="$SAVE_DIRS,$STEPDIR/color_mesh_from_knn_k10/render"
  CLICKED_PATHS="$CLICKED_PATHS,$STEPDIR/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png"
  MARKERS_PATHS="$MARKERS_PATHS,$STEPDIR/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_click_markers.pth"
done
PLY_PATHS="${PLY_PATHS#,}"
SAVE_DIRS="${SAVE_DIRS#,}"
CLICKED_PATHS="${CLICKED_PATHS#,}"
MARKERS_PATHS="${MARKERS_PATHS#,}"

# Stage 3b (batched): one process, mesh/Renderer built once, loops over all 12 .ply files.
echo "=== render_colored_mesh (batch, all 12 checkpoints, mesh+renderer built once) ==="
python analysis/render_colored_mesh.py \
  --ply_paths "$PLY_PATHS" \
  --save_dirs "$SAVE_DIRS" \
  --azim_deg 90 \
  --source_image_paths "$CLICKED_PATHS" \
  --click_markers_paths "$MARKERS_PATHS"

echo "-- build_step_grid --"
python analysis/build_step_grid.py \
  --side_by_side_glob "$OUT_ROOT/step{step}/color_mesh_from_knn_k10/render/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_side_by_side.png" \
  --steps 0,25,50,75,100,125,150,175,200,225,250,275,300 --cols 7 \
  --reference_image "$OUT_ROOT/step0/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png" \
  --output_path "$OUT_ROOT/step_progression_grid.png"

echo "ALL STEPS DONE"
