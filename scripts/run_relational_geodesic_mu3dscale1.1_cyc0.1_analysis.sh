#!/bin/bash
set -e
source /workspace/activate_bsb.sh

TRAIN_DIR=experiments/guitar/encoder_dino_relational_geodesic_mu3dscale1.1_cyc0.1
OUT_ROOT=demo/guitar/bsb_relational_geodesic_mu3dscale1.1_cyc0.1
MASK=demo/guitar/bsb_origin/guitar_01_click_130_140_sam_mask_2.pth
PIXFEAT=demo/guitar/bsb_origin/guitar_01_pix_dino_features.pth
POSCOLOR=demo/guitar/bsb_retrained/pixel_coloring/guitar_01_click_130_140_sam_mask_2_position_color.png

mkdir -p "$OUT_ROOT"

# step 0 = phase-1 checkpoint, before any training. Identical starting point
# to every other run (all resume from the same phase-1 checkpoint), so reuse
# the already-rendered step0 wholesale rather than recomputing.
cp -r demo/guitar/bsb_mapping_loss_repel/step0 "$OUT_ROOT/step0"

STEPS="25 50 75 100 125 150 175 200 225 250 275 300"

for step in $STEPS; do
  STEPDIR="$OUT_ROOT/step$step"
  mkdir -p "$STEPDIR" "$STEPDIR/pixel_to_vertex_nn" "$STEPDIR/color_mesh_from_knn_k10/render"
  ln -sf "$(pwd)/$MASK" "$STEPDIR/guitar_01_click_130_140_sam_mask_2.pth"
  ln -sf "$(pwd)/$PIXFEAT" "$STEPDIR/guitar_01_pix_dino_features.pth"
done

# Stage 1 (batched): one process, mesh/Encoder built once, loops over all 12
# checkpoints -- see analysis/export_pred_f.py's BATCH MODE docstring note.
CHECKPOINTS=""
PRED_F_OUTPUTS=""
for step in $STEPS; do
  CHECKPOINTS="$CHECKPOINTS,$TRAIN_DIR/encoder_checkpoint_step$step.pth"
  PRED_F_OUTPUTS="$PRED_F_OUTPUTS,$TRAIN_DIR/pred_f_step$step.pth"
done
CHECKPOINTS="${CHECKPOINTS#,}"
PRED_F_OUTPUTS="${PRED_F_OUTPUTS#,}"

echo "=== export_pred_f (batch, all 12 checkpoints, mesh+encoder built once) ==="
python analysis/export_pred_f.py \
  --checkpoints "$CHECKPOINTS" \
  --outputs "$PRED_F_OUTPUTS"

# Stage 2: pixel_to_vertex_nn.py in BATCH MODE -- one load of the 3.6 GB
# pixel-DINO-feature file, matched against all 12 checkpoints in a loop.
VERTEX_PATHS=""
OUTPUT_PATHS=""
for step in $STEPS; do
  VERTEX_PATHS="$VERTEX_PATHS,$TRAIN_DIR/pred_f_step$step.pth"
  OUTPUT_PATHS="$OUTPUT_PATHS,$OUT_ROOT/step$step/pixel_to_vertex_nn/guitar_01_click_130_140_sam_mask_2_pixel_to_vertex_nn.pth"
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

# Stage 3a: color_mesh_from_knn.py still per-checkpoint (its --knn_path genuinely
# differs every time) -- cheap anyway, never the bottleneck. Collect the paths it
# produces for stage 3b's single batched render call below.
PLY_PATHS=""
SAVE_DIRS=""
CLICKED_PATHS=""
MARKERS_PATHS=""
for step in $STEPS; do
  echo "=== step $step: color_mesh_from_knn ==="
  STEPDIR="$OUT_ROOT/step$step"

  python analysis/color_mesh_from_knn.py \
    --mask_path "$STEPDIR/guitar_01_click_130_140_sam_mask_2.pth" \
    --save_dir "$STEPDIR/color_mesh_from_knn_k10" \
    --knn_path "$STEPDIR/pixel_to_vertex_nn/guitar_01_click_130_140_sam_mask_2_pixel_to_vertex_nn.pth" \
    --color_mode position \
    --color_path "$POSCOLOR" \
    --k 10 \
    --num_pixels 7 \
    --seed 152

  PLY_PATHS="$PLY_PATHS,$STEPDIR/color_mesh_from_knn_k10/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152.ply"
  SAVE_DIRS="$SAVE_DIRS,$STEPDIR/color_mesh_from_knn_k10/render"
  CLICKED_PATHS="$CLICKED_PATHS,$STEPDIR/color_mesh_from_knn_k10/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_clicked_pixels.png"
  MARKERS_PATHS="$MARKERS_PATHS,$STEPDIR/color_mesh_from_knn_k10/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_click_markers.pth"
done
PLY_PATHS="${PLY_PATHS#,}"
SAVE_DIRS="${SAVE_DIRS#,}"
CLICKED_PATHS="${CLICKED_PATHS#,}"
MARKERS_PATHS="${MARKERS_PATHS#,}"

# Stage 3b (batched): one process, mesh/Renderer built once, loops over all 12
# .ply files -- see analysis/render_colored_mesh.py's BATCH MODE docstring note.
echo "=== render_colored_mesh (batch, all 12 checkpoints, mesh+renderer built once) ==="
python analysis/render_colored_mesh.py \
  --ply_paths "$PLY_PATHS" \
  --save_dirs "$SAVE_DIRS" \
  --azim_deg 90 \
  --source_image_paths "$CLICKED_PATHS" \
  --click_markers_paths "$MARKERS_PATHS"

echo "-- build_step_grid --"
python analysis/build_step_grid.py \
  --side_by_side_glob "$OUT_ROOT/step{step}/color_mesh_from_knn_k10/render/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_side_by_side.png" \
  --steps 0,25,50,75,100,125,150,175,200,225,250,275,300 --cols 7 \
  --reference_image "$OUT_ROOT/step0/color_mesh_from_knn_k10/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_clicked_pixels.png" \
  --output_path "$OUT_ROOT/step_progression_grid.png"

echo "ALL STEPS DONE"
