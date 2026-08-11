#!/bin/bash
set -e
source /workspace/activate_bsb.sh

TRAIN_DIR=experiments/guitar/encoder_dino_mapping_repel_wide300
OUT_ROOT=demo/guitar/bsb_mapping_loss_repel_wide300
MASK=demo/guitar/bsb_origin/guitar_01_click_130_140_sam_mask_2.pth
PIXFEAT=demo/guitar/bsb_origin/guitar_01_pix_dino_features.pth
POSCOLOR=demo/guitar/bsb_retrained/pixel_coloring/guitar_01_click_130_140_sam_mask_2_position_color.png

mkdir -p "$OUT_ROOT"

# step 0 = phase-1 checkpoint, before any mapping-loss training. Identical
# starting point to the earlier repel-150 run (both resume from the same
# phase-1 checkpoint), so reuse that run's already-rendered, single-view
# step0 wholesale rather than recomputing.
cp -r demo/guitar/bsb_mapping_loss_repel/step0 "$OUT_ROOT/step0"

# Trained every 5 steps (checkpoint_every=5, training itself is ~seconds so this
# is free) but only every 3rd saved checkpoint gets analyzed/rendered (= every
# 15 steps) -- the render/analysis chain, not training, is the actual cost
# (~2 min/checkpoint), so this keeps a fine-grained loss/drift curve for free
# while paying for renders only where they're looked at.
for step in 15 30 45 60 75 90 105 120 135 150 165 180 195 210 225 240 255 270 285 300; do
  echo "=== step $step ==="
  STEPDIR="$OUT_ROOT/step$step"
  mkdir -p "$STEPDIR" "$STEPDIR/pixel_to_vertex_nn" "$STEPDIR/color_mesh_from_knn_k10/render"
  ln -sf "$(pwd)/$MASK" "$STEPDIR/guitar_01_click_130_140_sam_mask_2.pth"
  ln -sf "$(pwd)/$PIXFEAT" "$STEPDIR/guitar_01_pix_dino_features.pth"

  PRED_F="$TRAIN_DIR/pred_f_step$step.pth"
  echo "-- export_pred_f --"
  python analysis/export_pred_f.py \
    --checkpoint "$TRAIN_DIR/encoder_checkpoint_step$step.pth" \
    --output "$PRED_F"

  echo "-- pixel_to_vertex_nn --"
  python analysis/pixel_to_vertex_nn.py \
    --mask_path "$STEPDIR/guitar_01_click_130_140_sam_mask_2.pth" \
    --features_path "$STEPDIR/guitar_01_pix_dino_features.pth" \
    --vertex_features_path "$PRED_F" \
    --save_dir "$STEPDIR/pixel_to_vertex_nn" \
    --top_k 10

  echo "-- color_mesh_from_knn --"
  python analysis/color_mesh_from_knn.py \
    --mask_path "$STEPDIR/guitar_01_click_130_140_sam_mask_2.pth" \
    --save_dir "$STEPDIR/color_mesh_from_knn_k10" \
    --knn_path "$STEPDIR/pixel_to_vertex_nn/guitar_01_click_130_140_sam_mask_2_pixel_to_vertex_nn.pth" \
    --color_mode position \
    --color_path "$POSCOLOR" \
    --k 10 \
    --num_pixels 7 \
    --seed 152

  PLY="$STEPDIR/color_mesh_from_knn_k10/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152.ply"
  CLICKED="$STEPDIR/color_mesh_from_knn_k10/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_clicked_pixels.png"
  MARKERS="$STEPDIR/color_mesh_from_knn_k10/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_click_markers.pth"

  echo "-- render_colored_mesh --"
  python analysis/render_colored_mesh.py \
    --ply_path "$PLY" \
    --save_dir "$STEPDIR/color_mesh_from_knn_k10/render" \
    --azim_deg 90 \
    --source_image_path "$CLICKED" \
    --click_markers_path "$MARKERS"

  echo "=== step $step done ==="
done

echo "-- build_step_grid --"
python analysis/build_step_grid.py \
  --side_by_side_glob "$OUT_ROOT/step{step}/color_mesh_from_knn_k10/render/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_side_by_side.png" \
  --steps 0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300 --cols 7 \
  --reference_image "$OUT_ROOT/step0/color_mesh_from_knn_k10/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_clicked_pixels.png" \
  --output_path "$OUT_ROOT/step_progression_grid.png"

echo "ALL STEPS DONE"
