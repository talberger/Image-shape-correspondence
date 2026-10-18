#!/bin/bash
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"

TRAIN_DIR=experiments/guitar/encoder_dino_mapping_repel
OUT_ROOT=demo/guitar/bsb_mapping_loss_repel
MASK=demo/guitar/bsb_origin/guitar_01_click_130_140_sam_mask_2.pth
PIXFEAT=demo/guitar/bsb_origin/guitar_01_pix_dino_features.pth
POSCOLOR=demo/guitar/bsb_retrained/pixel_coloring/guitar_01_click_130_140_sam_mask_2_position_color.png

mkdir -p "$OUT_ROOT"

# step 0 = phase-1 checkpoint, before any mapping-loss training. Reused (copied,
# not recomputed) from bsb_retrained's existing k10-raw seed-152 render, which
# already is exactly this: phase-1 weights through the same pipeline/markers.
# See PROJECT_NOTES.md 2026-08-10 for why this is safe to reuse as-is.
STEP0_SRC=demo/guitar/bsb_retrained/color_mesh_from_knn_num_pixels7_k10_raw
STEP0_DST="$OUT_ROOT/step0/color_mesh_from_knn_k10"
mkdir -p "$STEP0_DST/render"
cp "$STEP0_SRC"/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152.ply "$STEP0_DST/"
cp "$STEP0_SRC"/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_click_markers.pth "$STEP0_DST/"
cp "$STEP0_SRC"/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_clicked_pixels.png "$STEP0_DST/"
cp "$STEP0_SRC"/render/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_side_by_side.png "$STEP0_DST/render/"

for step in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150; do
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
  --steps 0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150 --cols 4 \
  --reference_image "$OUT_ROOT/step0/color_mesh_from_knn_k10/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_clicked_pixels.png" \
  --output_path "$OUT_ROOT/step_progression_grid.png"

echo "ALL STEPS DONE"
