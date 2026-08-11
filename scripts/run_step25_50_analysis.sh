#!/usr/bin/env bash
set -euo pipefail
source /workspace/activate_bsb.sh

OUT=experiments/guitar/encoder_dino_mapping_adjacent
ANALYSIS=$OUT/analysis
MASK=./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2.pth
POSITION_SRC=./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2_position_color.png

for S in 25 50; do
  echo "==================== step $S ===================="

  echo "--- export_pred_f step $S ---"
  python analysis/export_pred_f.py \
    --checkpoint $OUT/encoder_checkpoint_step${S}.pth \
    --output $OUT/pred_f_step${S}.pth

  echo "--- pixel_to_vertex_nn step $S ---"
  python analysis/pixel_to_vertex_nn.py \
    --vertex_features_path $OUT/pred_f_step${S}.pth \
    --output_path $ANALYSIS/pixel_to_vertex_nn_step${S}.pth

  echo "--- color_mesh_from_knn (full, position, k10) step $S ---"
  python analysis/color_mesh_from_knn.py \
    --knn_path $ANALYSIS/pixel_to_vertex_nn_step${S}.pth \
    --color_mode position --k 10 \
    --output_path $ANALYSIS/mesh_position_k10_step${S}.ply

  echo "--- color_mesh_from_knn (traced 5px seed0, position, k10) step $S ---"
  python analysis/color_mesh_from_knn.py \
    --knn_path $ANALYSIS/pixel_to_vertex_nn_step${S}.pth \
    --color_mode position --k 10 \
    --num_pixels 5 --seed 0 \
    --output_path $ANALYSIS/traced_5px_seed0_k10_step${S}.ply \
    --clicked_pixels_output_path $ANALYSIS/traced_5px_seed0_k10_step${S}_pixels_on_photo.png \
    --click_markers_output_path $ANALYSIS/traced_5px_seed0_k10_step${S}_markers.pth

  echo "--- render mesh_position_k10 step $S (slow, kaolin JIT) ---"
  python analysis/render_colored_mesh.py \
    --ply_path $ANALYSIS/mesh_position_k10_step${S}.ply \
    --save_dir $ANALYSIS/ \
    --source_image_path $POSITION_SRC

  echo "--- render traced_5px step $S (slow, kaolin JIT) ---"
  python analysis/render_colored_mesh.py \
    --ply_path $ANALYSIS/traced_5px_seed0_k10_step${S}.ply \
    --save_dir $ANALYSIS/ \
    --source_image_path $ANALYSIS/traced_5px_seed0_k10_step${S}_pixels_on_photo.png \
    --click_markers_path $ANALYSIS/traced_5px_seed0_k10_step${S}_markers.pth

  echo "==================== step $S done ===================="
done

echo "ALL DONE"
