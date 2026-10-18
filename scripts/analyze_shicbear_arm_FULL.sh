#!/bin/bash
# FULL analysis for a SHIC-bear arm (2026-09-17, Tal): the SMALL analysis (analyze_shicbear_arm.sh) plus
# mesh colorings k=1/10/100 + soft-snap + the 9 tracked pixels (rendered at the photo's azim 135),
# hard-snap markers, the step-progression grid (step0 = phase-1 ZS .. step300) and the proportion
# extent diagnostic.
# Env: P1 (phase-1 dir, for step0), P2 (phase-2 dir with step checkpoints), RUN (demo dir), PIX, READ_T, LABEL.
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

bash scripts/analyze_shicbear_arm.sh

OBJ=./demo/shic/shapes/bear.obj
NAME=shic_bear
MASK=demo/shic/masks/bear3_click_350_300_sam_mask_1.pth
POSCOLOR=demo/shic/bear3/bear3_click_350_300_sam_mask_1_position_color.png
MARKERCOLOR=demo/shic/bear3/tracked9_marker_colors.png
IMG=demo/shic/images/bear3.png
MB=bear3_click_350_300_sam_mask_1
PIXTAG=explicit9
PIXEL_INDICES="20742 76276 96620 139768 182723 203809 286889 319074 1637"
GEO3D=experiments/shic_bear/geodesic_tables/shic_bear_geo_table_3d.pth
OUT=$RUN/analyze
AZ="--azim_deg 135 --elev_deg 10"

echo "########## top_k=100 knn for the k=100 coloring ##########"
mkdir -p "$OUT/ours_k100"
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIX" \
  --vertex_features_path "$P2/pred_f.pth" --save_dir "$OUT/ours_k100" --top_k 100
KNN100="$OUT/ours_k100/${MB}_pixel_to_vertex_nn.pth"

echo "########## mesh coloring k=10 + 9 tracked pixels ##########"
mkdir -p "$OUT/color_mesh_from_knn_k10/render"
python analysis/color_mesh_from_knn.py --obj_path "$OBJ" --img_path "$IMG" --mask_path "$MASK" \
  --save_dir "$OUT/color_mesh_from_knn_k10" --knn_path "$KNN100" \
  --color_mode position --color_path "$MARKERCOLOR" --k 10 --pixel_indices $PIXEL_INDICES
python analysis/render_colored_mesh.py --obj_path "$OBJ" $AZ \
  --ply_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}.ply" \
  --save_dir "$OUT/color_mesh_from_knn_k10/render" \
  --source_image_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png" \
  --click_markers_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_click_markers.pth"

echo "########## mesh coloring k=1/10/100 (position colour) ##########"
for k in 1 10 100; do
  mkdir -p "$OUT/color_mesh_from_knn_k$k/render"
  python analysis/color_mesh_from_knn.py --obj_path "$OBJ" --img_path "$IMG" --mask_path "$MASK" \
    --save_dir "$OUT/color_mesh_from_knn_k$k" --knn_path "$KNN100" \
    --color_mode position --color_path "$POSCOLOR" --k $k
done
SOFT="$OUT/soft_mapping/${MB}_soft_snap_correspondence.pth"
mkdir -p "$OUT/soft_mapping/render"
PLYS="$OUT/color_mesh_from_knn_k1/${MB}_mesh_color_position_k1.ply,$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10.ply"
PLYS="$PLYS,$OUT/color_mesh_from_knn_k100/${MB}_mesh_color_position_k100.ply,$OUT/soft_mapping/${MB}_mesh_color_soft_snap_position_k10.ply"
python analysis/render_colored_mesh.py --obj_path "$OBJ" --ply_paths "$PLYS" $AZ \
  --save_dirs "$OUT/color_mesh_from_knn_k1/render,$OUT/color_mesh_from_knn_k10/render,$OUT/color_mesh_from_knn_k100/render,$OUT/soft_mapping/render" \
  --source_image_paths "$POSCOLOR,$POSCOLOR,$POSCOLOR,$POSCOLOR"

echo "########## hard-snap markers ##########"
python analysis/color_mesh_hard_snap_markers.py --obj_path "$OBJ" --mask_path "$MASK" \
  --soft_snap_correspondence_path "$SOFT" --color_path "$POSCOLOR" --pixel_indices $PIXEL_INDICES
python analysis/render_colored_mesh.py --obj_path "$OBJ" $AZ \
  --ply_path "$OUT/soft_mapping/${MB}_hard_snap_spheres_pix_${PIXTAG}.ply" \
  --save_dir "$OUT/soft_mapping/render" \
  --source_image_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png"

echo "########## step progression (step0 = phase-1 ZS) ##########"
STEPS="25 50 75 100 125 150 175 200 225 250 275 300"
for step in 0 $STEPS; do
  mkdir -p "$RUN/step$step/pixel_to_vertex_nn" "$RUN/step$step/color_mesh_from_knn_k10/render"
done
CKS=""; PFS=""; VPS="$P1/pred_f.pth"; OPS="$RUN/step0/pixel_to_vertex_nn/${MB}_pixel_to_vertex_nn.pth"
for step in $STEPS; do
  CKS="$CKS,$P2/encoder_checkpoint_step$step.pth"; PFS="$PFS,$P2/pred_f_step$step.pth"
  VPS="$VPS,$P2/pred_f_step$step.pth"; OPS="$OPS,$RUN/step$step/pixel_to_vertex_nn/${MB}_pixel_to_vertex_nn.pth"
done
python analysis/export_pred_f_l2norm.py --obj_path "$OBJ" --name "$NAME" \
  --checkpoints "${CKS#,}" --outputs "${PFS#,}" --width 2304 --n_classes 2304
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIX" \
  --vertex_features_paths "$VPS" --output_paths "$OPS" --top_k 10
KNNS=""; CSD=""; PLYS=""; SDS=""; CLK=""; MRK=""
for step in 0 $STEPS; do
  S=$RUN/step$step
  KNNS="$KNNS,$S/pixel_to_vertex_nn/${MB}_pixel_to_vertex_nn.pth"; CSD="$CSD,$S/color_mesh_from_knn_k10"
  PLYS="$PLYS,$S/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}.ply"
  SDS="$SDS,$S/color_mesh_from_knn_k10/render"
  CLK="$CLK,$S/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png"
  MRK="$MRK,$S/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_click_markers.pth"
done
python analysis/color_mesh_from_knn.py --obj_path "$OBJ" --img_path "$IMG" --mask_path "$MASK" \
  --knn_paths "${KNNS#,}" --save_dirs "${CSD#,}" \
  --color_mode position --color_path "$MARKERCOLOR" --k 10 --pixel_indices $PIXEL_INDICES
python analysis/render_colored_mesh.py --obj_path "$OBJ" $AZ \
  --ply_paths "${PLYS#,}" --save_dirs "${SDS#,}" --source_image_paths "${CLK#,}" --click_markers_paths "${MRK#,}"
python analysis/build_step_grid.py \
  --side_by_side_glob "$RUN/step{step}/color_mesh_from_knn_k10/render/${MB}_mesh_color_position_k10_pix_${PIXTAG}_side_by_side.png" \
  --steps 0,25,50,75,100,125,150,175,200,225,250,275,300 --cols 7 \
  --reference_image "$RUN/step0/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png" \
  --reference_label "source (9px)" --output_path "$RUN/step_progression_grid.png"
rm -f "$P2"/pred_f_step*.pth

echo "########## proportion extent (top1 vertex set, fraction of D) ##########"
python analysis/topmatch_geo_extent.py --geo_table_3d_path "$GEO3D" \
  --knn_paths "$OUT/ours/${MB}_pixel_to_vertex_nn.pth" --labels "$LABEL" | tee "$OUT/proportion_extent.txt"
echo "SHIC-BEAR FULL ANALYSIS DONE: $RUN"

echo "########## analysis_summary (10-panel grid) ##########"
python analysis/build_analysis_summary.py --analyze_dir "$OUT" --mask_base "$MB" --soft_dir_name soft_mapping \
  --bp_k1_path "$OUT/back_projection/bear3_OURS_top1_backproj_checker16_uv_k1_withbg_side_by_side.png" \
  --bp_soft_path "$OUT/back_projection/bear3_OURS_soft_backproj_checker16_uv_k1_withbg_side_by_side.png" \
  --knn_split_x 808 --backproj_split_x 750 --hard_snap_split_x 808 \
  --output_path "$RUN/analysis_summary.png"
