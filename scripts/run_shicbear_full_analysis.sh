#!/usr/bin/env bash
# SHIC-comparison bear: full analysis for a given phase-2 run (2026-09-03).
# Usage: ./run_shicbear_full_analysis.sh <mu_3d_scale>   e.g. 0.75
# Mirrors the mu=1 baseline's own analysis exactly, same directory convention as the guitar
# (demo/shic/bear3_<run>/analyze/{back_projection,features_analyze/{pca,sim_heatmap},ours,ours_soft,
# soft_mapping}, step0..step300 directly under the run dir).
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

MU=$1
[ -z "$MU" ] && { echo "usage: $0 <mu_3d_scale>"; exit 1; }

OBJ=./demo/shic/shapes/bear.obj
NAME=shic_bear
P2=experiments/shic_bear/encoder_relational_geodesic_mu3dscale${MU}_T1_seed0_phase2
RUN=demo/shic/bear3_relational_geodesic_mu3dscale${MU}_T1_seed0_phase2
MASK=demo/shic/masks/bear3_click_350_300_sam_mask_1.pth
PIX=demo/shic/features_analyze/stride14_squash/bear3_pix_stride14_squash.pth
POSCOLOR=demo/shic/bear3/bear3_click_350_300_sam_mask_1_position_color.png
MARKERCOLOR=demo/shic/bear3/tracked9_marker_colors.png
CHK=demo/shic/colored_mesh/bear_shic_normalized_mesh_checker16_uv_color.pth
RENDER=demo/shic/colored_mesh/checker16_uv_azim135.png
IMG=demo/shic/images/bear3.png
MASKBASE=bear3_click_350_300_sam_mask_1
PIXTAG=explicit9
PIXEL_INDICES="20742 76276 96620 139768 182723 203809 286889 319074 1637"

mkdir -p "$RUN/analyze/back_projection" "$RUN/analyze/ours" "$RUN/analyze/ours_soft" \
         "$RUN/analyze/soft_mapping" "$RUN/analyze/features_analyze/pca" \
         "$RUN/analyze/features_analyze/sim_heatmap" "$RUN/comparison_vs_shic"

echo "########## top1 correspondence ##########"
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIX" \
  --vertex_features_path "$P2/pred_f.pth" --save_dir "$RUN/analyze/ours" --top_k 10

echo "########## top1 back-projection ##########"
python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$RUN/analyze/ours" \
  --knn_path "$RUN/analyze/ours/${MASKBASE}_pixel_to_vertex_nn.pth" \
  --vertex_colors_path "$CHK" --source_image_path "$IMG" \
  --k 1 --save_dir "$RUN/analyze/back_projection" \
  --output_path "$RUN/analyze/back_projection/bear3_OURS_top1_backproj_checker16_uv_k1_withbg.png" \
  --mesh_render_path "$RENDER"
rm -f "$RUN/analyze/back_projection/bear3_OURS_top1_backproj_checker16_uv_k1_withbg.png"

echo "########## soft-snap correspondence ##########"
python analysis/color_mesh_from_soft_mapping.py --mask_path "$MASK" --img_path "$IMG" \
  --save_dir "$RUN/analyze/soft_mapping" \
  --features_path "$PIX" --vertex_features_path "$P2/pred_f.pth" \
  --obj_path "$OBJ" --name "$NAME" \
  --color_mode position --color_path "$POSCOLOR" \
  --k_vertices_per_pixel 10 --temperature 1.0

echo "########## soft back-projection ##########"
python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$RUN/analyze/soft_mapping" \
  --knn_path "$RUN/analyze/soft_mapping/${MASKBASE}_soft_snap_correspondence.pth" \
  --vertex_colors_path "$CHK" --source_image_path "$IMG" \
  --k 1 --save_dir "$RUN/analyze/back_projection" \
  --output_path "$RUN/analyze/back_projection/bear3_OURS_soft_backproj_checker16_uv_k1_withbg.png" \
  --mesh_render_path "$RENDER"
rm -f "$RUN/analyze/back_projection/bear3_OURS_soft_backproj_checker16_uv_k1_withbg.png"

echo "########## PCA ##########"
python analysis/pca_feature_compare.py --obj_path "$OBJ" --display both \
  --save_dir "$RUN/analyze/features_analyze/pca" \
  --mask_path "$MASK" --features_path "$PIX" --vertex_features_path "$P2/pred_f.pth" \
  --azim_deg 135 --elev_deg 10

echo "########## similarity heatmap ##########"
python analysis/feature_similarity_heatmap.py --obj_path "$OBJ" \
  --query_pixels 46 549 193 773 362 239 362 581 --query_vertices 1675 177 242 470 \
  --save_dir "$RUN/analyze/features_analyze/sim_heatmap" --mask_path "$MASK" --features_path "$PIX" \
  --vertex_features_path "$P2/pred_f.pth" --render_res 300 --elev_deg 10 --azim_deg 135

echo "########## step progression ##########"
mkdir -p "$RUN/step0/pixel_to_vertex_nn" "$RUN/step0/color_mesh_from_knn_k10/render"
ln -sf "$(pwd)/$MASK" "$RUN/step0/$MASKBASE.pth"
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIX" \
  --vertex_features_path "experiments/shic_bear/encoder_dfd_crossview_phase1/pred_f.pth" \
  --save_dir "$RUN/step0/pixel_to_vertex_nn" --top_k 10
python analysis/color_mesh_from_knn.py --obj_path "$OBJ" --img_path "$IMG" \
  --mask_path "$RUN/step0/$MASKBASE.pth" \
  --save_dir "$RUN/step0/color_mesh_from_knn_k10" \
  --knn_path "$RUN/step0/pixel_to_vertex_nn/${MASKBASE}_pixel_to_vertex_nn.pth" \
  --color_mode position --color_path "$MARKERCOLOR" --k 10 --pixel_indices $PIXEL_INDICES
python analysis/render_colored_mesh.py --obj_path "$OBJ" \
  --ply_path "$RUN/step0/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}.ply" \
  --save_dir "$RUN/step0/color_mesh_from_knn_k10/render" --azim_deg 135 --elev_deg 10 \
  --source_image_path "$RUN/step0/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png" \
  --click_markers_path "$RUN/step0/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_click_markers.pth"

STEPS="25 50 75 100 125 150 175 200 225 250 275 300"
for step in $STEPS; do
  mkdir -p "$RUN/step$step/pixel_to_vertex_nn" "$RUN/step$step/color_mesh_from_knn_k10/render"
  ln -sf "$(pwd)/$MASK" "$RUN/step$step/$MASKBASE.pth"
done
CHECKPOINTS=""; PRED_F_OUTPUTS=""
for step in $STEPS; do
  CHECKPOINTS="$CHECKPOINTS,$P2/encoder_checkpoint_step$step.pth"
  PRED_F_OUTPUTS="$PRED_F_OUTPUTS,$P2/pred_f_step$step.pth"
done
python analysis/export_pred_f_l2norm.py --obj_path "$OBJ" --name "$NAME" \
  --checkpoints "${CHECKPOINTS#,}" --outputs "${PRED_F_OUTPUTS#,}" --width 2304 --n_classes 2304

VERTEX_PATHS=""; OUTPUT_PATHS=""
for step in $STEPS; do
  VERTEX_PATHS="$VERTEX_PATHS,$P2/pred_f_step$step.pth"
  OUTPUT_PATHS="$OUTPUT_PATHS,$RUN/step$step/pixel_to_vertex_nn/${MASKBASE}_pixel_to_vertex_nn.pth"
done
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIX" \
  --vertex_features_paths "${VERTEX_PATHS#,}" --output_paths "${OUTPUT_PATHS#,}" --top_k 10

KNN_PATHS=""; CMK_SAVE_DIRS=""
for step in $STEPS; do
  KNN_PATHS="$KNN_PATHS,$RUN/step$step/pixel_to_vertex_nn/${MASKBASE}_pixel_to_vertex_nn.pth"
  CMK_SAVE_DIRS="$CMK_SAVE_DIRS,$RUN/step$step/color_mesh_from_knn_k10"
done
python analysis/color_mesh_from_knn.py --obj_path "$OBJ" --img_path "$IMG" \
  --mask_path "$MASK" --knn_paths "${KNN_PATHS#,}" --save_dirs "${CMK_SAVE_DIRS#,}" \
  --color_mode position --color_path "$MARKERCOLOR" --k 10 --pixel_indices $PIXEL_INDICES

PLY_PATHS=""; SAVE_DIRS=""; CLICKED_PATHS=""; MARKERS_PATHS=""
for step in $STEPS; do
  PLY_PATHS="$PLY_PATHS,$RUN/step$step/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}.ply"
  SAVE_DIRS="$SAVE_DIRS,$RUN/step$step/color_mesh_from_knn_k10/render"
  CLICKED_PATHS="$CLICKED_PATHS,$RUN/step$step/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png"
  MARKERS_PATHS="$MARKERS_PATHS,$RUN/step$step/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_click_markers.pth"
done
python analysis/render_colored_mesh.py --obj_path "$OBJ" \
  --ply_paths "${PLY_PATHS#,}" --save_dirs "${SAVE_DIRS#,}" --azim_deg 135 --elev_deg 10 \
  --source_image_paths "${CLICKED_PATHS#,}" --click_markers_paths "${MARKERS_PATHS#,}"

python analysis/build_step_grid.py \
  --side_by_side_glob "$RUN/step{step}/color_mesh_from_knn_k10/render/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_side_by_side.png" \
  --steps 0,25,50,75,100,125,150,175,200,225,250,275,300 --cols 7 \
  --reference_image "$RUN/step0/color_mesh_from_knn_k10/${MASKBASE}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png" \
  --reference_label "source (9px)" \
  --output_path "$RUN/step_progression_grid.png"

echo "########## comparison grid vs SHIC ##########"
python -c "
from PIL import Image, ImageDraw
def load(p): return Image.open(p).convert('RGB')
shic_top1 = load('demo/shic/bear3/back_projection/bear3_top1_backproj_checker16_uv_k1_withbg_side_by_side.png')
shic_soft = load('demo/shic/bear3/back_projection/bear3_soft_backproj_checker16_uv_k1_withbg_side_by_side.png')
ours_top1 = load('$RUN/analyze/back_projection/bear3_OURS_top1_backproj_checker16_uv_k1_withbg_side_by_side.png')
ours_soft = load('$RUN/analyze/back_projection/bear3_OURS_soft_backproj_checker16_uv_k1_withbg_side_by_side.png')
h=360
def rs(im): return im.resize((int(im.width*h/im.height), h))
row1 = [rs(shic_top1), rs(shic_soft)]
row2 = [rs(ours_top1), rs(ours_soft)]
band=32
w = max(sum(im.width for im in row1), sum(im.width for im in row2))
c = Image.new('RGB', (w, band*2+h*2+10), (255,255,255))
d = ImageDraw.Draw(c)
y=0
d.text((10,y+8),'SHIC -- top1 / soft', fill=(0,0,0))
y+=band
x=0
for im in row1: c.paste(im,(x,y)); x+=im.width
y+=h+10
d.text((10,y+8),'OURS (mu=$MU) -- top1 / soft', fill=(0,0,0))
y+=band
x=0
for im in row2: c.paste(im,(x,y)); x+=im.width
c.save('$RUN/comparison_vs_shic/comparison_grid_shic_vs_ours.png')
print('saved')
"

echo "SHIC-BEAR FULL ANALYSIS DONE mu=$MU"
