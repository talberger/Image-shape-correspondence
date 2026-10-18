#!/usr/bin/env bash
# SHIC-comparison bear: full analysis for the ZS (phase-1 only, NO phase-2) arm (2026-09-04).
# Usage: ./run_shicbear_zs_analysis.sh
# Mirrors scripts/run_shicbear_full_analysis.sh exactly, with two differences:
#   1. vertex features come from the phase-1 encoder's pred_f.pth, not a phase-2 run's
#   2. NO step progression -- ZS *is* step 0, there are no checkpoints to walk
# Naming follows the guitar's own ZS convention (demo/guitar/bsb_<phase1_tag>_ZS).
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

OBJ=./demo/shic/shapes/bear.obj
NAME=shic_bear
P1=experiments/shic_bear/encoder_dfd_crossview_phase1
RUN=demo/shic/bear3_dfd_crossview_ZS
MASK=demo/shic/masks/bear3_click_350_300_sam_mask_1.pth
PIX=demo/shic/features_analyze/stride14_squash/bear3_pix_stride14_squash.pth
POSCOLOR=demo/shic/bear3/bear3_click_350_300_sam_mask_1_position_color.png
CHK=demo/shic/colored_mesh/bear_shic_normalized_mesh_checker16_uv_color.pth
RENDER=demo/shic/colored_mesh/checker16_uv_azim135.png
IMG=demo/shic/images/bear3.png
MASKBASE=bear3_click_350_300_sam_mask_1

mkdir -p "$RUN/analyze/back_projection" "$RUN/analyze/ours" "$RUN/analyze/ours_soft" \
         "$RUN/analyze/soft_mapping" "$RUN/analyze/features_analyze/pca" \
         "$RUN/analyze/features_analyze/sim_heatmap" "$RUN/comparison_vs_shic"

echo "########## top1 correspondence ##########"
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIX" \
  --vertex_features_path "$P1/pred_f.pth" --save_dir "$RUN/analyze/ours" --top_k 10

echo "########## top1 back-projection ##########"
python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$RUN/analyze/ours" \
  --knn_path "$RUN/analyze/ours/${MASKBASE}_pixel_to_vertex_nn.pth" \
  --vertex_colors_path "$CHK" --source_image_path "$IMG" \
  --k 1 --save_dir "$RUN/analyze/back_projection" \
  --output_path "$RUN/analyze/back_projection/bear3_OURS_ZS_top1_backproj_checker16_uv_k1_withbg.png" \
  --mesh_render_path "$RENDER"
rm -f "$RUN/analyze/back_projection/bear3_OURS_ZS_top1_backproj_checker16_uv_k1_withbg.png"

echo "########## soft-snap correspondence ##########"
python analysis/color_mesh_from_soft_mapping.py --mask_path "$MASK" --img_path "$IMG" \
  --save_dir "$RUN/analyze/soft_mapping" \
  --features_path "$PIX" --vertex_features_path "$P1/pred_f.pth" \
  --obj_path "$OBJ" --name "$NAME" \
  --color_mode position --color_path "$POSCOLOR" \
  --k_vertices_per_pixel 10 --temperature 1.0

echo "########## soft back-projection ##########"
python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$RUN/analyze/soft_mapping" \
  --knn_path "$RUN/analyze/soft_mapping/${MASKBASE}_soft_snap_correspondence.pth" \
  --vertex_colors_path "$CHK" --source_image_path "$IMG" \
  --k 1 --save_dir "$RUN/analyze/back_projection" \
  --output_path "$RUN/analyze/back_projection/bear3_OURS_ZS_soft_backproj_checker16_uv_k1_withbg.png" \
  --mesh_render_path "$RENDER"
rm -f "$RUN/analyze/back_projection/bear3_OURS_ZS_soft_backproj_checker16_uv_k1_withbg.png"

echo "########## PCA ##########"
python analysis/pca_feature_compare.py --obj_path "$OBJ" --display both \
  --save_dir "$RUN/analyze/features_analyze/pca" \
  --mask_path "$MASK" --features_path "$PIX" --vertex_features_path "$P1/pred_f.pth" \
  --azim_deg 135 --elev_deg 10

echo "########## similarity heatmap ##########"
python analysis/feature_similarity_heatmap.py --obj_path "$OBJ" \
  --query_pixels 46 549 193 773 362 239 362 581 --query_vertices 1675 177 242 470 \
  --save_dir "$RUN/analyze/features_analyze/sim_heatmap" --mask_path "$MASK" --features_path "$PIX" \
  --vertex_features_path "$P1/pred_f.pth" --render_res 300 --elev_deg 10 --azim_deg 135

echo "########## comparison grid vs SHIC ##########"
python -c "
from PIL import Image, ImageDraw
def load(p): return Image.open(p).convert('RGB')
shic_top1 = load('demo/shic/bear3/back_projection/bear3_top1_backproj_checker16_uv_k1_withbg_side_by_side.png')
shic_soft = load('demo/shic/bear3/back_projection/bear3_soft_backproj_checker16_uv_k1_withbg_side_by_side.png')
ours_top1 = load('$RUN/analyze/back_projection/bear3_OURS_ZS_top1_backproj_checker16_uv_k1_withbg_side_by_side.png')
ours_soft = load('$RUN/analyze/back_projection/bear3_OURS_ZS_soft_backproj_checker16_uv_k1_withbg_side_by_side.png')
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
d.text((10,y+8),'OURS (ZS, phase-1 only) -- top1 / soft', fill=(0,0,0))
y+=band
x=0
for im in row2: c.paste(im,(x,y)); x+=im.width
c.save('$RUN/comparison_vs_shic/comparison_grid_shic_vs_ours.png')
print('saved')
"

echo "SHIC-BEAR ZS ANALYSIS DONE"
