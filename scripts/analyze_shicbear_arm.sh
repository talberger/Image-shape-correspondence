#!/bin/bash
# SMALL analysis for a SHIC-bear arm (2026-09-16): top1 + soft back-projection, coverage, PCA, sim heatmap,
# and the comparison grid vs SHIC. Same calls as run_shicbear_full_analysis.sh minus step progression,
# with the soft-snap read temperature taken from READ_T instead of hard-coded 1.0.
# Env: P2 (encoder dir), RUN (demo dir), PIX (photo features), READ_T, LABEL.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb

OBJ=./demo/shic/shapes/bear.obj
NAME=shic_bear
# All five inputs are env-overridable (2026-09-20) so the same analysis can run on a DIFFERENT
# resolution of bear3 (the long_side-400 comparison) without a forked copy of this script. The
# defaults are the usual 808x750 bear3 arm, so every existing caller is unchanged.
MASK=${MASK:-demo/shic/masks/bear3_click_350_300_sam_mask_1.pth}
POSCOLOR=${POSCOLOR:-demo/shic/bear3/bear3_click_350_300_sam_mask_1_position_color.png}
CHK=demo/shic/colored_mesh/bear_shic_normalized_mesh_checker16_uv_color.pth
RENDER=demo/shic/colored_mesh/checker16_uv_azim135.png
IMG=${IMG:-demo/shic/images/bear3.png}
MASKBASE=${MASKBASE:-bear3_click_350_300_sam_mask_1}

mkdir -p "$RUN/analyze/back_projection" "$RUN/analyze/ours" "$RUN/analyze/soft_mapping" \
         "$RUN/analyze/features_analyze/pca" "$RUN/analyze/features_analyze/sim_heatmap" "$RUN/comparison_vs_shic"

echo "########## top1 correspondence ##########"
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIX" \
  --vertex_features_path "$P2/pred_f.pth" --save_dir "$RUN/analyze/ours" --top_k 10

python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$RUN/analyze/ours" --knn_path "$RUN/analyze/ours/${MASKBASE}_pixel_to_vertex_nn.pth" \
  --vertex_colors_path "$CHK" --source_image_path "$IMG" --k 1 --save_dir "$RUN/analyze/back_projection" \
  --output_path "$RUN/analyze/back_projection/bear3_OURS_top1_backproj_checker16_uv_k1_withbg.png" \
  --mesh_render_path "$RENDER"
rm -f "$RUN/analyze/back_projection/bear3_OURS_top1_backproj_checker16_uv_k1_withbg.png"

echo "########## soft-snap correspondence (read T=$READ_T) ##########"
python analysis/color_mesh_from_soft_mapping.py --mask_path "$MASK" --img_path "$IMG" \
  --save_dir "$RUN/analyze/soft_mapping" --features_path "$PIX" --vertex_features_path "$P2/pred_f.pth" \
  --obj_path "$OBJ" --name "$NAME" --color_mode position --color_path "$POSCOLOR" \
  --k_vertices_per_pixel 10 --temperature "$READ_T"

python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$RUN/analyze/soft_mapping" \
  --knn_path "$RUN/analyze/soft_mapping/${MASKBASE}_soft_snap_correspondence.pth" \
  --vertex_colors_path "$CHK" --source_image_path "$IMG" --k 1 --save_dir "$RUN/analyze/back_projection" \
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

echo "########## comparison grid vs SHIC ##########"
python - "$RUN" "$LABEL" <<'EOF'
import sys
from PIL import Image, ImageDraw
run, label = sys.argv[1], sys.argv[2]
old = 'demo/shic/bear3_relational_geodesic_mu3dscale0.6_T1_seed0_phase2/analyze/back_projection'
rows = [
    ('SHIC (their weights) -- top1 / soft', 'demo/shic/bear3/back_projection/bear3_{}_backproj_checker16_uv_k1_withbg_side_by_side.png'),
    (label + ' -- top1 / soft', run + '/analyze/back_projection/bear3_OURS_{}_backproj_checker16_uv_k1_withbg_side_by_side.png'),
    ('previous bear pick (old recipe, mu=0.6, T=1) -- top1 / soft', old + '/bear3_OURS_{}_backproj_checker16_uv_k1_withbg_side_by_side.png'),
]
h, band = 360, 32
# Extra mesh views (Tal, 2026-09-16): full side + head-on front, next to the photo.
views = [Image.open(p).convert('RGB') for p in ('demo/shic/images/bear3.png',
         'demo/shic/colored_mesh/checker16_uv_azim180.png', 'demo/shic/colored_mesh/checker16_uv_azim90.png')]
built = [('photo / mesh side (azim 180) / mesh front (azim 90)', [im.resize((int(im.width * h / im.height), h)) for im in views])]
for title, pat in rows:
    ims = [Image.open(pat.format(k)).convert('RGB') for k in ('top1', 'soft')]
    built.append((title, [im.resize((int(im.width * h / im.height), h)) for im in ims]))
w = max(sum(im.width for im in ims) for _, ims in built)
c = Image.new('RGB', (w, len(built) * (band + h + 10)), (255, 255, 255))
d = ImageDraw.Draw(c)
y = 0
for title, ims in built:
    d.text((10, y + 8), title, fill=(0, 0, 0)); y += band
    x = 0
    for im in ims:
        c.paste(im, (x, y)); x += im.width
    y += h + 10
out = run + '/comparison_vs_shic/comparison_grid_shic_vs_ours.png'
c.save(out); print('saved', out)
EOF
echo "SHIC-BEAR ARM ANALYSIS DONE"
