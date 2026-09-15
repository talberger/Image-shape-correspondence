#!/bin/bash
# SMALL ANALYZE for ISOBBOX runs TRAINED AT T=1 (2026-09-03, Tal: "when training with T=1 no need
# to analyze with T read = something else").
#
# Derived from scripts/small_analyze_isobbox.sh. THREE differences, all deliberate:
#   1. Stage 7 (read T=0.01 soft mapping + back-projection) is REMOVED, and with it the
#      "readT matched" comparison figure that consumed its output. Everything here is read T=1.
#   2. The MATCHED CONTROL is added: the isobbox T=1 mu_3d_scale=1 run -- same mesh, same T, same
#      seed, same features, only mu_3d_scale differs. That is the comparison this run exists for.
#   3. Eval produces TWO figures: vs the same-mesh isobbox controls (no rescaling needed), and vs
#      the two old-mesh baselines (--rescale_across_meshes, as every isobbox run has done).
#
# Usage:  ./small_analyze_isobbox_readT1.sh <RUN_DIR_NAME> <DEMO_DIR_NAME> <EVAL_TAG> [pix_features]
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb

RUN_NAME=$1
DEMO_NAME=$2
EVAL_TAG=$3
PIXFEAT_OVERRIDE=${4:-}

[ -z "$EVAL_TAG" ] && { echo "usage: $0 <run_dir_name> <demo_dir_name> <eval_tag> [pix_features_path]"; exit 1; }

OBJ=./meshes/guitar_isotropic_bboxnorm.obj
NAME=guitar_isobbox
RUN=experiments/guitar/$RUN_NAME
DEMO=demo/guitar/$DEMO_NAME
OUT=$DEMO/analyze
EV=$DEMO/eval
EL=demo/guitar/eval_labels

MASK=demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
PIXFEAT=${PIXFEAT_OVERRIDE:-demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth}
POSCOLOR=demo/guitar/segment_masks_pixel270_370/pixel_coloring/guitar_01_click_270_370_sam_mask_2_position_color.png
CHECKER=demo/guitar/colored_mesh_isobbox/guitar_isotropic_bboxnorm_mesh_checker16_color.pth
GEO3D=./experiments/guitar/geodesic_tables/guitar_isobbox_geo_table_3d.pth
EVALPAIRS=$EL/isobbox/pixel_vertex_eval_pairs.pth
MB=guitar_01_click_270_370_sam_mask_2
CHK_STEM=$(basename "${CHECKER%.pth}")
PIXTAG=explicit9
PIXEL_INDICES="9501 22740 31719 46561 75904 92318 156233 184915 188068"
QUERY_VERTICES="535 549 1017 2885"
PRED_F=$RUN/pred_f.pth
MU=$(echo "$RUN_NAME" | sed -n 's/.*mu3dscale\([0-9.]*\)_.*/\1/p')
[ -z "$MU" ] && MU="?"
# train temperature, also from the run name -- the labels used to hardcode "T=1" and would have
# mislabelled any other arm (2026-09-03, caught before the T=10 run reached this stage).
TT=$(echo "$RUN_NAME" | sed -n 's/.*_T\([0-9.]*\)_seed.*/\1/p')
[ -z "$TT" ] && TT="?"

mkdir -p "$OUT/color_mesh_from_knn_k10/render" "$OUT/color_mesh_from_soft_mapping/render" \
         "$OUT/back_projection" "$OUT/features_analyze/pca" "$OUT/features_analyze/sim_heatmap" "$EV"

CHECKER_VIEWS="${CHECKER%.pth}_rendered_views.png"
if [ ! -f "$CHECKER_VIEWS" ]; then
  echo "=== [0/8] rendering the isotropic checker mesh views (one-off prerequisite) ==="
  python analysis/render_colored_mesh.py --obj_path "$OBJ" \
    --ply_path "${CHECKER%.pth}.ply" --save_dir "$(dirname "$CHECKER")" --save_rendered_views_grid
fi

echo "=== [1/8] pixel_to_vertex_nn (top_k=10) -- mesh-free, pure feature matching ==="
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT" --top_k 10
KNN="$OUT/${MB}_pixel_to_vertex_nn.pth"

echo "=== [2/8] final-state mesh coloring (k=10, 9 tracked pixels) ==="
python analysis/color_mesh_from_knn.py --obj_path "$OBJ" --mask_path "$MASK" \
  --save_dir "$OUT/color_mesh_from_knn_k10" \
  --knn_path "$KNN" --color_mode position --color_path "$POSCOLOR" --k 10 --pixel_indices $PIXEL_INDICES
python analysis/render_colored_mesh.py --obj_path "$OBJ" \
  --ply_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}.ply" \
  --save_dir "$OUT/color_mesh_from_knn_k10/render" --azim_deg 90 \
  --source_image_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png" \
  --click_markers_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_click_markers.pth"

echo "=== [3/8] soft mapping (read T=1) -- coverage ==="
python analysis/color_mesh_from_soft_mapping.py --obj_path "$OBJ" --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT/color_mesh_from_soft_mapping" \
  --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10
SOFT="$OUT/color_mesh_from_soft_mapping/${MB}_soft_snap_correspondence.pth"

echo "=== [4/8] back-projection, top1k + soft-snap (read T=1) ==="
python analysis/color_image_from_knn.py --mask_path "$MASK" --knn_dir "$OUT" --knn_path "$KNN" \
  --vertex_colors_path "$CHECKER" --k 1 --save_dir "$OUT/back_projection"
python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$OUT/color_mesh_from_soft_mapping" --knn_path "$SOFT" \
  --vertex_colors_path "$CHECKER" --k 1 --save_dir "$OUT/back_projection" \
  --output_path "$OUT/back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1.png"
rm -f "$OUT/back_projection/${MB}_backproj_${CHK_STEM}_k1.png" \
      "$OUT/back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1.png"

echo "=== [5/8] feature analysis: PCA ==="
python analysis/pca_feature_compare.py --obj_path "$OBJ" --display both --save_dir "$OUT/features_analyze/pca" \
  --mask_path "$MASK" --features_path "$PIXFEAT" --vertex_features_path "$PRED_F"

echo "=== [6/8] feature analysis: similarity heatmap (REMAPPED query vertices) ==="
python analysis/feature_similarity_heatmap.py --obj_path "$OBJ" \
  --query_pixels 151 94 571 206 820 360 820 740 --query_vertices $QUERY_VERTICES \
  --save_dir "$OUT/features_analyze/sim_heatmap" --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F"

COV=$(python - "$DEMO" <<'PY'
import sys, torch, glob, os
p = glob.glob(os.path.join(sys.argv[1], 'analyze', 'color_mesh_from_soft_mapping',
                           '*_soft_snap_correspondence.pth'))
if not p: print('n/a'); raise SystemExit
c = torch.load(p[0], map_location='cpu')
n = int(c['num_vertices']); k = int(torch.unique(c['top1_vertex_idx']).numel())
print(f'{k}/{n} = {100*k/n:.2f}%')
PY
)
echo "coverage (read T=1): $COV   [denominator is 3886, NOT the old mesh's 4036]"

BP=back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1_side_by_side.png
BPK1=back_projection/${MB}_backproj_${CHK_STEM}_k1_side_by_side.png
BP_OLD=back_projection/${MB}_backproj_guitar_mesh_checker16_color_soft_snap_k1_side_by_side.png
BPK1_OLD=back_projection/${MB}_backproj_guitar_mesh_checker16_color_k1_side_by_side.png
CTRL=demo/guitar/runs/isobbox/bsb_isobbox_sdsquash_T1_seed0_phase2
B001=demo/guitar/runs/oldmesh_sweeps/stride14/bsb_sd_dino_stride14_nosdsquare_T0.01_seed0_phase2
B1=demo/guitar/runs/oldmesh_sweeps/stride14/bsb_sd_dino_stride14_nosdsquare_T1_seed0_phase2

echo "=== [7/8] back-projection comparisons, all at read T=1 ==="
# (a) THE matched control: same mesh, same T, same seed -- mu_3d_scale is the only difference.
python analysis/stack_images_labeled.py \
  --top_path "$OUT/$BP" --top_label "isobbox train T=$TT, mu_3d_scale=$MU (this run), read T=1 (soft-snap)" \
  --bottom_path "$CTRL/analyze/$BP" \
  --bottom_label "MATCHED CONTROL isobbox T=1, mu_3d_scale=1, read T=1 (soft-snap)" \
  --title "back-projection soft-snap k=1 -- mu_3d_scale $MU (train T=$TT) vs mu=1 T=1, same mesh / same seed" \
  --output_path "$DEMO/comparison_vs_mu1_control/1_back_projection_soft_readT1.png"
python analysis/stack_images_labeled.py \
  --top_path "$OUT/$BPK1" --top_label "isobbox train T=$TT, mu_3d_scale=$MU (this run), read T=1 (top1k)" \
  --bottom_path "$CTRL/analyze/$BPK1" \
  --bottom_label "MATCHED CONTROL isobbox T=1, mu_3d_scale=1, read T=1 (top1k)" \
  --title "back-projection top1k -- mu_3d_scale $MU (train T=$TT) vs mu=1 T=1, same mesh / same seed" \
  --output_path "$DEMO/comparison_vs_mu1_control/2_back_projection_top1k_readT1.png"
# (b) the two old-mesh baselines, for continuity with every other isobbox run
for pair in "T0.01:$B001" "T1:$B1"; do
  tag=${pair%%:*}; bdir=${pair#*:}
  python analysis/stack_images_labeled.py \
    --top_path "$OUT/$BP" --top_label "isobbox mu_3d_scale=$MU train T=$TT, read T=1 (soft-snap)" \
    --bottom_path "$bdir/analyze/$BP_OLD" --bottom_label "BASELINE $tag original mesh, read T=1 (soft-snap)" \
    --title "back-projection soft-snap k=1, read T=1 -- mu$MU isobbox vs baseline $tag" \
    --output_path "$DEMO/comparison_vs_baseline_$tag/8_back_projection_soft_readT1.png"
  python analysis/stack_images_labeled.py \
    --top_path "$OUT/$BPK1" --top_label "isobbox mu_3d_scale=$MU train T=$TT, read T=1 (top1k)" \
    --bottom_path "$bdir/analyze/$BPK1_OLD" --bottom_label "BASELINE $tag original mesh, read T=1 (top1k)" \
    --title "back-projection top1k, read T=1 -- mu$MU isobbox vs baseline $tag" \
    --output_path "$DEMO/comparison_vs_baseline_$tag/9_back_projection_top1k_readT1.png"
done

echo "=== [8/8] ground-truth eval, on the ISOBBOX eval pairs and geodesic table ==="
python analysis/eval_geodesic_error.py --obj_path "$OBJ" --name "$NAME" \
  --run_label "$EVAL_TAG (isobbox mesh, 3886v)" \
  --geo_table_path "$GEO3D" \
  --eval_pairs_path "$EVALPAIRS" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --k_vertices_per_pixel 10 \
  --save_path "$EL/eval_${EVAL_TAG}.pth" --plot_path "$EL/eval_${EVAL_TAG}.png"
# same-mesh controls: no rescaling involved, these three share a geodesic diameter and a denominator
python analysis/compare_eval_runs.py \
  --runs $EL/eval_${EVAL_TAG}.pth \
         $EL/eval_isobbox_stride14_nosdsquare_T1_seed0.pth \
         $EL/eval_isobbox_stride14_nosdsquare_T0.01_seed0.pth \
  --labels "mu_3d_scale=$MU train T=$TT" "CONTROL mu_3d_scale=1 T=1" "isobbox arm mu=1 T=0.01" \
  --save_path "$EV/eval_vs_isobbox_controls.png"
# old-mesh baselines: different mesh, so convert to the raw .obj scale first
python analysis/compare_eval_runs.py --rescale_across_meshes \
  --runs $EL/eval_${EVAL_TAG}.pth \
         $EL/eval_stride14_nosdsquare_T0.01_seed0.pth \
         $EL/eval_stride14_nosdsquare_T1_seed0.pth \
  --labels "mu_3d_scale=$MU train T=$TT (3886v)" "BASELINE T=0.01 (4036v)" "BASELINE companion T=1 (4036v)" \
  --save_path "$EV/eval_vs_both_baselines.png"

echo "===== SMALL ANALYZE (ISOBBOX, read T=1 only) DONE: $DEMO ====="
