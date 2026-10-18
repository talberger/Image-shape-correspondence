#!/bin/bash
# SMALL ANALYZE for the ISOTROPIC-MESH pipeline (2026-09-02).
# Usage:  ./small_analyze_iso.sh <RUN_DIR_NAME> <DEMO_DIR_NAME> <EVAL_TAG>
#
# Same 9 stages as scripts/small_analyze.sh -- see that file for what the gate contains and what it
# deliberately drops. THREE things differ, all of them consequences of the mesh change:
#
#   1. --obj_path ./meshes/guitar_isotropic.obj is passed to EVERY mesh-consuming call. The default
#      in ~30 scripts here is './meshes/guitar.obj', so an omitted flag silently runs the old mesh.
#   2. Mesh-side artifacts point at their isotropic rebuilds: the checker16 vertex colors, the eval
#      pairs, and the 3D geodesic table the eval normalizes by.
#   3. --query_vertices is REMAPPED: 1527 1545 2760 642 -> 535 549 1017 2885 (nearest vertex on the
#      new mesh, scripts/map_vertices_between_meshes.py, gaps <= 0.0068). The old ids are all still
#      < 3886, so leaving them would have silently queried different anatomy rather than erroring.
#
# Comparisons against the OLD-MESH baselines: the back-projection images are photo-space and are
# directly comparable (that is the visual gate). The eval numbers are NOT on a common denominator --
# geodesic diameter 1.9026 vs 1.63806, coverage /4036 vs /3886 -- so read AUC, not raw percentages,
# and say which mesh a number came from.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

RUN_NAME=$1
DEMO_NAME=$2
EVAL_TAG=$3
PIXFEAT_OVERRIDE=${4:-}   # optional: a different photo-feature file (e.g. the DINO-only arm)
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
# color_image_from_knn.py names its outputs after the VERTEX COLORS file, not the mesh -- so with
# the isotropic checker they are guitar_isotropic_mesh_checker16_color_*, not guitar_mesh_*.
# Derive the stem instead of hardcoding it (2026-09-02: hardcoding the old name broke stage 8).
CHK_STEM=$(basename "${CHECKER%.pth}")
PIXTAG=explicit9
PIXEL_INDICES="9501 22740 31719 46561 75904 92318 156233 184915 188068"
QUERY_VERTICES="535 549 1017 2885"
PRED_F=$RUN/pred_f.pth
# READ TEMPERATURE. This was hardcoded to 0.01 until 2026-09-03, which silently produced a
# trained-at-0.5 / read-at-0.01 analyze_readT/ for the T=0.5 arm -- exactly the read-temperature
# confound the 2026-08-31 notes flagged. It is now taken from the run directory name, so every arm
# is read at the temperature it was TRAINED at.
READ_T=$(echo "$RUN_NAME" | sed -n 's/.*_T\([0-9.]*\)_seed.*/\1/p')
[ -z "$READ_T" ] && { echo "cannot parse train temperature from run name '$RUN_NAME'"; exit 1; }
RT=$DEMO/analyze_readT${READ_T}

mkdir -p "$OUT/color_mesh_from_knn_k10/render" "$OUT/color_mesh_from_soft_mapping/render" \
         "$OUT/back_projection" "$OUT/features_analyze/pca" "$OUT/features_analyze/sim_heatmap" \
         "$RT/color_mesh_from_soft_mapping/render" "$RT/back_projection" "$EV"

# color_image_from_knn.py builds its side-by-side against <vertex_colors_path>_rendered_views.png.
# That render exists for the original mesh's checker but has to be produced once per new mesh, or
# stage 8 dies on a missing input several minutes in (2026-09-02).
CHECKER_VIEWS="${CHECKER%.pth}_rendered_views.png"
if [ ! -f "$CHECKER_VIEWS" ]; then
  echo "=== [0/9] rendering the isotropic checker mesh views (one-off prerequisite) ==="
  python analysis/render_colored_mesh.py --obj_path "$OBJ" \
    --ply_path "${CHECKER%.pth}.ply" --save_dir "$(dirname "$CHECKER")" --save_rendered_views_grid
fi

echo "=== [1/9] pixel_to_vertex_nn (top_k=10) -- mesh-free, pure feature matching ==="
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT" --top_k 10
KNN="$OUT/${MB}_pixel_to_vertex_nn.pth"

echo "=== [2/9] final-state mesh coloring (k=10, 9 tracked pixels) ==="
python analysis/color_mesh_from_knn.py --obj_path "$OBJ" --mask_path "$MASK" \
  --save_dir "$OUT/color_mesh_from_knn_k10" \
  --knn_path "$KNN" --color_mode position --color_path "$POSCOLOR" --k 10 --pixel_indices $PIXEL_INDICES
python analysis/render_colored_mesh.py --obj_path "$OBJ" \
  --ply_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}.ply" \
  --save_dir "$OUT/color_mesh_from_knn_k10/render" --azim_deg 90 \
  --source_image_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png" \
  --click_markers_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_click_markers.pth"

echo "=== [3/9] soft mapping (read T=1) -- coverage ==="
python analysis/color_mesh_from_soft_mapping.py --obj_path "$OBJ" --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT/color_mesh_from_soft_mapping" \
  --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10
SOFT="$OUT/color_mesh_from_soft_mapping/${MB}_soft_snap_correspondence.pth"

echo "=== [4/9] back-projection, top1k + soft-snap (read T=1) ==="
python analysis/color_image_from_knn.py --mask_path "$MASK" --knn_dir "$OUT" --knn_path "$KNN" \
  --vertex_colors_path "$CHECKER" --k 1 --save_dir "$OUT/back_projection"
python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$OUT/color_mesh_from_soft_mapping" --knn_path "$SOFT" \
  --vertex_colors_path "$CHECKER" --k 1 --save_dir "$OUT/back_projection" \
  --output_path "$OUT/back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1.png"
rm -f "$OUT/back_projection/${MB}_backproj_${CHK_STEM}_k1.png" \
      "$OUT/back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1.png"

echo "=== [5/9] feature analysis: PCA ==="
python analysis/pca_feature_compare.py --obj_path "$OBJ" --display both --save_dir "$OUT/features_analyze/pca" \
  --mask_path "$MASK" --features_path "$PIXFEAT" --vertex_features_path "$PRED_F"

echo "=== [6/9] feature analysis: similarity heatmap (REMAPPED query vertices) ==="
python analysis/feature_similarity_heatmap.py --obj_path "$OBJ" \
  --query_pixels 151 94 571 206 820 360 820 740 --query_vertices $QUERY_VERTICES \
  --save_dir "$OUT/features_analyze/sim_heatmap" --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F"

echo "=== [7/9] read T=$READ_T (matched to training): soft mapping + back-projection ==="
python analysis/color_mesh_from_soft_mapping.py --obj_path "$OBJ" --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$RT/color_mesh_from_soft_mapping" \
  --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10 --temperature $READ_T
python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$RT/color_mesh_from_soft_mapping" \
  --knn_path "$RT/color_mesh_from_soft_mapping/${MB}_soft_snap_correspondence.pth" \
  --vertex_colors_path "$CHECKER" --k 1 --save_dir "$RT/back_projection" \
  --output_path "$RT/back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1.png"
rm -f "$RT/back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1.png"
python analysis/render_colored_mesh.py --obj_path "$OBJ" \
  --ply_path "$RT/color_mesh_from_soft_mapping/${MB}_mesh_color_soft_snap_position_k10.ply" \
  --save_dir "$RT/color_mesh_from_soft_mapping/render" --azim_deg 90 \
  --source_image_path "$POSCOLOR"

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
# the old-mesh baselines were rendered against demo/guitar/colored_mesh/guitar_mesh_checker16_color
BP_OLD=back_projection/${MB}_backproj_guitar_mesh_checker16_color_soft_snap_k1_side_by_side.png
BPK1_OLD=back_projection/${MB}_backproj_guitar_mesh_checker16_color_k1_side_by_side.png
B001=demo/guitar/runs/oldmesh_sweeps/stride14/bsb_sd_dino_stride14_nosdsquare_T0.01_seed0_phase2
B1=demo/guitar/runs/oldmesh_sweeps/stride14/bsb_sd_dino_stride14_nosdsquare_T1_seed0_phase2

echo "=== [8/9] back-projection comparisons vs BOTH old-mesh baselines (photo space, comparable) ==="
python analysis/stack_images_labeled.py \
  --top_path "$RT/$BP" --top_label "ISOTROPIC MESH (bbox-normalized) (3886v), train T=$READ_T, read T=$READ_T" \
  --bottom_path "$B001/analyze_readT/$BP_OLD" \
  --bottom_label "BASELINE original mesh (4036v) stride14 squash T=0.01 seed0, read T=0.01" \
  --title "back-projection (checker16, soft-snap k=1) -- each at its own trained T (top $READ_T / bottom 0.01), isotropic vs original mesh" \
  --output_path "$DEMO/comparison_vs_baseline_T0.01/7_back_projection_readT_matched.png"
for pair in "T0.01:$B001" "T1:$B1"; do
  tag=${pair%%:*}; bdir=${pair#*:}
  python analysis/stack_images_labeled.py \
    --top_path "$OUT/$BP" --top_label "ISOTROPIC MESH (bbox-normalized), read T=1 (soft-snap)" \
    --bottom_path "$bdir/analyze/$BP_OLD" --bottom_label "BASELINE $tag original mesh, read T=1 (soft-snap)" \
    --title "back-projection soft-snap k=1, read T=1 -- isotropic vs baseline $tag" \
    --output_path "$DEMO/comparison_vs_baseline_$tag/8_back_projection_soft_readT1.png"
  python analysis/stack_images_labeled.py \
    --top_path "$OUT/$BPK1" --top_label "ISOTROPIC MESH (bbox-normalized), read T=1 (top1k)" \
    --bottom_path "$bdir/analyze/$BPK1_OLD" --bottom_label "BASELINE $tag original mesh, read T=1 (top1k)" \
    --title "back-projection top1k, read T=1 -- isotropic vs baseline $tag" \
    --output_path "$DEMO/comparison_vs_baseline_$tag/9_back_projection_top1k_readT1.png"
done

echo "=== [9/9] ground-truth eval, on the ISOTROPIC eval pairs and geodesic table ==="
python analysis/eval_geodesic_error.py --obj_path "$OBJ" --name "$NAME" \
  --run_label "$EVAL_TAG (isotropic mesh, 3886v)" \
  --geo_table_path "$GEO3D" \
  --eval_pairs_path "$EVALPAIRS" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --k_vertices_per_pixel 10 \
  --save_path "$EL/eval_${EVAL_TAG}.pth" --plot_path "$EL/eval_${EVAL_TAG}.png"
# --rescale_across_meshes: the two baselines live on meshes/guitar.obj, this run on the isotropic
# mesh, so their error units differ. The flag converts every run to the raw .obj scale and scores
# them against one reference length instead of refusing the comparison.
python analysis/compare_eval_runs.py --rescale_across_meshes \
  --runs $EL/eval_${EVAL_TAG}.pth \
         $EL/eval_stride14_nosdsquare_T0.01_seed0.pth \
         $EL/eval_stride14_nosdsquare_T1_seed0.pth \
  --labels "ISOTROPIC 3886v" "BASELINE T=0.01 (4036v)" "BASELINE companion T=1 (4036v)" \
  --save_path "$EV/eval_vs_both_baselines.png"

echo "===== SMALL ANALYZE (ISO) DONE: $DEMO ====="
