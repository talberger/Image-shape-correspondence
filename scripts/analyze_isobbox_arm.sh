#!/bin/bash
# FULL per-arm analysis for one isobbox arm (ZS or phase-2), read T=1.
# 2026-09-08, Tal: "features analyze should always be there" -- PCA and the similarity heatmap are
# stages of THIS script, not of an optional deeper battery, and the run-summary grid is built at
# the end so every arm has its one-glance sheet.
#
# Usage: ./analyze_isobbox_arm.sh <demo_run_name> <pred_f_path> <pix_features_path> [eval_tag]
#   demo_run_name : dir under demo/guitar/runs/isobbox/, or isobbox_prop/ when the name carries `_mu_`
#                   (lib_isobbox_common.sh run_dir; created if missing)
#   pred_f_path   : the encoder's pred_f.pth -- phase-1's for a ZS arm, phase-2's otherwise
#   pix_features_path : the photo features, i.e. WHICH FUSE ARM this is
#   eval_tag      : if given, also runs the v2 ground-truth eval into <EVALDIR>/eval_<tag>_v2.pth
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh
[ -n "$OLDMESH" ] && source scripts/lib_oldmesh_override.sh
[ -n "$NECKRINGS4" ] && source scripts/lib_neckrings4_override.sh

DEMO_NAME=$1; PRED_F=$2; PIXFEAT=$3; EVAL_TAG=${4:-}
[ -z "$PIXFEAT" ] && { echo "usage: $0 <demo_run_name> <pred_f_path> <pix_features_path> [eval_tag]"; exit 1; }
[ -f "$PRED_F" ]  || { echo "no such pred_f: $PRED_F"; exit 1; }
[ -f "$PIXFEAT" ] || { echo "no such photo features: $PIXFEAT"; exit 1; }

DEMO=$(run_dir "$DEMO_NAME")
OUT=$DEMO/analyze
mkdir -p "$OUT/color_mesh_from_knn_k10/render" "$OUT/color_mesh_from_soft_mapping/render" \
         "$OUT/back_projection" "$OUT/features_analyze/pca" "$OUT/features_analyze/sim_heatmap" "$DEMO/eval"
echo "arm: $DEMO_NAME"; echo "  pred_f : $PRED_F"; echo "  photo  : $PIXFEAT"

CHECKER_VIEWS="${CHECKER%.pth}_rendered_views.png"
if [ ! -f "$CHECKER_VIEWS" ]; then
  echo "=== [0/7] rendering the checker mesh views (one-off prerequisite) ==="
  # --render_res 512, not the 224 default (2026-09-16): the checker view is the reference panel the
  # back-projection figures are read against, and at 224 its silhouette is visibly stair-stepped.
  # demo/guitar/colored_mesh/'s original was made at 512; this call had been leaving it at 224.
  python analysis/render_colored_mesh.py --obj_path "$OBJ" --render_res 512 \
    --ply_path "${CHECKER%.pth}.ply" --save_dir "$(dirname "$CHECKER")" --save_rendered_views_grid
fi

# SMALL=1 (2026-09-15, Tal): ONE python process -- back-projection top1k + soft-snap, coverage, the
# similarity heatmap as a single row, eval v2. No PCA, no 2x2 grid, no 28 MB knn file; the photo
# features come from an in-mask cache instead of the 8 GB file. See analysis/analyze_arm_small.py.
if [ -n "$SMALL" ]; then
  python analysis/analyze_arm_small.py --save_dir "$OUT" \
    --vertex_features_path "$PRED_F" --features_path "$PIXFEAT" --mask_path "$MASK" \
    --obj_path "$OBJ" --name "$NAME" --checker_path "$CHECKER" \
    --query_pixels $QUERY_PIXELS --query_vertices $QUERY_VERTICES --k_vertices_per_pixel 10 \
    --temperature ${READ_T:-1} \
    ${EVAL_TAG:+--eval_pairs_path "$EVALPAIRS" --geo_table_path "$GEO3D" \
      --eval_save_path "$EVALDIR/eval_${EVAL_TAG}_v2.pth" --eval_plot_path "$EVALDIR/eval_${EVAL_TAG}_v2.png" \
      --run_label "$EVAL_TAG (${MESH_LABEL:-isobbox 3886v}, eval v2)"}
  echo "===== ARM ANALYSIS DONE: $DEMO ====="
  exit 0
fi

# FULL mode needs top_k=100 for the k=100 mesh coloring in the summary grid; k<=10 reads are unchanged.
TOPK=100; [ -n "$SMALL" ] && TOPK=10
echo "=== [1/7] pixel_to_vertex_nn (top_k=$TOPK) ==="
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT" --top_k $TOPK
KNN="$OUT/${MB}_pixel_to_vertex_nn.pth"

# SMALL=1 (2026-09-10, Tal: "for a start no full analysis -- features + back-projection"): skips
# ONLY this stage, the mesh coloring + kaolin render. Back-projection, PCA, sim heatmap, coverage and
# eval all still run, so compare_isobbox_arms.sh works unchanged on a SMALL arm.
if [ -n "$SMALL" ]; then
echo "=== [2/7] mesh coloring SKIPPED (SMALL=1) ==="
else
echo "=== [2/7] mesh coloring k=10 + the 9 tracked pixels ==="
python analysis/color_mesh_from_knn.py --obj_path "$OBJ" --mask_path "$MASK" \
  --save_dir "$OUT/color_mesh_from_knn_k10" --knn_path "$KNN" \
  --color_mode position --color_path "$POSCOLOR" --k 10 --pixel_indices $PIXEL_INDICES
python analysis/render_colored_mesh.py --obj_path "$OBJ" \
  --ply_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}.ply" \
  --save_dir "$OUT/color_mesh_from_knn_k10/render" --azim_deg 90 \
  --source_image_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png" \
  --click_markers_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_click_markers.pth"
# plain (no tracked pixels) k=1/10/100 colorings -- panels of the analysis_summary grid
echo "=== [2b/7] mesh coloring k=1/10/100 (summary-grid panels) ==="
mkdir -p "$OUT/color_mesh_from_knn_k1/render" "$OUT/color_mesh_from_knn_k100/render"
for k in 1 10 100; do
  python analysis/color_mesh_from_knn.py --obj_path "$OBJ" --mask_path "$MASK" \
    --save_dir "$OUT/color_mesh_from_knn_k$k" --knn_path "$KNN" \
    --color_mode position --color_path "$POSCOLOR" --k $k
done
fi

echo "=== [3/7] soft mapping (read T=1) ==="
python analysis/color_mesh_from_soft_mapping.py --obj_path "$OBJ" --mask_path "$MASK" \
  --features_path "$PIXFEAT" --vertex_features_path "$PRED_F" \
  --save_dir "$OUT/color_mesh_from_soft_mapping" \
  --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10
SOFT="$OUT/color_mesh_from_soft_mapping/${MB}_soft_snap_correspondence.pth"

if [ -z "$SMALL" ]; then
echo "=== [3b/7] render k=1/10/100 + soft-snap colorings, hard-snap markers (summary-grid panels) ==="
PLYS="$OUT/color_mesh_from_knn_k1/${MB}_mesh_color_position_k1.ply,$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10.ply"
PLYS="$PLYS,$OUT/color_mesh_from_knn_k100/${MB}_mesh_color_position_k100.ply,$OUT/color_mesh_from_soft_mapping/${MB}_mesh_color_soft_snap_position_k10.ply"
python analysis/render_colored_mesh.py --obj_path "$OBJ" --ply_paths "$PLYS" --azim_deg 90 \
  --save_dirs "$OUT/color_mesh_from_knn_k1/render,$OUT/color_mesh_from_knn_k10/render,$OUT/color_mesh_from_knn_k100/render,$OUT/color_mesh_from_soft_mapping/render" \
  --source_image_paths "$POSCOLOR,$POSCOLOR,$POSCOLOR,$POSCOLOR"
python analysis/color_mesh_hard_snap_markers.py --obj_path "$OBJ" --mask_path "$MASK" \
  --soft_snap_correspondence_path "$SOFT" --color_path "$POSCOLOR" --pixel_indices $PIXEL_INDICES
python analysis/render_colored_mesh.py --obj_path "$OBJ" \
  --ply_path "$OUT/color_mesh_from_soft_mapping/${MB}_hard_snap_spheres_pix_${PIXTAG}.ply" \
  --save_dir "$OUT/color_mesh_from_soft_mapping/render" --azim_deg 90 \
  --source_image_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png"
fi

echo "=== [4/7] back-projection, top1k + soft-snap (read T=1) ==="
# color_image_from_knn.py now deletes the standalone PNG itself (2026-09-08) -- the side-by-side's
# right panel is bit-identical to it, and keeping both produced mismatched-panel comparisons.
python analysis/color_image_from_knn.py --mask_path "$MASK" --knn_dir "$OUT" --knn_path "$KNN" \
  --vertex_colors_path "$CHECKER" --k 1 --save_dir "$OUT/back_projection"
python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$OUT/color_mesh_from_soft_mapping" --knn_path "$SOFT" \
  --vertex_colors_path "$CHECKER" --k 1 --save_dir "$OUT/back_projection" \
  --output_path "$OUT/back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1.png"

echo "=== [5/7] features_analyze: PCA (shared basis, image+mesh) ==="
python analysis/pca_feature_compare.py --obj_path "$OBJ" --display both \
  --save_dir "$OUT/features_analyze/pca" \
  --mask_path "$MASK" --features_path "$PIXFEAT" --vertex_features_path "$PRED_F"

echo "=== [6/7] features_analyze: similarity heatmap (fixed query order) ==="
python analysis/feature_similarity_heatmap.py --obj_path "$OBJ" \
  --query_pixels $QUERY_PIXELS --query_vertices $QUERY_VERTICES \
  --save_dir "$OUT/features_analyze/sim_heatmap" \
  --mask_path "$MASK" --features_path "$PIXFEAT" --vertex_features_path "$PRED_F"

COV=$(python - "$SOFT" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu')
n = int(c['num_vertices']); k = int(torch.unique(c['top1_vertex_idx']).numel())
print(f'{k}/{n} = {100*k/n:.2f}%')
PY
)
echo "soft-snap coverage (read T=1): $COV"

if [ -n "$EVAL_TAG" ]; then
  echo "=== [7/7] ground-truth eval, EVAL SET v2 (isobbox-only) ==="
  mkdir -p "$EVALDIR"
  python analysis/eval_geodesic_error.py --obj_path "$OBJ" --name "$NAME" \
    --run_label "$EVAL_TAG (${MESH_LABEL:-isobbox 3886v}, eval v2)" \
    --geo_table_path "$GEO3D" --eval_pairs_path "$EVALPAIRS" \
    --features_path "$PIXFEAT" --vertex_features_path "$PRED_F" --k_vertices_per_pixel 10 \
    --save_path "$EVALDIR/eval_${EVAL_TAG}_v2.pth" --plot_path "$EVALDIR/eval_${EVAL_TAG}_v2.png"
else
  echo "=== [7/7] eval SKIPPED (no eval_tag given) ==="
fi
if [ -z "$SMALL" ]; then
  echo "=== analysis_summary (10-panel one-glance grid) ==="
  python analysis/build_analysis_summary.py --analyze_dir "$OUT" --checker_stem "$CHK_STEM" \
    --output_path "$DEMO/analysis_summary.png"
fi
echo "===== ARM ANALYSIS DONE: $DEMO ====="
