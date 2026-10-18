#!/bin/bash
# Head-to-head comparison of TWO isobbox arms, written into the FIRST arm's own folder.
# 2026-09-08, Tal: the comparison must also carry the FEATURE ANALYSIS, not only the
# back-projections -- so PCA and the similarity heatmap are stacked here too.
#
# Usage: ./compare_isobbox_arms.sh <demo_A> <label_A> <demo_B> <label_B> <out_subdir> \
#                                  [eval_tag_A] [eval_tag_B]
# Everything is read T=1. Both arms must already have been through analyze_isobbox_arm.sh.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

A=$1; LA=$2; B=$3; LB=$4; SUB=$5; TA=${6:-}; TB=${7:-}
[ -z "$SUB" ] && { echo "usage: $0 <demo_A> <label_A> <demo_B> <label_B> <out_subdir> [eval_tag_A] [eval_tag_B]"; exit 1; }
DA=$(run_dir "$A")/analyze; DB=$(run_dir "$B")/analyze
OUT=$(run_dir "$A")/$SUB
mkdir -p "$OUT"
for d in "$DA" "$DB"; do [ -d "$d" ] || { echo "missing analyze dir: $d"; exit 1; }; done

BP=back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1_side_by_side.png
BPK1=back_projection/${MB}_backproj_${CHK_STEM}_k1_side_by_side.png
PCA=features_analyze/pca/guitar_isotropic_bboxnorm_mesh_pca_shared_color_side_by_side.png
# 2026-09-14, Tal: the similarity heatmap must be ONE ROW of 4 per arm, rows stacked, so each panel sits
# directly above the matching panel of the other arm. The 2x2 grid breaks that alignment. Same layout
# compare_isobbox_3way.sh has used since 2026-09-08; the strip is built here if the arm lacks it.
SIM=features_analyze/sim_heatmap/simheatmap_row.png
build_row () {  # <analyze_dir> ; needs the arm's pred_f + pixel features, found from its run_config
  local D=$1/features_analyze/sim_heatmap
  [ -f "$D/simheatmap_row.png" ] && return 0
  local PF=$2
  [ -f "$PF" ] || { echo "  no pred_f for $1, cannot build the row strip"; return 1; }
  python analysis/feature_similarity_heatmap.py --obj_path "$OBJ" \
    --query_pixels $QUERY_PIXELS --query_vertices $QUERY_VERTICES \
    --save_dir "$D" --grid_rows 1 --grid_output_path "$D/simheatmap_row.png" \
    --mask_path "$MASK" --features_path "${PIX_FEATS:-$PIX_sdsquare_fuse16}" --vertex_features_path "$PF"
}

echo "=== [1/5] back-projection, soft-snap k=1 (read T=1) ==="
python analysis/stack_images_labeled.py \
  --top_path "$DA/$BP" --top_label "$LA (soft-snap, read T=1)" \
  --bottom_path "$DB/$BP" --bottom_label "$LB (soft-snap, read T=1)" \
  --title "back-projection soft-snap k=1 -- $LA vs $LB" \
  --output_path "$OUT/1_back_projection_soft_readT1.png"

echo "=== [2/5] back-projection, top1k (read T=1) ==="
python analysis/stack_images_labeled.py \
  --top_path "$DA/$BPK1" --top_label "$LA (top1k, read T=1)" \
  --bottom_path "$DB/$BPK1" --bottom_label "$LB (top1k, read T=1)" \
  --title "back-projection top1k -- $LA vs $LB" \
  --output_path "$OUT/2_back_projection_top1k_readT1.png"

echo "=== [3/5] features_analyze: PCA, shared basis ==="
if [ -f "$DA/$PCA" ] && [ -f "$DB/$PCA" ]; then
  python analysis/stack_images_labeled.py \
    --top_path "$DA/$PCA" --top_label "$LA (PCA, shared basis image+mesh)" \
    --bottom_path "$DB/$PCA" --bottom_label "$LB (PCA, shared basis image+mesh)" \
    --title "PCA feature compare -- $LA vs $LB" \
    --output_path "$OUT/3_features_pca.png"
else
  echo "  SKIPPED: PCA missing in one arm (run analyze_isobbox_arm.sh first)"
fi

echo "=== [4/5] features_analyze: similarity heatmap (one row per arm) ==="
# PF_A / PF_B (2026-09-14): the arms' pred_f.pth, set by the queue scripts, so a missing row strip can be
# built here. Without them an arm that only has the old 2x2 grid falls back to it.
build_row "$DA" "${PF_A:-}" || true
build_row "$DB" "${PF_B:-}" || true
if [ -f "$DA/$SIM" ] && [ -f "$DB/$SIM" ]; then
  # Both strips must have been built with the SAME --query_vertices ORDER; lib_isobbox_common.sh
  # pins it, because the panels are laid out in argument order (2026-09-07 gotcha).
  python analysis/stack_images_labeled.py \
    --top_path "$DA/$SIM" --top_label "$LA (sim heatmap, queries $QUERY_VERTICES)" \
    --bottom_path "$DB/$SIM" --bottom_label "$LB (sim heatmap, queries $QUERY_VERTICES)" \
    --title "feature similarity heatmap, one row per arm -- $LA vs $LB" \
    --output_path "$OUT/4_features_sim_heatmap.png"
elif [ -f "$DA/features_analyze/sim_heatmap/simheatmap_grid.png" ] && [ -f "$DB/features_analyze/sim_heatmap/simheatmap_grid.png" ]; then
  echo "  FALLBACK to the 2x2 grid: no row strip and no pred_f (set PF_A/PF_B to get the row layout)"
  python analysis/stack_images_labeled.py \
    --top_path "$DA/features_analyze/sim_heatmap/simheatmap_grid.png" --top_label "$LA (sim heatmap, queries $QUERY_VERTICES)" \
    --bottom_path "$DB/features_analyze/sim_heatmap/simheatmap_grid.png" --bottom_label "$LB (sim heatmap, queries $QUERY_VERTICES)" \
    --title "feature similarity heatmap -- $LA vs $LB" \
    --output_path "$OUT/4_features_sim_heatmap.png"
else
  echo "  SKIPPED: sim heatmap missing in one arm"
fi

echo "=== [5/5] ground-truth eval, v2 ==="
if [ -n "$TA" ] && [ -n "$TB" ] && [ -f "$EVALDIR/eval_${TA}_v2.pth" ] && [ -f "$EVALDIR/eval_${TB}_v2.pth" ]; then
  # Same mesh and same eval set on both sides, so NO --rescale_across_meshes.
  python analysis/compare_eval_runs.py \
    --runs "$EVALDIR/eval_${TA}_v2.pth" "$EVALDIR/eval_${TB}_v2.pth" \
    --labels "$LA" "$LB" \
    --save_path "$OUT/5_eval_v2.png"
  cp "$OUT/5_eval_v2.png" "$(run_dir "$A")/eval/eval_vs_${B}_v2.png" 2>/dev/null || true
else
  echo "  SKIPPED: need both eval tags with existing v2 .pth files"
fi
echo "===== COMPARISON DONE: $OUT ====="
