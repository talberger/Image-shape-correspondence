#!/bin/bash
# Three-way ZS comparison of the fuse arms -- back-projections, feature analysis, and one eval
# figure carrying all three. 2026-09-08 (Tal): the ZS arms are the ones worth comparing, because
# phase-2 converges every arm to ~4.7% mean error regardless of its input.
#
# Everything is read T=1 and eval v2. All three arms must already have been through
# analyze_isobbox_arm.sh (they were, on 2026-09-08) -- this script only assembles, it does not
# recompute. Output goes into the BEST arm's own folder, per the "a run's own comparison lives in
# that run's folder" rule.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh

A=bsb_isobbox_ZS_sdsquare;  LA="ZS sdsquare_fuse16  (2.7% mean err, 87.9/86.2/85.7)"
B=bsb_isobbox_ZS_sdsquare_fuse32;  LB="ZS sdsquare_fuse32  (3.2% mean err, 83.6/84.2/81.4)"
C=bsb_isobbox_ZS_sdsquash;  LC="ZS sdsquash_fuse16  (6.3% mean err, 68.8/68.8/68.0)"

DA=$RUNS/$A/analyze; DB=$RUNS/$B/analyze; DC=$RUNS/$C/analyze
OUT=$RUNS/$A/comparison_ZS_3way
mkdir -p "$OUT"
for d in "$DA" "$DB" "$DC"; do [ -d "$d" ] || { echo "missing analyze dir: $d"; exit 1; }; done

BP=back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1_side_by_side.png
BPK1=back_projection/${MB}_backproj_${CHK_STEM}_k1_side_by_side.png
PCA=features_analyze/pca/guitar_isotropic_bboxnorm_mesh_pca_shared_color_side_by_side.png
SIM=features_analyze/sim_heatmap/simheatmap_grid.png

echo "=== [1/5] back-projection, soft-snap k=1 (read T=1) ==="
python analysis/stack_images_nway.py \
  --panel "$DA/$BP" "$LA -- soft-snap, read T=1" \
  --panel "$DB/$BP" "$LB -- soft-snap, read T=1" \
  --panel "$DC/$BP" "$LC -- soft-snap, read T=1" \
  --title "ZS back-projection, soft-snap k=1, read T=1 -- three fuse arms" \
  --output_path "$OUT/1_back_projection_soft_readT1.png"

echo "=== [2/5] back-projection, top1k (read T=1) ==="
python analysis/stack_images_nway.py \
  --panel "$DA/$BPK1" "$LA -- top1k, read T=1" \
  --panel "$DB/$BPK1" "$LB -- top1k, read T=1" \
  --panel "$DC/$BPK1" "$LC -- top1k, read T=1" \
  --title "ZS back-projection, top1k, read T=1 -- three fuse arms" \
  --output_path "$OUT/2_back_projection_top1k_readT1.png"

echo "=== [3/5] features_analyze: PCA, shared basis ==="
python analysis/stack_images_nway.py \
  --panel "$DA/$PCA" "$LA -- PCA, shared basis image+mesh" \
  --panel "$DB/$PCA" "$LB -- PCA, shared basis image+mesh" \
  --panel "$DC/$PCA" "$LC -- PCA, shared basis image+mesh" \
  --title "ZS PCA feature compare -- three fuse arms" \
  --output_path "$OUT/3_features_pca.png"

echo "=== [4/5] features_analyze: similarity heatmap ==="
# All three grids were built with the SAME --query_vertices ORDER (lib_isobbox_common.sh pins it);
# panels are laid out in argument order, so a different order silently compares different vertices.
python analysis/stack_images_nway.py \
  --panel "$DA/$SIM" "$LA -- sim heatmap, queries $QUERY_VERTICES" \
  --panel "$DB/$SIM" "$LB -- sim heatmap, queries $QUERY_VERTICES" \
  --panel "$DC/$SIM" "$LC -- sim heatmap, queries $QUERY_VERTICES" \
  --title "ZS feature similarity heatmap -- three fuse arms" \
  --output_path "$OUT/4_features_sim_heatmap.png"

echo "=== [5/5] ground-truth eval v2, all three arms in one figure ==="
# Same mesh and same eval set on all three, so NO --rescale_across_meshes.
python analysis/compare_eval_runs.py \
  --runs "$EVALDIR/eval_isobbox_ZS_sdsquare_fuse16_v2.pth" \
         "$EVALDIR/eval_isobbox_ZS_sdsquare_fuse32_v2.pth" \
         "$EVALDIR/eval_isobbox_ZS_sdsquash_fuse16_v2.pth" \
  --labels "ZS sdsquare_fuse16" "ZS sdsquare_fuse32" "ZS sdsquash_fuse16" \
  --save_path "$OUT/5_eval_v2_3way.png"
cp "$OUT/5_eval_v2_3way.png" "$RUNS/$A/eval/eval_ZS_3way_v2.png"

echo "===== ZS 3-WAY COMPARISON DONE: /workspace/bsb/$OUT ====="
