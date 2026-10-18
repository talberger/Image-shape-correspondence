#!/bin/bash
# Feature analysis of the three ZS arms, laid out as ONE LONG ROW per arm with the arms stacked
# one above the other. 2026-09-08, Tal: the 2x4 similarity grid is hard to compare across arms --
# a single strip per arm puts the same query in the same column all the way down.
#
# The 2x4 grids under each arm's analyze/ are NOT touched; the row version is written alongside
# them as simheatmap_row.png. PCA already renders as one row (image | mesh) per arm, so it only
# needs stacking.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

A=bsb_isobbox_ZS_sdsquare;  LA="ZS sdsquare_fuse16  (2.7% mean err, 87.9/86.2/85.7)"; FA=$PIX_sdsquare_fuse16
B=bsb_isobbox_ZS_sdsquare_fuse32;  LB="ZS sdsquare_fuse32  (3.2% mean err, 83.6/84.2/81.4)"; FB=$PIX_sdsquare_fuse32
C=bsb_isobbox_ZS_sdsquash;  LC="ZS sdsquash_fuse16  (6.3% mean err, 68.8/68.8/68.0)"; FC=$PIX_sdsquash_fuse16

# ZS = phase-1 only. fuse16 arms share the production phase-1; fuse32 has its own (common_grid).
P1_FUSE16=./experiments/guitar/encoder_isobbox_dino_sd_dfd_crossview_phase1/pred_f.pth
P1_FUSE32=./experiments/guitar/encoder_isobbox_fuse32_phase1/pred_f.pth

OUT=$RUNS/$A/comparison_ZS_3way
mkdir -p "$OUT"

build_row () {  # <demo_name> <pred_f> <pix_features>
  local D=$RUNS/$1/analyze/features_analyze/sim_heatmap
  if [ -f "$D/simheatmap_row.png" ]; then echo "  row grid already built: $D/simheatmap_row.png"; return; fi
  python analysis/feature_similarity_heatmap.py --obj_path "$OBJ" \
    --query_pixels $QUERY_PIXELS --query_vertices $QUERY_VERTICES \
    --save_dir "$D" --grid_rows 1 --grid_output_path "$D/simheatmap_row.png" \
    --mask_path "$MASK" --features_path "$3" --vertex_features_path "$2"
}

echo "=== [1/3] one-row similarity strips ==="
build_row "$A" "$P1_FUSE16" "$FA"
build_row "$B" "$P1_FUSE32" "$FB"
build_row "$C" "$P1_FUSE16" "$FC"

echo "=== [2/3] stack the three strips ==="
python analysis/stack_images_nway.py \
  --panel "$RUNS/$A/analyze/features_analyze/sim_heatmap/simheatmap_row.png" "$LA -- sim heatmap" \
  --panel "$RUNS/$B/analyze/features_analyze/sim_heatmap/simheatmap_row.png" "$LB -- sim heatmap" \
  --panel "$RUNS/$C/analyze/features_analyze/sim_heatmap/simheatmap_row.png" "$LC -- sim heatmap" \
  --title "ZS similarity heatmap, one row per arm -- queries: pixels (151,94) (571,206) (820,360) (820,740), vertices 535 549 1017 2885" \
  --output_path "$OUT/4_features_sim_heatmap_rows.png"

echo "=== [3/3] stack the PCA rows (already image | mesh per arm) ==="
PCA=analyze/features_analyze/pca/guitar_isotropic_bboxnorm_mesh_pca_shared_color_side_by_side.png
python analysis/stack_images_nway.py \
  --panel "$RUNS/$A/$PCA" "$LA -- PCA, shared basis image | mesh" \
  --panel "$RUNS/$B/$PCA" "$LB -- PCA, shared basis image | mesh" \
  --panel "$RUNS/$C/$PCA" "$LC -- PCA, shared basis image | mesh" \
  --title "ZS PCA, shared basis, one row per arm" \
  --output_path "$OUT/3_features_pca.png"

echo "===== DONE: $DISC_ROOT/$OUT ====="
