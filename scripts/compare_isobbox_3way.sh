#!/bin/bash
# Three-way comparison of the fuse arms, at ZS or at phase-2 -- back-projections, feature analysis
# (PCA + similarity heatmap, ONE LONG ROW PER ARM stacked one above the other), and one eval figure
# carrying all three. Everything read T=1, eval v2.
#
# Usage: ./compare_isobbox_3way.sh ZS|P2
#
# Generalizes compare_isobbox_ZS_3way.sh + compare_isobbox_ZS_3way_featurerow.sh (2026-09-08),
# which were written for the ZS set only; those are kept as the record of how the ZS figures were
# built. Assembly only -- every arm must already have been through analyze_isobbox_arm.sh -- except
# the one-row similarity strip, which does not exist yet for an arm and is built here.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh

MODE=${1:-}
EXP=./experiments/guitar
case "$MODE" in
  ZS)
    A=bsb_isobbox_ZS_sdsquare; PA=$EXP/encoder_isobbox_dino_sd_dfd_crossview_phase1/pred_f.pth
    FA=$PIX_sdsquare_fuse16; EA=isobbox_ZS_sdsquare_fuse16
    LA="ZS sdsquare_fuse16  (2.7% mean err, 87.9/86.2/85.7)"
    B=bsb_isobbox_ZS_sdsquare_fuse32; PB=$EXP/encoder_isobbox_fuse32_phase1/pred_f.pth
    FB=$PIX_sdsquare_fuse32; EB=isobbox_ZS_sdsquare_fuse32
    LB="ZS sdsquare_fuse32  (3.2% mean err, 83.6/84.2/81.4)"
    C=bsb_isobbox_ZS_sdsquash; PC=$EXP/encoder_isobbox_dino_sd_dfd_crossview_phase1/pred_f.pth
    FC=$PIX_sdsquash_fuse16; EC=isobbox_ZS_sdsquash_fuse16
    LC="ZS sdsquash_fuse16  (6.3% mean err, 68.8/68.8/68.0)"
    TITLE="ZS (phase-1 only)"
    ;;
  P2)
    # experiments/ was never renamed (run_config.json stores absolute out_dir paths), so the
    # encoder dirs still carry the old long names while the demo dirs carry the new ones.
    A=bsb_isobbox_sdsquare_olddiag2d_T1_seed0_phase2
    PA=$EXP/encoder_isobbox_dino_sd_stride14_sdsquare_relational_geodesic_mu3dscale1_T1_seed0_phase2/pred_f.pth
    FA=$PIX_sdsquare_fuse16; EA=isobbox_sdsquare_fuse16_T1_seed0
    LA="P2 sdsquare_fuse16 T=1  (4.7% mean err, 76.3/76.7/75.0)"
    B=bsb_isobbox_sdsquare_fuse32_T1_seed0_phase2
    PB=$EXP/encoder_isobbox_sdsquare_fuse32_T1_seed0_phase2/pred_f.pth
    FB=$PIX_sdsquare_fuse32; EB=isobbox_sdsquare_fuse32_T1_seed0
    LB="P2 sdsquare_fuse32 T=1  (4.6% mean err, 76.4/77.2/74.4)"
    C=bsb_isobbox_sdsquash_T1_seed0_phase2
    PC=$EXP/encoder_isobbox_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T1_seed0_phase2/pred_f.pth
    FC=$PIX_sdsquash_fuse16; EC=isobbox_sdsquash_fuse16_T1_seed0
    LC="P2 sdsquash_fuse16 T=1  (4.7% mean err, 77.6/76.4/74.8)"
    TITLE="phase-2, T=1 seed0"
    ;;
  *) echo "usage: $0 ZS|P2"; exit 1;;
esac

DA=$RUNS/$A/analyze; DB=$RUNS/$B/analyze; DC=$RUNS/$C/analyze
OUT=$RUNS/$A/comparison_3way
mkdir -p "$OUT" "$RUNS/$A/eval"
for d in "$DA" "$DB" "$DC"; do [ -d "$d" ] || { echo "missing analyze dir: $d"; exit 1; }; done

BP=back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1_side_by_side.png
BPK1=back_projection/${MB}_backproj_${CHK_STEM}_k1_side_by_side.png
PCA=features_analyze/pca/guitar_isotropic_bboxnorm_mesh_pca_shared_color_side_by_side.png
ROW=features_analyze/sim_heatmap/simheatmap_row.png

build_row () {  # <analyze_dir> <pred_f> <pix_features>
  local D=$1/features_analyze/sim_heatmap
  [ -f "$D/simheatmap_row.png" ] && { echo "  already built: $D/simheatmap_row.png"; return; }
  python analysis/feature_similarity_heatmap.py --obj_path "$OBJ" \
    --query_pixels $QUERY_PIXELS --query_vertices $QUERY_VERTICES \
    --save_dir "$D" --grid_rows 1 --grid_output_path "$D/simheatmap_row.png" \
    --mask_path "$MASK" --features_path "$3" --vertex_features_path "$2"
}

echo "=== [1/5] one-row similarity strips (built if missing) ==="
build_row "$DA" "$PA" "$FA"
build_row "$DB" "$PB" "$FB"
build_row "$DC" "$PC" "$FC"

echo "=== [2/5] back-projection, soft-snap k=1 (read T=1) ==="
python analysis/stack_images_nway.py \
  --panel "$DA/$BP" "$LA -- soft-snap, read T=1" \
  --panel "$DB/$BP" "$LB -- soft-snap, read T=1" \
  --panel "$DC/$BP" "$LC -- soft-snap, read T=1" \
  --title "$TITLE -- back-projection, soft-snap k=1, read T=1" \
  --output_path "$OUT/1_back_projection_soft_readT1.png"

echo "=== [3/5] back-projection, top1k (read T=1) ==="
python analysis/stack_images_nway.py \
  --panel "$DA/$BPK1" "$LA -- top1k, read T=1" \
  --panel "$DB/$BPK1" "$LB -- top1k, read T=1" \
  --panel "$DC/$BPK1" "$LC -- top1k, read T=1" \
  --title "$TITLE -- back-projection, top1k, read T=1" \
  --output_path "$OUT/2_back_projection_top1k_readT1.png"

echo "=== [4/5] features_analyze: PCA rows + similarity strips ==="
python analysis/stack_images_nway.py \
  --panel "$DA/$PCA" "$LA -- PCA, shared basis image | mesh" \
  --panel "$DB/$PCA" "$LB -- PCA, shared basis image | mesh" \
  --panel "$DC/$PCA" "$LC -- PCA, shared basis image | mesh" \
  --title "$TITLE -- PCA, shared basis, one row per arm" \
  --output_path "$OUT/3_features_pca.png"
# All three strips share the pinned --query_vertices ORDER from lib_isobbox_common.sh, so the same
# query sits in the same column in every row; a different order would compare different vertices.
python analysis/stack_images_nway.py \
  --panel "$DA/$ROW" "$LA -- sim heatmap" \
  --panel "$DB/$ROW" "$LB -- sim heatmap" \
  --panel "$DC/$ROW" "$LC -- sim heatmap" \
  --title "$TITLE -- similarity heatmap, one row per arm: pixels (151,94) (571,206) (820,360) (820,740), vertices $QUERY_VERTICES" \
  --output_path "$OUT/4_features_sim_heatmap_rows.png"

echo "=== [5/5] ground-truth eval v2, all three arms ==="
# Same mesh and same eval set on all three, so NO --rescale_across_meshes.
python analysis/compare_eval_runs.py \
  --runs "$EVALDIR/eval_${EA}_v2.pth" "$EVALDIR/eval_${EB}_v2.pth" "$EVALDIR/eval_${EC}_v2.pth" \
  --labels "$LA" "$LB" "$LC" \
  --save_path "$OUT/5_eval_v2_3way.png"
cp "$OUT/5_eval_v2_3way.png" "$RUNS/$A/eval/eval_3way_v2.png"

echo "===== 3-WAY ($MODE) DONE: /workspace/bsb/$OUT ====="
