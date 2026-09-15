#!/bin/bash
# Three-way comparison, 2026-09-15: ZS / in-segment-only + closed-form per-segment mu (best arm of 09-14) /
# in-segment-only + LEARNED per-segment mu. Assembly only -- all three arms are already analyzed.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh

A=$RUNS/bsb_isobbox_ZS_sdsquare;                                   LA="ZS (phase-1 only)"
B=$PROP_RUNS/bsb_isobbox_sdsquare_mu_closed_perseg_T1_seed0_phase2;    LB="L_spatial only, 16 in-segment nbrs, mu per segment CLOSED-FORM"
C=$PROP_RUNS/bsb_isobbox_sdsquare_mu_learn_perseg_T1_seed0_phase2;  LC="L_spatial only, 16 in-segment nbrs, mu per segment LEARNED (lr 1e-2)"
TA=isobbox_ZS_sdsquare_fuse16
TB=isobbox_sdsquare_fuse16_truediag2d_mu_closed_perseg_T1_seed0
TC=isobbox_sdsquare_fuse16_truediag2d_mu_learn_perseg_T1_seed0
OUT=$C/comparison_vs_closedform_and_ZS
mkdir -p "$OUT"

BP=analyze/back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1_side_by_side.png
BPK1=analyze/back_projection/${MB}_backproj_${CHK_STEM}_k1_side_by_side.png
PCA=analyze/features_analyze/pca/guitar_isotropic_bboxnorm_mesh_pca_shared_color_side_by_side.png
SIM=analyze/features_analyze/sim_heatmap/simheatmap_row.png

stack() {  # <rel path> <label suffix> <title> <output>
  python analysis/stack_images_nway.py \
    --panel "$A/$1" "$LA$2" --panel "$B/$1" "$LB$2" --panel "$C/$1" "$LC$2" \
    --title "$3" --output_path "$OUT/$4"
}
stack "$BP"   " (soft-snap, read T=1)" "back-projection soft-snap k=1 -- closed-form vs learned per-segment mu" 1_back_projection_soft_readT1.png
stack "$BPK1" " (top1k, read T=1)"     "back-projection top1k -- closed-form vs learned per-segment mu"        2_back_projection_top1k_readT1.png
stack "$PCA"  " (PCA, shared basis)"   "PCA feature compare -- closed-form vs learned per-segment mu"          3_features_pca.png
stack "$SIM"  " (sim heatmap, queries $QUERY_VERTICES)" "feature similarity heatmap, one row per arm"          4_features_sim_heatmap.png
python analysis/compare_eval_runs.py \
  --runs "$EVALDIR/eval_${TA}_v2.pth" "$EVALDIR/eval_${TB}_v2.pth" "$EVALDIR/eval_${TC}_v2.pth" \
  --labels "$LA" "$LB" "$LC" --save_path "$OUT/5_eval_v2.png"
echo "===== 3WAY DONE: $OUT ====="
