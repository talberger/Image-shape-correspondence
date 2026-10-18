#!/bin/bash
# 2026-09-15, Tal's verdicts on the first angle / cycle / anchor arms (see PROJECT_NOTES.md):
#   1. ANGLE-SPATIAL stronger: 3x and 10x the 1/1 weight (8.8313e-3).
#   2. The anchor + cycle + L_dist combination ON THE NEW MAIN, anchor weakened a little (1/2 of the
#      equal-norm point), cycle unchanged, L_dist at SHIC's 1/50 of the weighted anchor (calibrated here).
# Then the two comparison sets. One GPU job at a time.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

ANCHOR_W=3.11e-07          # 1/2 of 6.221e-7 (equal grad norm with L_spatial at MAIN step 25)
CYC_W=1.676e-05            # 1/5 of L_spatial grad, unchanged

echo "===== 1/4 stronger spatial angle ====="
./scripts/run_isobbox_prop_QUEUE.sh \
  "sangle3_w2.65e-2|--rkd_spatial_angle3_weight 2.6494e-02|+ rkd_spatial_angle3 (Euclidean), w=2.65e-2 (3x of 1/1)" \
  "sangle3_w8.83e-2|--rkd_spatial_angle3_weight 8.8313e-02|+ rkd_spatial_angle3 (Euclidean), w=8.83e-2 (10x of 1/1)"

echo "===== 2/4 L_dist calibration at MAIN step 25 (anchor lambda $ANCHOR_W) ====="
CK=experiments/guitar/encoder_isobbox_sdsquare_fuse16_truediag2d_mu_learn_perseg_T1_seed0_phase2/encoder_checkpoint_step25.pth
CAL=$DISC_WS/logs/prop_dist_calibrate.log
python phase2/train_relational_loss_l2norm.py \
  --obj_path "$OBJ" --name "$NAME" --init_encoder_dir "$PHASE1" \
  --object_mask_path "$MASK" --fine_seg_path "$FINE" --pix_dino_features_path "$PIX_sdsquare_fuse16" \
  --width 2304 --n_classes 2304 --geodesic --temperature 1.0 --train_seed 0 --geo_diag_2d 1175.75 \
  --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 \
  --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
  --mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs \
  --anchor_temperature 0.01 --anchor_weight "$ANCHOR_W" --anchor_balance_from "$CK" \
  --dist_balance --dist_temperature 1 \
  --out_dir experiments/guitar/_prop_dist_calibrate_scratch 2>&1 | tee "$CAL"
G=$(grep -m1 "dist_weight at 1/50 of anchor" "$CAL" | sed -E 's/.*: ([0-9.e+-]+)$/\1/')
[ -n "$G" ] || { echo "L_dist calibration parse FAILED"; exit 1; }
G=$(python -c "print('%.4g' % float('$G'))")
echo "L_dist gamma at 1/50 of weighted anchor: $G"

echo "===== 3/4 MAIN + anchor (weakened) + cycle + L_dist ====="
COMBO="anchorsoftT0.01_w${ANCHOR_W}_cycv2T0.05_c${CYC_W}_retstride5_distT1_g${G}"
VS_SECONDARY=1 ./scripts/run_isobbox_prop_QUEUE.sh \
  "${COMBO}|--anchor_weight $ANCHOR_W --anchor_temperature 0.01 --cyc_weight $CYC_W --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5 --dist_weight $G --dist_temperature 1|+ soft anchor lambda=$ANCHOR_W (1/2) + cycle c=$CYC_W (1/5) + L_dist gamma=$G (1/50 of anchor)"

echo "===== 4/4 comparisons ====="
M=bsb_isobbox_sdsquare_mu_learn_perseg
T=isobbox_sdsquare_fuse16_truediag2d_mu_learn_perseg
BP=analyze/back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1_side_by_side.png
SIM=analyze/features_analyze/sim_heatmap/simheatmap_row.png
PCA=analyze/features_analyze/pca/guitar_isotropic_bboxnorm_mesh_pca_shared_color_side_by_side.png
set4() {  # <out_subdir> <title> then 4 x (<run tag suffix, "" = MAIN> <label>)
  local OUT=$PROP_RUNS/$1 TITLE=$2; shift 2; mkdir -p "$OUT"
  local -a P=() R=() L=()
  while [ $# -gt 0 ]; do R+=("$1"); L+=("$2"); shift 2; done
  for spec in "BP:1_back_projection_soft_readT1" "SIM:4_features_sim_heatmap" "PCA:3_features_pca"; do
    local v=${spec%%:*} o=${spec#*:} rel; rel=${!v}; P=()
    for i in "${!R[@]}"; do P+=(--panel "$PROP_RUNS/${M}${R[$i]}_T1_seed0_phase2/$rel" "${L[$i]}"); done
    python analysis/stack_images_nway.py "${P[@]}" --title "$TITLE -- $o" --output_path "$OUT/$o.png"
  done
  local -a EV=("$EVALDIR/eval_isobbox_ZS_sdsquare_fuse16_v2.pth"); local -a EL=("ZS")
  for i in "${!R[@]}"; do EV+=("$EVALDIR/eval_${T}${R[$i]}_T1_seed0_v2.pth"); EL+=("${L[$i]}"); done
  python analysis/compare_eval_runs.py --runs "${EV[@]}" --labels "${EL[@]}" --save_path "$OUT/5_eval_v2.png"
}
set4 comparison_sangle3_strong "rkd_spatial_angle3, stronger weights (Euclidean angles) on MAIN" \
  "" "MAIN baseline (mu_learn_perseg)" \
  "_sangle3_w8.83e-3" "+ rkd_spatial_angle3 w=8.83e-3 (1/1)" \
  "_sangle3_w2.65e-2" "+ rkd_spatial_angle3 w=2.65e-2 (3x)" \
  "_sangle3_w8.83e-2" "+ rkd_spatial_angle3 w=8.83e-2 (10x)"
set4 comparison_combo_new_main "anchor / cycle / L_dist on the NEW main" \
  "" "MAIN baseline (mu_learn_perseg)" \
  "_cycv2T0.05_c1.676e-05_retstride5" "+ cycle only" \
  "_anchorsoftT0.01_w6.221e-07" "+ anchor only (1/1)" \
  "_${COMBO}" "+ anchor 1/2 + cycle + L_dist 1/50"
echo "===== CHAIN DONE ====="
