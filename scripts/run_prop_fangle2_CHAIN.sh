#!/bin/bash
# 2026-09-15, Tal: MAIN (rkd_spatial_dist, learned per-segment mu, 16 in-segment neighbours) + rkd_feat_angle2
# (2-point feature cosine, soft mesh feature, in-segment neighbours). Calibrate by gradient norm at MAIN step 25,
# then three weights (1/10, 1/3, 1/1 of rkd_spatial_dist's grad), then the MAIN + 3 arms comparison.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh
python -m py_compile phase2/train_relational_loss.py

echo "===== 1/3 rkd_feat_angle2 calibration at MAIN step 25 ====="
CK=experiments/guitar/encoder_isobbox_sdsquare_fuse16_truediag2d_mu_learn_perseg_T1_seed0_phase2/encoder_checkpoint_step25.pth
CAL=/workspace/logs/prop_fangle2_calibrate.log
python phase2/train_relational_loss_l2norm.py \
  --obj_path "$OBJ" --name "$NAME" --init_encoder_dir "$PHASE1" \
  --object_mask_path "$MASK" --fine_seg_path "$FINE" --pix_dino_features_path "$PIX_sdsquare_fuse16" \
  --width 2304 --n_classes 2304 --geodesic --temperature 1.0 --train_seed 0 --geo_diag_2d 1175.75 \
  --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 \
  --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
  --mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs \
  --rkd_feat_angle2_neighbors in_segment --anchor_balance_from "$CK" --rkd_feat_angle2_balance \
  --out_dir experiments/guitar/_prop_fangle2_calibrate_scratch 2>&1 | tee "$CAL"
wt() { grep -m1 "rkd_feat_angle2_weight at 1/$1 of rkd_spatial_dist" "$CAL" | sed -E 's/.*: ([0-9.e+-]+)$/\1/'; }
W1=$(wt 1); W3=$(wt 3); W10=$(wt 10)
[ -n "$W1" ] && [ -n "$W3" ] && [ -n "$W10" ] || { echo "calibration parse FAILED"; exit 1; }
# Tag notation (2026-09-15): mantissa to 3 significant digits + plain exponent, e.g. 1.3e-4, 4.33e-5 --
# '%.3g' alone mixed notations (0.00013 vs 4.33e-05).
tagw() { python -c "m, e = ('%.2e' % float('$1')).split('e'); print(('%g' % float(m)) + 'e' + str(int(e)))"; }
S1=$(tagw $W1); S3=$(tagw $W3); S10=$(tagw $W10)
echo "rkd_feat_angle2 weights: 1/1 $W1 | 1/3 $W3 | 1/10 $W10"

echo "===== 2/3 three arms ====="
./scripts/run_isobbox_prop_QUEUE.sh \
  "fangle2_w${S10}|--rkd_feat_angle2_weight $W10 --rkd_feat_angle2_neighbors in_segment|+ rkd_feat_angle2 (soft, in-segment) w=${S10} (1/10)" \
  "fangle2_w${S3}|--rkd_feat_angle2_weight $W3 --rkd_feat_angle2_neighbors in_segment|+ rkd_feat_angle2 (soft, in-segment) w=${S3} (1/3)" \
  "fangle2_w${S1}|--rkd_feat_angle2_weight $W1 --rkd_feat_angle2_neighbors in_segment|+ rkd_feat_angle2 (soft, in-segment) w=${S1} (1/1)"

echo "===== 3/3 comparison: MAIN + 3 arms ====="
M=bsb_isobbox_sdsquare_mu_learn_perseg; T=isobbox_sdsquare_fuse16_truediag2d_mu_learn_perseg
OUT=$PROP_RUNS/comparison_fangle2_sweep; mkdir -p "$OUT"
R=("" "_fangle2_w${S10}" "_fangle2_w${S3}" "_fangle2_w${S1}")
L=("MAIN baseline (mu_learn_perseg)" "+ rkd_feat_angle2 w=${S10} (1/10)" "+ rkd_feat_angle2 w=${S3} (1/3)" "+ rkd_feat_angle2 w=${S1} (1/1)")
for spec in "back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1_side_by_side.png:1_back_projection_soft_readT1" \
            "back_projection/${MB}_backproj_${CHK_STEM}_k1_side_by_side.png:2_back_projection_top1k_readT1" \
            "features_analyze/sim_heatmap/simheatmap_row.png:4_features_sim_heatmap"; do
  rel=${spec%%:*}; o=${spec#*:}; P=()
  for i in 0 1 2 3; do P+=(--panel "$PROP_RUNS/${M}${R[$i]}_T1_seed0_phase2/analyze/$rel" "${L[$i]}"); done
  python analysis/stack_images_nway.py "${P[@]}" --title "rkd_feat_angle2 sweep on MAIN -- $o" --output_path "$OUT/$o.png"
done
EV=("$EVALDIR/eval_isobbox_ZS_sdsquare_fuse16_v2.pth"); EL=("ZS")
for i in 0 1 2 3; do EV+=("$EVALDIR/eval_${T}${R[$i]}_T1_seed0_v2.pth"); EL+=("${L[$i]}"); done
python analysis/compare_eval_runs.py --runs "${EV[@]}" --labels "${EL[@]}" --save_path "$OUT/5_eval_v2.png"
echo "===== CHAIN DONE ====="
