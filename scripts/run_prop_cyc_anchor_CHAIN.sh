#!/bin/bash
# 2026-09-15, Tal: MAIN baseline + cycle only, and MAIN baseline + anchor only.
# Waits for the running queue (PID arg) so the GPU runs one job at a time, then recalibrates both terms
# at step 25 of mu_learn_perseg and launches the two arms. Ratios agreed with Tal, matching where the old
# weights sat against the OLD L_spatial: anchor at 1/1 of L_spatial's grad norm (old lambda 3.5e-7 was
# exactly the equal-norm point), cycle at 1/5 (old c=1.05e-5 was ~1/5).
# Usage: ./run_prop_cyc_anchor_CHAIN.sh <pid_to_wait_for>
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh

WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  echo "waiting for PID $WAIT_PID ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi

CK=experiments/guitar/encoder_isobbox_sdsquare_fuse16_truediag2d_mu_learn_perseg_T1_seed0_phase2/encoder_checkpoint_step25.pth
CAL=/workspace/logs/prop_cyc_anchor_calibrate.log
echo "===== calibration at $CK ====="
python phase2/train_relational_loss_l2norm.py \
  --obj_path "$OBJ" --name "$NAME" --init_encoder_dir "$PHASE1" \
  --object_mask_path "$MASK" --fine_seg_path "$FINE" --pix_dino_features_path "$PIX_sdsquare_fuse16" \
  --width 2304 --n_classes 2304 --geodesic --temperature 1.0 --train_seed 0 --geo_diag_2d 1175.75 \
  --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 \
  --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
  --mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs \
  --anchor_temperature 0.01 --anchor_weight 3.5e-7 --anchor_balance_from "$CK" \
  --cyc_weight 1 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5 \
  --out_dir experiments/guitar/_prop_cyc_anchor_calibrate_scratch 2>&1 | tee "$CAL"

G_REL=$(grep -m1 "relational: loss" "$CAL" | sed -E 's/.*grad norm ([0-9.e+-]+).*/\1/')
LAM=$(grep -m1 "lambda at equal grad norms" "$CAL" | sed -E 's/.*: ([0-9.e+-]+).*/\1/')
G_CYC=$(grep -m1 "cycle v2  : loss" "$CAL" | sed -E 's/.*grad norm ([0-9.e+-]+).*/\1/')
[ -n "$G_REL" ] && [ -n "$LAM" ] && [ -n "$G_CYC" ] || { echo "calibration parse FAILED (rel=$G_REL lam=$LAM cyc=$G_CYC)"; exit 1; }
LAM=$(python -c "print('%.4g' % float('$LAM'))")
CW=$(python -c "print('%.4g' % (float('$G_REL') / (5.0 * float('$G_CYC'))))")
echo "L_spatial grad $G_REL | anchor lambda (1/1) $LAM | cycle grad $G_CYC -> cyc_weight (1/5) $CW"

VS_SECONDARY=1 ./scripts/run_isobbox_prop_QUEUE.sh \
  "cycv2T0.05_c${CW}_retstride5|--cyc_weight $CW --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5|+ cycle v2 only (T_cyc=0.05, c=$CW = 1/5 of L_spatial grad, stride 5)" \
  "anchorsoftT0.01_w${LAM}|--anchor_weight $LAM --anchor_temperature 0.01|+ soft anchor only (T=0.01, lambda=$LAM = 1/1 of L_spatial grad)"
echo "===== CHAIN DONE ====="
