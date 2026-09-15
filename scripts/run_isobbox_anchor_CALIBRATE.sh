#!/bin/bash
# ZS anchor loss, lambda calibration (2026-09-10). Writes nothing but a log.
# The soft anchor's gradient is exactly 0 at phase-1 (p == q), so the balance is measured at the
# step-25 checkpoint of the baseline step ladder -- the state right after the 11-AUC collapse.
# Prints, for soft and hard targets, the lambda at which the anchor and relational gradient norms match.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh

STEP=${1:-25}
CKPT=experiments/guitar/encoder_isobbox_sdsquare_fuse16_T1_seed0_phase2_stepladder/encoder_checkpoint_step${STEP}.pth
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }
SCRATCH=experiments/guitar/_anchor_calibrate_scratch

# ANCHOR_T / MODES env (2026-09-10): lambda depends on T (CE gradient ~ 1/T), so recalibrate per T.
ANCHOR_T=${ANCHOR_T:-0.01}
MODES=${MODES:-"soft hard"}
# CYC_T env (2026-09-10): also report the cycle-v2 gradient and the cyc_weight at SHIC's 1/50 of the
# anchor with weight ANCHOR_W. --cyc_weight 1 only switches the cycle setup on; nothing trains here.
# CYC_STRIDE (2026-09-14): default 5, same as the queue scripts, so the calibration matches what will train.
# Measured 09-13: stride 1 and stride 5 agree on the cycle loss and its gradient norm to within 0.01%.
CYC_STRIDE=${CYC_STRIDE:-5}
CYC_ARGS=""
[ -n "$CYC_T" ] && CYC_ARGS="--cyc_weight 1 --cyc_v2 --cyc_temperature $CYC_T --anchor_weight ${ANCHOR_W:-3.5e-7} ${CYC_ALLPIX:+--cyc_return_all_pixels} $([ "$CYC_STRIDE" != 1 ] && echo --cyc_return_stride $CYC_STRIDE)"
# DIST_T env (2026-09-12): also report SHIC L_dist's gradient and the dist_weight at 1/10 and 1/50 of
# the weighted anchor. Same ANCHOR_W as CYC_ARGS; harmless to set both (--anchor_weight repeats, last wins).
DIST_ARGS=""
[ -n "$DIST_T" ] && DIST_ARGS="--dist_balance --dist_temperature $DIST_T --anchor_weight ${ANCHOR_W:-3.5e-7}"
# DIST_QT env (2026-09-13): soft L_dist target, q_ZS at this T.
[ -n "$DIST_T" ] && [ -n "$DIST_QT" ] && DIST_ARGS="$DIST_ARGS --dist_soft_target_temperature $DIST_QT"
for MODE in $MODES; do
  EXTRA=""; [ "$MODE" = hard ] && EXTRA="--anchor_hard"
  echo "===== $MODE target, balance at ladder step $STEP ====="
  python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" \
    --init_encoder_dir "$PHASE1" \
    --object_mask_path "$MASK" \
    --fine_seg_path "$FINE" \
    --pix_dino_features_path "$PIX_sdsquare_fuse16" \
    --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 1.0 --train_seed 0 \
    --geo_diag_2d 1175.75 \
    --geo_table_3d_path "$GEO3D" \
    --geo_table_2d_path "$GEO2D" \
    --anchor_temperature "$ANCHOR_T" --anchor_balance_from "$CKPT" $EXTRA $CYC_ARGS $DIST_ARGS \
    --out_dir "$SCRATCH"
done
rmdir "$SCRATCH" 2>/dev/null || true
