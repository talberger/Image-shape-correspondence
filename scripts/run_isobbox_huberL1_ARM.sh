#!/bin/bash
# THE L1 ARM -- the third point on the huber_delta axis. 2026-09-09, Tal's ask.
#
# delta is a transition point on the per-pair residual |norm_d_3D - norm_d_2D|, measured at
# mean 0.057 / max 0.361 (--dump_residuals). So:
#   delta >= 0.36  -> nothing clips     -> pure L2   (delta=1.0, where this project has always been)
#   delta = 0.1    -> p90, 16.7% clip   -> real Huber (the arm running now)
#   delta = 0.001  -> 100% clip         -> pure L1
# Every pair sits in the linear branch at 0.001, so the gradient magnitude is the constant
# +/-delta for every pair regardless of how wrong it is -- that IS L1. The leftover factor of
# delta is a global loss scale, and Adam (eps=1e-12, single loss term) is invariant to it, so
# this is L1 exactly, not L1-times-something.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

BASE_RUN=bsb_isobbox_sdsquare_T1_seed0_phase2
BASE_TAG=isobbox_sdsquare_fuse16_truediag2d_T1_seed0
LDIR=./experiments/guitar/encoder_isobbox_sdsquare_fuse16_truediag2d_huberL1_T1_seed0_phase2
LRUN=bsb_isobbox_sdsquare_huberL1_T1_seed0_phase2

if [ -f "$LDIR/pred_f.pth" ]; then echo "already trained"; else
  time python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" \
    --init_encoder_dir "$PHASE1" \
    --object_mask_path "$MASK" --fine_seg_path "$FINE" \
    --pix_dino_features_path "$PIX_sdsquare_fuse16" \
    --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 1.0 --train_seed 0 \
    --geo_diag_2d 1175.75 --huber_delta 0.001 \
    --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
    --out_dir "$LDIR"
fi

./scripts/analyze_isobbox_arm.sh "$LRUN" "$LDIR/pred_f.pth" "$PIX_sdsquare_fuse16" \
  isobbox_sdsquare_fuse16_truediag2d_huberL1_T1_seed0

./scripts/compare_isobbox_arms.sh \
  "$LRUN" "huber_delta 0.001 = pure L1 (100% of pairs clipped)" \
  "$BASE_RUN" "BASELINE, huber_delta 1.0 = pure L2 (0% clipped)" \
  comparison_vs_baseline \
  isobbox_sdsquare_fuse16_truediag2d_huberL1_T1_seed0 "$BASE_TAG"

echo "===== L1 ARM DONE ====="
