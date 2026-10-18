#!/bin/bash
# ZS ANCHOR arms (2026-09-10). L = L_rel + lambda * L_anchor, modules/anchor_loss.py.
# Every arm = the truediag baseline recipe + the anchor, nothing else changed.
#
# Usage: ./run_isobbox_anchor_QUEUE.sh <soft|hard> <lambda> [<lambda> ...]
#   soft : q = softmax(S_ZS / 0.01) over all 3886 vertices (our version)
#   hard : one-hot argmax of the same similarity (SHIC's L_pseudo, the control)
# lambda values come from scripts/run_isobbox_anchor_CALIBRATE.sh (equal gradient norms at ladder step 25).
# Each arm gets the full analysis battery, then comparisons against BOTH the phase-2 baseline and ZS.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

MODE=$1; shift
[ "$MODE" = soft ] || [ "$MODE" = hard ] || { echo "usage: $0 <soft|hard> <lambda> [<lambda> ...]"; exit 1; }
[ $# -ge 1 ] || { echo "usage: $0 <soft|hard> <lambda> [<lambda> ...]"; exit 1; }
EXTRA=""; [ "$MODE" = hard ] && EXTRA="--anchor_hard"
# ANCHOR_T env (Tal, 2026-09-10: try anchor T=1). Recalibrate lambda for any T other than 0.01.
ANCHOR_T=${ANCHOR_T:-0.01}
# Small analysis by default (Tal, 2026-09-10): back-projection + PCA + sim heatmap + eval, no mesh
# renders. Run with SMALL= (empty) for the full battery.
export SMALL=${SMALL-1}

# MU=0.75 (Tal, 2026-09-10): rerun the anchor arms on top of --mu_3d_scale 0.75. The matched control
# is then the mu=0.75 arm WITHOUT the anchor, not the mu=1 baseline. Default 1 = original behaviour.
MU=${MU:-1}
if [ "$MU" = 1 ]; then
  MUTAG=""
  BASE_RUN=bsb_isobbox_sdsquare_T1_seed0_phase2
  BASE_TAG=isobbox_sdsquare_fuse16_truediag2d_T1_seed0
  BASE_LABEL="BASELINE phase-2 (no anchor)"
else
  MUTAG=_mu${MU}
  BASE_RUN=bsb_isobbox_sdsquare${MUTAG}_T1_seed0_phase2
  BASE_TAG=isobbox_sdsquare_fuse16_truediag2d${MUTAG}_T1_seed0
  BASE_LABEL="mu=$MU phase-2 (no anchor)"
fi
ZS_RUN=bsb_isobbox_ZS_sdsquare
ZS_TAG=isobbox_ZS_sdsquare_fuse16

for W in "$@"; do
  SUFFIX=sdsquare_fuse16_truediag2d${MUTAG}_anchor${MODE}T${ANCHOR_T}_w${W}_T1_seed0
  DIR=./experiments/guitar/encoder_isobbox_${SUFFIX}_phase2
  # demo dirs dropped _fuse16/_truediag2d on 2026-09-10 (Tal); encoder dirs and eval tags keep the long names
  RUN=bsb_isobbox_${SUFFIX/_fuse16_truediag2d/}_phase2
  TAG=isobbox_${SUFFIX}
  echo "===================== ANCHOR $MODE lambda=$W ====================="

  if [ -f "$DIR/pred_f.pth" ]; then echo "already trained: $DIR"; else
    time python phase2/train_relational_loss_l2norm.py \
      --obj_path "$OBJ" --name "$NAME" \
      --init_encoder_dir "$PHASE1" \
      --object_mask_path "$MASK" --fine_seg_path "$FINE" \
      --pix_dino_features_path "$PIX_sdsquare_fuse16" \
      --width 2304 --n_classes 2304 --geodesic --mu_3d_scale "$MU" --temperature 1.0 --train_seed 0 \
      --geo_diag_2d 1175.75 \
      --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
      --anchor_weight "$W" --anchor_temperature "$ANCHOR_T" $EXTRA \
      --out_dir "$DIR"
  fi

  ./scripts/analyze_isobbox_arm.sh "$RUN" "$DIR/pred_f.pth" "$PIX_sdsquare_fuse16" "$TAG"

  ./scripts/compare_isobbox_arms.sh \
    "$RUN" "anchor $MODE T=$ANCHOR_T lambda=$W mu=$MU" \
    "$BASE_RUN" "$BASE_LABEL" \
    comparison_vs_baseline "$TAG" "$BASE_TAG"

  ./scripts/compare_isobbox_arms.sh \
    "$RUN" "anchor $MODE T=$ANCHOR_T lambda=$W" \
    "$ZS_RUN" "ZS (phase-1 only)" \
    comparison_vs_ZS "$TAG" "$ZS_TAG"
done
echo "===== ANCHOR $MODE QUEUE DONE: $* ====="
