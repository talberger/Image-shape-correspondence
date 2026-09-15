#!/bin/bash
# SHIC L_dist on top of the ZS anchor + cycle v2 (2026-09-12).
#   L = REL_W*L_rel + ANCHOR_W*L_anchor + CYC_W*L_cyc_v2 + gamma*L_dist
# modules/dist_loss.py: L_dist = sum_x p(x|u) * d_geo(x, x~_u) / D, HARD x~_u = argmax q_ZS.
# The METRIC term the anchor is blind to -- the anchor's CE gives zero pull to a vertex q dislikes
# no matter how far away it sits; L_dist is signed by distance. See the module docstring.
#
# Usage: ./run_isobbox_dist_QUEUE.sh <gamma> [<gamma> ...]
#   env: ANCHOR_W (3.5e-7), ANCHOR_T (0.01), CYC_W (1.05e-5), CYC_T (0.05), DIST_T (1), MU (1), REL_W (1)
# gamma values come from run_isobbox_anchor_CALIBRATE.sh with DIST_T set:
#   at ladder step 25, anchor lambda=3.5e-7 -> gamma 1.40e-5 (1/10 of anchor), 2.80e-6 (1/50).
# Matched control = the SAME anchor+cycle arm without L_dist. Small analysis, comparisons vs that
# control and vs ZS.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh

[ $# -ge 1 ] || { echo "usage: $0 <gamma> [<gamma> ...]"; exit 1; }
ANCHOR_W=${ANCHOR_W:-3.5e-7}
ANCHOR_T=${ANCHOR_T:-0.01}
CYC_W=${CYC_W:-1.05e-5}
CYC_T=${CYC_T:-0.05}
DIST_T=${DIST_T:-1}
MU=${MU:-1}
REL_W=${REL_W:-1}
RELTAG=""; [ "$REL_W" = 1 ] || RELTAG=_rkdw${REL_W}
export SMALL=${SMALL-1}
MUTAG=""; [ "$MU" = 1 ] || MUTAG=_mu${MU}

# Matched control: anchor + cycle, no L_dist. Same naming rules as run_isobbox_cycv2_QUEUE.sh
# (demo dirs dropped _fuse16/_truediag2d on 2026-09-10; encoder dirs and eval tags keep them).
# ANCHOR_HARD=1 (2026-09-13): SHIC L_pseudo (one-hot argmax) instead of the soft q_ZS target; dir tag
# anchorhardT. No hard anchor+cycle arm without L_dist exists, so the hard queue needs CTRL_BASE set
# to an existing analyzed arm (encoder-dir suffix without encoder_isobbox_/_phase2), CTRL_LABEL, and
# optionally CTRL_OUT (comparison folder name, default comparison_vs_anchor_cycle).
AKIND=soft; HARDFLAG=""; [ "${ANCHOR_HARD:-0}" = 1 ] && { AKIND=hard; HARDFLAG=--anchor_hard; }
# CYC_STRIDE (2026-09-14): same flag and same default 5 as run_isobbox_cycv2_QUEUE.sh -- this queue ignored it until
# now, so every L_dist arm before today trained with the full 212549-pixel return leg (15 min instead of 4).
CYC_STRIDE=${CYC_STRIDE:-5}
STRIDE_FLAG=""; STRIDE_TAG=""
[ "$CYC_STRIDE" != 1 ] && { STRIDE_FLAG="--cyc_return_stride $CYC_STRIDE"; STRIDE_TAG="_retstride${CYC_STRIDE}"; }
BASE=sdsquare_fuse16_truediag2d${MUTAG}_anchor${AKIND}T${ANCHOR_T}_w${ANCHOR_W}_cycv2T${CYC_T}_c${CYC_W}${STRIDE_TAG}${RELTAG}_T1_seed0
CTRL=${CTRL_BASE:-$BASE}
# DIST_QT (2026-09-13): soft L_dist target, q_ZS at this T (dir tag _softqT<T>). Unset = hard argmax.
DTARGET="HARD target"; [ -n "$DIST_QT" ] && DTARGET="SOFT target q_T=$DIST_QT"
[ "$AKIND" = hard ] && [ -z "$CTRL_BASE" ] && { echo "ANCHOR_HARD=1 needs CTRL_BASE"; exit 1; }
CTRL_LABEL=${CTRL_LABEL:-"anchor lambda=$ANCHOR_W + cycle c=$CYC_W${STRIDE_TAG:+, return grid stride $CYC_STRIDE}, NO L_dist"}
# demo dirs drop _fuse16/_truediag2d (2026-09-10) and `allpix` (2026-09-14); encoder dirs and eval tags keep them
demo_run() { local s=${1/_fuse16_truediag2d/}; echo "bsb_isobbox_${s/allpix/}_phase2"; }
CTRL_RUN=$(demo_run "$CTRL")
CTRL_TAG=isobbox_${CTRL}
[ -d "$RUNS/$CTRL_RUN/analyze" ] || { echo "matched control not analyzed: $RUNS/$CTRL_RUN"; exit 1; }
ZS_RUN=bsb_isobbox_ZS_sdsquare
ZS_TAG=isobbox_ZS_sdsquare_fuse16

for G in "$@"; do
  SUFFIX=${BASE%_T1_seed0}_distT${DIST_T}${DIST_QT:+_softqT${DIST_QT}}_g${G}_T1_seed0
  DIR=./experiments/guitar/encoder_isobbox_${SUFFIX}_phase2
  RUN=$(demo_run "$SUFFIX")
  TAG=isobbox_${SUFFIX}
  LABEL="$AKIND anchor T_anchor=$ANCHOR_T lambda=$ANCHOR_W + cycle T_cyc=$CYC_T c=$CYC_W${STRIDE_TAG:+, return grid stride $CYC_STRIDE} + L_dist T_dist=$DIST_T $DTARGET gamma=$G${RELTAG:+, w_spatial=$REL_W}"
  echo "===================== L_dist T_dist=$DIST_T gamma=$G on anchor lambda=$ANCHOR_W + cycle c=$CYC_W mu=$MU w_spatial=$REL_W ====================="

  if [ -f "$DIR/pred_f.pth" ]; then echo "already trained: $DIR"; else
    time python phase2/train_relational_loss_l2norm.py \
      --obj_path "$OBJ" --name "$NAME" \
      --init_encoder_dir "$PHASE1" \
      --object_mask_path "$MASK" --fine_seg_path "$FINE" \
      --pix_dino_features_path "$PIX_sdsquare_fuse16" \
      --width 2304 --n_classes 2304 --geodesic --mu_3d_scale "$MU" --temperature 1.0 --train_seed 0 \
      --geo_diag_2d 1175.75 --rkd_spatial_dist_weight "$REL_W" \
      --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
      --anchor_weight "$ANCHOR_W" --anchor_temperature "$ANCHOR_T" $HARDFLAG \
      --cyc_weight "$CYC_W" --cyc_v2 --cyc_temperature "$CYC_T" --cyc_return_all_pixels $STRIDE_FLAG \
      --dist_weight "$G" --dist_temperature "$DIST_T" ${DIST_QT:+--dist_soft_target_temperature $DIST_QT} \
      --out_dir "$DIR"
  fi

  ./scripts/analyze_isobbox_arm.sh "$RUN" "$DIR/pred_f.pth" "$PIX_sdsquare_fuse16" "$TAG"

  # PF_A/PF_B (2026-09-14, Tal): pred_f of both sides, so the comparison can build the ONE-ROW similarity
  # strip (4 panels side by side) instead of the 2x2 grid -- panels then line up across arms.
  PF_A="$DIR/pred_f.pth" PF_B="./experiments/guitar/encoder_isobbox_${CTRL}_phase2/pred_f.pth" \
  ./scripts/compare_isobbox_arms.sh \
    "$RUN" "$LABEL" \
    "$CTRL_RUN" "$CTRL_LABEL" \
    ${CTRL_OUT:-comparison_vs_anchor_cycle} "$TAG" "$CTRL_TAG"

  PF_A="$DIR/pred_f.pth" PF_B="$PHASE1/pred_f.pth" \
  ./scripts/compare_isobbox_arms.sh \
    "$RUN" "$LABEL" \
    "$ZS_RUN" "ZS (phase-1 only)" \
    comparison_vs_ZS "$TAG" "$ZS_TAG"
done
echo "===== L_dist QUEUE DONE: $* (T_dist=$DIST_T, on anchor $ANCHOR_W + cycle $CYC_W) ====="
