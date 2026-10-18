#!/bin/bash
# CYCLE LOSS v2 on top of the ZS anchor (2026-09-10). L = REL_W * L_rel + ANCHOR_W * L_anchor + c * L_cyc_v2.
# modules/cycle_consistency_loss_v2.py: SHIC L_cyc, all vertices, temperature on both legs (CSE).
#
# Usage: ./run_isobbox_cycv2_QUEUE.sh <cyc_weight> [<cyc_weight> ...]
#   env: ANCHOR_W (3.5e-7), ANCHOR_T (0.01), CYC_T (0.05), MU (1), REL_W (1)
# cyc_weight values come from run_isobbox_anchor_CALIBRATE.sh with CYC_T set (SHIC ratio: 1/50 of the anchor).
# REL_W (2026-09-11, Tal): weakens the spatial-RKD term. NOT a no-op here -- it scales only L_rel, so it changes
# the ratio to the anchor and cycle terms (Adam-equivalent to multiplying ANCHOR_W and c by 1/REL_W).
# Matched control = the SAME anchor arm without cycle. Small analysis, comparisons vs that control and vs ZS.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

[ $# -ge 1 ] || { echo "usage: $0 <cyc_weight> [<cyc_weight> ...]"; exit 1; }
ANCHOR_W=${ANCHOR_W:-3.5e-7}
ANCHOR_T=${ANCHOR_T:-0.01}
CYC_T=${CYC_T:-0.05}
# CYC_ALLPIX=1 (Tal, 2026-09-10): return leg over ALL mask pixels (SHIC's L_cyc as written), not the 2048 centers.
ALLPIX_FLAG=""; ALLPIX_TAG=""
# ALLPIX_TAG emptied 2026-09-14 (Tal): the all-pixel return is the only variant in use, and defaults are
# not named. The centers-return arm that used to collide with the short name is now tagged _retcenters,
# in its encoder dir, its eval tag and its demo dir alike.
[ -n "$CYC_ALLPIX" ] && { ALLPIX_FLAG="--cyc_return_all_pixels"; ALLPIX_TAG=""; }
# CYC_STRIDE=<s> (2026-09-13): all-pixel return leg on an s-pixel grid only (--cyc_return_stride). Dir tag _retstride<s>.
# DEFAULT 5 since 2026-09-14 (Tal): 3.5x faster training (4.3 min vs 15.4). It costs a little QUALITY -- Tal sees
# slightly less smoothness in feature space -- and is taken for the runtime alone. CYC_STRIDE=1 = the full return.
CYC_STRIDE=${CYC_STRIDE:-5}
STRIDE_TAG=""
[ -n "$CYC_ALLPIX" ] && [ "$CYC_STRIDE" != 1 ] && {
  ALLPIX_FLAG="$ALLPIX_FLAG --cyc_return_stride $CYC_STRIDE"; STRIDE_TAG="_retstride${CYC_STRIDE}"; }
MU=${MU:-1}
REL_W=${REL_W:-1}
RELTAG=""; [ "$REL_W" = 1 ] || RELTAG=_rkdw${REL_W}
export SMALL=${SMALL-1}
MUTAG=""; [ "$MU" = 1 ] || MUTAG=_mu${MU}

CTRL=sdsquare_fuse16_truediag2d${MUTAG}_anchorsoftT${ANCHOR_T}_w${ANCHOR_W}_T1_seed0
# demo dirs dropped _fuse16/_truediag2d on 2026-09-10 and `allpix` on 2026-09-14 (Tal: only non-defaults are named,
# and the centers-return arm that used to collide with it was deleted); encoder dirs and eval tags keep the long names
demo_run() { local s=${1/_fuse16_truediag2d/}; echo "bsb_isobbox_${s/allpix/}_phase2"; }
CTRL_RUN=$(demo_run "$CTRL")
CTRL_TAG=isobbox_${CTRL}
[ -d "$RUNS/$CTRL_RUN/analyze" ] || { echo "matched control not analyzed: $RUNS/$CTRL_RUN"; exit 1; }
ZS_RUN=bsb_isobbox_ZS_sdsquare
ZS_TAG=isobbox_ZS_sdsquare_fuse16

for C in "$@"; do
  SUFFIX=${CTRL%_T1_seed0}_cycv2${ALLPIX_TAG}T${CYC_T}_c${C}${STRIDE_TAG}${RELTAG}_T1_seed0
  DIR=./experiments/guitar/encoder_isobbox_${SUFFIX}_phase2
  RUN=$(demo_run "$SUFFIX")
  TAG=isobbox_${SUFFIX}
  LABEL="anchor T_anchor=$ANCHOR_T lambda=$ANCHOR_W + cycle T_cyc=$CYC_T c=$C${STRIDE_TAG:+, return grid stride $CYC_STRIDE}${RELTAG:+, w_spatial=$REL_W}"
  echo "===================== CYCLE v2 T_cyc=$CYC_T c=$C on anchor T_anchor=$ANCHOR_T lambda=$ANCHOR_W mu=$MU w_spatial=$REL_W ====================="

  if [ -f "$DIR/pred_f.pth" ]; then echo "already trained: $DIR"; else
    time python phase2/train_relational_loss_l2norm.py \
      --obj_path "$OBJ" --name "$NAME" \
      --init_encoder_dir "$PHASE1" \
      --object_mask_path "$MASK" --fine_seg_path "$FINE" \
      --pix_dino_features_path "$PIX_sdsquare_fuse16" \
      --width 2304 --n_classes 2304 --geodesic --mu_3d_scale "$MU" --temperature 1.0 --train_seed 0 \
      --geo_diag_2d 1175.75 --rkd_spatial_dist_weight "$REL_W" \
      --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
      --anchor_weight "$ANCHOR_W" --anchor_temperature "$ANCHOR_T" \
      --cyc_weight "$C" --cyc_v2 --cyc_temperature "$CYC_T" $ALLPIX_FLAG \
      --out_dir "$DIR"
  fi

  ./scripts/analyze_isobbox_arm.sh "$RUN" "$DIR/pred_f.pth" "$PIX_sdsquare_fuse16" "$TAG"

  # PF_A/PF_B (2026-09-14, Tal): pred_f of both sides -> one-row similarity strips, not the 2x2 grid.
  PF_A="$DIR/pred_f.pth" PF_B="./experiments/guitar/encoder_isobbox_${CTRL}_phase2/pred_f.pth" \
  ./scripts/compare_isobbox_arms.sh \
    "$RUN" "$LABEL" \
    "$CTRL_RUN" "anchor T=$ANCHOR_T lambda=$ANCHOR_W, NO cycle" \
    comparison_vs_anchor_only "$TAG" "$CTRL_TAG"

  PF_A="$DIR/pred_f.pth" PF_B="$PHASE1/pred_f.pth" \
  ./scripts/compare_isobbox_arms.sh \
    "$RUN" "$LABEL" \
    "$ZS_RUN" "ZS (phase-1 only)" \
    comparison_vs_ZS "$TAG" "$ZS_TAG"
done
echo "===== CYCLE v2 QUEUE DONE: $* (rkd_w=$REL_W) ====="
