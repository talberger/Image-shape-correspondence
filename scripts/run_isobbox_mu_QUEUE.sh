#!/bin/bash
# MU-LEARNED (2026-09-14, Tal) -- the proportion/zoom coefficient between the two normalized
# distance spaces becomes a FITTED quantity instead of a hand-swept constant.
#   L = L_spatial only. No anchor, no cycle, no L_dist -- deliberately, so the effect of the
#   scale is isolated. The matched control is therefore the SPATIAL-ONLY baseline
#   (bsb_isobbox_sdsquare_T1_seed0_phase2, 77.0 / 76.8 / 75.1, coverage 40.1%).
#
# Usage: ./run_isobbox_mu_QUEUE.sh <mode:scope> [<mode:scope> ...]
#   e.g. ./run_isobbox_mu_QUEUE.sh optimal:global learned:global optimal:segment learned:segment
#   env: NBR_IN/NBR_OUT (8/8) -- NBR_OUT=0 NBR_IN=16 keeps every pair INSIDE the centre's segment,
#        which is the only way a per-segment scale never has to serve a pair that spans two scales
#        (Tal, 2026-09-14). MU_LR (1e-2, learned only), MU_FIT (abs|log), TAG_SUFFIX (appended to the dir tag)
# Dir tag (renamed 2026-09-15, old->new table in PROJECT_NOTES.md): _mu_<closed|learn>_<global|perseg>, plus
# _nbrmix when NBR_OUT != 0 and _logfit when MU_FIT=log. e.g. NBR_IN=16 NBR_OUT=0 learned:segment -> _mu_learn_perseg.
# Output demo dirs land in demo/guitar/runs/isobbox_prop/.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh

[ $# -ge 1 ] || { echo "usage: $0 <mode:scope> [<mode:scope> ...]"; exit 1; }
export SMALL=${SMALL-1}
MU_LR=${MU_LR:-1e-2}
MU_FIT=${MU_FIT:-abs}
NBR_IN=${NBR_IN:-8}; NBR_OUT=${NBR_OUT:-8}

CTRL=sdsquare_fuse16_truediag2d_T1_seed0
demo_run() { local s=${1/_fuse16_truediag2d/}; echo "bsb_isobbox_${s/allpix/}_phase2"; }
CTRL_RUN=$(demo_run "$CTRL")
CTRL_TAG=isobbox_${CTRL}
[ -d "$RUNS/$CTRL_RUN/analyze" ] || { echo "matched control not analyzed: $RUNS/$CTRL_RUN"; exit 1; }
ZS_RUN=bsb_isobbox_ZS_sdsquare
ZS_TAG=isobbox_ZS_sdsquare_fuse16

for SPEC in "$@"; do
  MODE=${SPEC%%:*}; SCOPE=${SPEC##*:}
  case "$MODE" in optimal) MTAG=closed;; learned) MTAG=learn;; *) echo "bad mode $MODE"; exit 1;; esac
  case "$SCOPE" in global) STAG=global;; segment) STAG=perseg;; *) echo "bad scope $SCOPE"; exit 1;; esac
  # Names (Tal, 2026-09-15): mu_<closed|learn>_<global|perseg>, then only what differs from the
  # isobbox_prop default -- 16 neighbours all in-segment, absolute fit. `_mu_` routes the arm to
  # $PROP_RUNS (lib_isobbox_common.sh run_dir).
  NBRTAG=""; [ "$NBR_OUT" != 0 ] && NBRTAG=_nbrmix
  FITTAG=""; [ "$MU_FIT" = log ] && FITTAG=_logfit
  SUFFIX=sdsquare_fuse16_truediag2d_mu_${MTAG}_${STAG}${NBRTAG}${FITTAG}${TAG_SUFFIX}_T1_seed0
  DIR=./experiments/guitar/encoder_isobbox_${SUFFIX}_phase2
  RUN=$(demo_run "$SUFFIX")
  TAG=isobbox_${SUFFIX}
  # NOTE: no $( ... && echo ... ) in this assignment -- under `set -e` a command substitution that
  # exits non-zero takes the whole script down silently (cost 10 minutes on 2026-09-14).
  LRNOTE=""; [ "$MODE" = learned ] && LRNOTE=" (lr $MU_LR)"
  FITNOTE=""; [ "$MU_FIT" = log ] && FITNOTE=", relative fit"
  NBRNOTE=", ${NBR_IN}+${NBR_OUT} neighbours"; [ "$NBR_OUT" = 0 ] && NBRNOTE=", ${NBR_IN} neighbours ALL in-segment"
  LABEL="L_spatial only, mu_3d_scale ${MODE} ${SCOPE}${LRNOTE}${FITNOTE}${NBRNOTE}"
  echo "===================== MU $MODE $SCOPE ====================="

  if [ -f "$DIR/pred_f.pth" ]; then echo "already trained: $DIR"; else
    time python phase2/train_relational_loss_l2norm.py \
      --obj_path "$OBJ" --name "$NAME" \
      --init_encoder_dir "$PHASE1" \
      --object_mask_path "$MASK" --fine_seg_path "$FINE" \
      --pix_dino_features_path "$PIX_sdsquare_fuse16" \
      --width 2304 --n_classes 2304 --geodesic --temperature 1.0 --train_seed 0 \
      --geo_diag_2d 1175.75 \
      --num_neighbors_in_segment "$NBR_IN" --num_neighbors_out_segment "$NBR_OUT" \
      --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
      --mu_mode "$MODE" --mu_scope "$SCOPE" --mu_lr "$MU_LR" --mu_fit_space "$MU_FIT" \
      --out_dir "$DIR"
  fi

  ./scripts/analyze_isobbox_arm.sh "$RUN" "$DIR/pred_f.pth" "$PIX_sdsquare_fuse16" "$TAG"

  PF_A="$DIR/pred_f.pth" PF_B="./experiments/guitar/encoder_isobbox_${CTRL}_phase2/pred_f.pth" \
  ./scripts/compare_isobbox_arms.sh \
    "$RUN" "$LABEL" \
    "$CTRL_RUN" "L_spatial only, mu_3d_scale FIXED 1 (baseline)" \
    comparison_vs_mu1 "$TAG" "$CTRL_TAG"

  PF_A="$DIR/pred_f.pth" PF_B="$PHASE1/pred_f.pth" \
  ./scripts/compare_isobbox_arms.sh \
    "$RUN" "$LABEL" \
    "$ZS_RUN" "ZS (phase-1 only)" \
    comparison_vs_ZS "$TAG" "$ZS_TAG"
done
echo "===== MU QUEUE DONE: $* ====="
