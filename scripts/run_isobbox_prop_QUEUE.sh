#!/bin/bash
# Arms ON TOP OF THE MAIN BASELINE (Tal, 2026-09-15): L_spatial, 16 in-segment neighbours, learned
# per-segment mu (`mu_learn_perseg`) + whatever extra trainer flags an arm adds. Output goes to
# demo/guitar/runs/isobbox_prop/ (the `_mu_` token routes it, lib_isobbox_common.sh run_dir).
#
# Usage: ./run_isobbox_prop_QUEUE.sh "<name_tag>|<extra trainer flags>|<label>" [...]
#   e.g. ./run_isobbox_prop_QUEUE.sh "sangle1e-2|--spatial_angle_weight 1e-2|+ spatial angle (Euclidean) w=1e-2"
#   name_tag is appended after mu_learn_perseg: ..._mu_learn_perseg_<name_tag>_T1_seed0_phase2
#   env: VS_SECONDARY=1 also compares against the SECONDARY baseline (anchor+cycle+L_dist) -- use it
#        for any arm that adds a feature-domain term. SMALL analysis by default (SMALL= for full).
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh

[ $# -ge 1 ] || { echo "usage: $0 \"<name_tag>|<extra flags>|<label>\" [...]"; exit 1; }
export SMALL=${SMALL-1}
demo_run() { local s=${1/_fuse16_truediag2d/}; echo "bsb_isobbox_${s}_phase2"; }

MAIN=sdsquare_fuse16_truediag2d_mu_learn_perseg_T1_seed0
MAIN_RUN=$(demo_run "$MAIN"); MAIN_TAG=isobbox_${MAIN}
SEC=sdsquare_fuse16_truediag2d_anchorsoftT0.01_w3.5e-7_cycv2T0.05_c1.05e-5_retstride5_distT1_g2.7951e-06_T1_seed0
SEC_RUN=$(demo_run "$SEC"); SEC_TAG=isobbox_${SEC}
ZS_RUN=bsb_isobbox_ZS_sdsquare; ZS_TAG=isobbox_ZS_sdsquare_fuse16
for r in "$MAIN_RUN" "$ZS_RUN"; do [ -d "$(run_dir "$r")/analyze" ] || { echo "reference not analyzed: $r"; exit 1; }; done

for SPEC in "$@"; do
  NTAG=${SPEC%%|*}; REST=${SPEC#*|}; EXTRA=${REST%%|*}; LABEL=${REST#*|}
  SUFFIX=sdsquare_fuse16_truediag2d_mu_learn_perseg_${NTAG}_T1_seed0
  DIR=./experiments/guitar/encoder_isobbox_${SUFFIX}_phase2
  RUN=$(demo_run "$SUFFIX"); TAG=isobbox_${SUFFIX}
  echo "===================== PROP ARM $NTAG  [$EXTRA] ====================="

  if [ -f "$DIR/pred_f.pth" ]; then echo "already trained: $DIR"; else
    # $EXTRA is deliberately unquoted: it is a list of flags
    time python phase2/train_relational_loss_l2norm.py \
      --obj_path "$OBJ" --name "$NAME" \
      --init_encoder_dir "$PHASE1" \
      --object_mask_path "$MASK" --fine_seg_path "$FINE" \
      --pix_dino_features_path "$PIX_sdsquare_fuse16" \
      --width 2304 --n_classes 2304 --geodesic --temperature 1.0 --train_seed 0 \
      --geo_diag_2d 1175.75 \
      --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 \
      --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
      --mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs \
      $EXTRA \
      --out_dir "$DIR"
  fi

  ./scripts/analyze_isobbox_arm.sh "$RUN" "$DIR/pred_f.pth" "$PIX_sdsquare_fuse16" "$TAG"

  PF_A="$DIR/pred_f.pth" PF_B="./experiments/guitar/encoder_isobbox_${MAIN}_phase2/pred_f.pth" \
  ./scripts/compare_isobbox_arms.sh "$RUN" "$LABEL" "$MAIN_RUN" "MAIN baseline (mu_learn_perseg)" \
    comparison_vs_main "$TAG" "$MAIN_TAG"
  # No per-arm comparison vs ZS any more (Tal, 2026-09-15): the ZS side is identical for every arm and
  # MAIN vs ZS already exists. ZS stays in the eval figures of the sweep comparisons.

  if [ -n "$VS_SECONDARY" ]; then
    PF_A="$DIR/pred_f.pth" PF_B="./experiments/guitar/encoder_isobbox_${SEC}_phase2/pred_f.pth" \
    ./scripts/compare_isobbox_arms.sh "$RUN" "$LABEL" "$SEC_RUN" "SECONDARY baseline (anchor+cycle+L_dist)" \
      comparison_vs_secondary "$TAG" "$SEC_TAG"
  fi
done
echo "===== PROP QUEUE DONE ====="
