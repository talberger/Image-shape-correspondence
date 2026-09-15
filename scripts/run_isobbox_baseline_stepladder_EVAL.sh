#!/bin/bash
# THE STEP-LADDER PROBE. Evaluate the baseline at every phase-2 step checkpoint, so the SHAPE of
# the 2.7% -> 4.7% degradation can be seen rather than guessed.
#
# 2026-09-08. The open question is whether phase-2's damage to a good input is a property of the
# relational loss or a bug. The two look different on this curve:
#   gradual decline over 300 steps -> the loss really does pull there
#   a jump inside the first 25 steps, then flat -> initialization or a bug
# Step 0 is ZS (phase-1), already evaluated as isobbox_ZS_sdsquare_fuse16.
#
# --l2norm IS NOT OPTIONAL (added to export_pred_f.py 2026-09-08): every current recipe trains an
# EncoderL2Norm, and Tanh/L2Normalize are both parameter-free, so building the wrong one loads
# cleanly and silently exports features that are simply wrong.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh

RUN=./experiments/guitar/encoder_isobbox_sdsquare_fuse16_T1_seed0_phase2_stepladder
PIX=$PIX_sdsquare_fuse16
OUTDIR=$RUN/stepladder
STEPS="25 50 75 100 125 150 175 200 225 250 275 300"
mkdir -p "$OUTDIR"

CKPTS=""; OUTS=""
for s in $STEPS; do
  CKPTS="$CKPTS $RUN/encoder_checkpoint_step${s}.pth"
  OUTS="$OUTS $OUTDIR/pred_f_step${s}.pth"
done

echo "=== [1/2] export pred_f for every step checkpoint (batch mode, one mesh load) ==="
python analysis/export_pred_f.py \
  --checkpoints "$(echo $CKPTS | tr ' ' ',')" \
  --outputs "$(echo $OUTS | tr ' ' ',')" \
  --l2norm --obj_path "$OBJ" --name "$NAME" \
  --width 2304 --n_classes 2304

echo "=== [2/2] eval each one against eval v2 ==="
for s in $STEPS; do
  echo "--- step $s"
  python analysis/eval_geodesic_error.py --obj_path "$OBJ" --name "$NAME" \
    --run_label "stepladder step $s" \
    --geo_table_path "$GEO3D" --eval_pairs_path "$EVALPAIRS" \
    --features_path "$PIX" --vertex_features_path "$OUTDIR/pred_f_step${s}.pth" \
    --k_vertices_per_pixel 10 \
    --save_path "$OUTDIR/eval_step${s}_v2.pth" \
    --plot_path "$OUTDIR/eval_step${s}_v2.png"
done

echo "===== STEP-LADDER EVAL DONE: /workspace/bsb/$OUTDIR ====="
