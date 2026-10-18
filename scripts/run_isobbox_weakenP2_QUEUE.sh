#!/bin/bash
# WEAKENING PHASE-2 -- three jobs, 2026-09-09, Tal's call.
#
# Premise (Tal): ZS's feature space looks better, phase-2 is visually necessary but has
# side-effects that may come from training too strongly. --rkd_spatial_dist_weight is a NO-OP here
# (single loss term + Adam eps=1e-12 => scale-invariant), so the real knobs are elsewhere.
#
#  [1] steps 25 / 50 / 300 of the EXISTING step-ladder, analyzed visually for the first time.
#      The ladder measured the NUMBER at every step and never looked at the PICTURE.
#      NOTE: the ladder ran with geo_diag_2d=None, i.e. it is a sibling of the PRE-truediag
#      baseline, and on a different GPU (mean per-vertex cos 0.9126 to it). Shape, not identity.
#  [2] --huber_delta 0.1. Measured with --dump_residuals at step 1: residuals are mean 0.057 /
#      rms 0.071 / max 0.361, so the default delta=1.0 clips 0.00% -- Huber never leaves its
#      quadratic branch and the loss is plain MSE. 0.1 is exactly p90 (clips 16.7%); 0.2 would
#      clip 0.34% and measure nothing. Targeted: changes WHICH pairs push, not the step size.
#  [3] --resample_centers. The 2048 centers are currently fixed for all 300 steps (neighbours
#      already resample every step). Tried 2026-08-18 and came out flat -- but that was plain
#      DINO, old mesh, different recipe. Retested under the overfit framing.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

BASE_RUN=bsb_isobbox_sdsquare_T1_seed0_phase2
BASE_TAG=isobbox_sdsquare_fuse16_truediag2d_T1_seed0
LADDER=./experiments/guitar/encoder_isobbox_sdsquare_fuse16_T1_seed0_phase2_stepladder

train_arm () {   # $1=out_dir  $2=extra args
  if [ -f "$1/pred_f.pth" ]; then echo "already trained: $1"; return; fi
  time python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" \
    --init_encoder_dir "$PHASE1" \
    --object_mask_path "$MASK" --fine_seg_path "$FINE" \
    --pix_dino_features_path "$PIX_sdsquare_fuse16" \
    --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 1.0 --train_seed 0 \
    --geo_diag_2d 1175.75 \
    --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
    $2 --out_dir "$1"
}

train_arm_pool () {   # $1=out_dir  $2=extra args  $3=2D table (candidate pool)
  if [ -f "$1/pred_f.pth" ]; then echo "already trained: $1"; return; fi
  time python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" \
    --init_encoder_dir "$PHASE1" \
    --object_mask_path "$MASK" --fine_seg_path "$FINE" \
    --pix_dino_features_path "$PIX_sdsquare_fuse16" \
    --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 1.0 --train_seed 0 \
    --geo_diag_2d 1175.75 \
    --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$3" \
    $2 --out_dir "$1"
}

########################################################################
echo "##### [1/3] STEP-LADDER VISUALS: steps 25, 50, 300 #####"
for s in 25 50 300; do
  echo "--- ladder step $s"
  ./scripts/analyze_isobbox_arm.sh "bsb_isobbox_stepladder_step${s}" \
    "$LADDER/stepladder/pred_f_step${s}.pth" "$PIX_sdsquare_fuse16" \
    "isobbox_stepladder_step${s}"
done

echo "--- 4-panel back-projection: ZS -> 25 -> 50 -> 300"
OUT4=$RUNS/$BASE_RUN/comparison_stepladder_visual
mkdir -p "$OUT4"
BP=analyze/back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1_side_by_side.png
BPK1=analyze/back_projection/${MB}_backproj_${CHK_STEM}_k1_side_by_side.png
for pair in "soft:$BP" "top1k:$BPK1"; do
  KIND=${pair%%:*}; REL=${pair#*:}
  python analysis/stack_images_nway.py \
    --panel "$RUNS/bsb_isobbox_ZS_sdsquare/$REL" "step 0 = ZS (phase-1 only)  AUC 87.9/86.2/85.7" \
    --panel "$RUNS/bsb_isobbox_stepladder_step25/$REL" "step 25   AUC 76.8/76.3/75.2  drift 0.92" \
    --panel "$RUNS/bsb_isobbox_stepladder_step50/$REL" "step 50   AUC 78.1/77.2/76.1  drift 0.89" \
    --panel "$RUNS/bsb_isobbox_stepladder_step300/$REL" "step 300 = full phase-2  AUC 76.3/75.6/74.7  drift 0.43" \
    --title "phase-2 step ladder, back-projection $KIND (read T=1) -- ladder run, geo_diag_2d=None" \
    --output_path "$OUT4/stepladder_backproj_${KIND}.png"
done

########################################################################
echo "##### [2/3] --huber_delta 0.1 #####"
HDIR=./experiments/guitar/encoder_isobbox_sdsquare_fuse16_truediag2d_huber0.1_T1_seed0_phase2
HRUN=bsb_isobbox_sdsquare_huber0.1_T1_seed0_phase2
train_arm "$HDIR" "--huber_delta 0.1"
./scripts/analyze_isobbox_arm.sh "$HRUN" "$HDIR/pred_f.pth" "$PIX_sdsquare_fuse16" \
  isobbox_sdsquare_fuse16_truediag2d_huber0.1_T1_seed0
./scripts/compare_isobbox_arms.sh \
  "$HRUN" "huber_delta 0.1 (clips p90, 16.7% of pairs)" \
  "$BASE_RUN" "BASELINE, huber_delta 1.0 (clips 0% = plain MSE)" \
  comparison_vs_baseline \
  isobbox_sdsquare_fuse16_truediag2d_huber0.1_T1_seed0 "$BASE_TAG"

########################################################################
echo "##### [3/3] --resample_centers #####"
# The candidate-pool 2D table this needs was deleted 2026-08-18 (disk pressure) and must be
# rebuilt. stride 5, not the code default 3: stride 3 would be a ~23.6k-row pool and a ~20 GB
# table; stride 5 gives ~8.5k rows / ~7 GB, and it is the only pool configuration this project
# has ever validated (the 2026-08-18 --resample_centers test used it). --center_grid_stride 5
# below MUST match --grid_stride here or the run's stored-centers check refuses the table.
POOL=./experiments/guitar/geodesic_tables_pool_stride5
POOLTBL=$POOL/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth
if [ -f "$POOLTBL" ]; then echo "pool table already built: $POOLTBL"; else
  echo "--- building the candidate-pool 2D geodesic table (grid stride 5, ~11 min, ~7 GB)"
  mkdir -p "$POOL"
  time python analysis/build_geodesic_tables.py --obj_path "$OBJ" --name "$NAME" \
    --object_mask_path "$MASK" --fine_seg_path "$FINE" \
    --center_mode grid --grid_stride 5 --skip_3d --out_dir "$POOL"
  ls -la "$POOL"
fi
POOLTBL=$(ls "$POOL"/*geo_table_2d*.pth | head -1)
echo "pool table: $POOLTBL"

RDIR=./experiments/guitar/encoder_isobbox_sdsquare_fuse16_truediag2d_resamplecenters_T1_seed0_phase2
RRUN=bsb_isobbox_sdsquare_resamplecenters_T1_seed0_phase2
train_arm_pool "$RDIR" "--resample_centers --center_grid_stride 5" "$POOLTBL"
./scripts/analyze_isobbox_arm.sh "$RRUN" "$RDIR/pred_f.pth" "$PIX_sdsquare_fuse16" \
  isobbox_sdsquare_fuse16_truediag2d_resamplecenters_T1_seed0
./scripts/compare_isobbox_arms.sh \
  "$RRUN" "resample_centers (fresh 2048 centers every step)" \
  "$BASE_RUN" "BASELINE, 2048 centers fixed for all 300 steps" \
  comparison_vs_baseline \
  isobbox_sdsquare_fuse16_truediag2d_resamplecenters_T1_seed0 "$BASE_TAG"

echo "===== WEAKEN-P2 QUEUE DONE ====="
