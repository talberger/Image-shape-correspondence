#!/bin/bash
# The baseline recipe with the TRUE 2D geodesic diameter as the d_2D normalizer.
# 2026-09-08, Tal. The stored geo_diag_2d is a max over the 2048 sampled CENTER rows only, so it
# underestimates; the 3D side is a true all-pairs max. Measured with
# analysis/measure_2d_geodesic_diameter.py: 1175.75 px true vs 1171.65 stored, 0.35% low.
# Only --geo_diag_2d differs from scripts/run_isobbox_sdsquare_T1_phase2_train.sh.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

P2DIR=./experiments/guitar/encoder_isobbox_sdsquare_fuse16_truediag2d_T1_seed0_phase2
RUN=bsb_isobbox_sdsquare_T1_seed0_phase2

echo "########## [1/3] phase-2, T=1 seed0, geo_diag_2d = 1175.75 ##########"
if [ -f "$P2DIR/pred_f.pth" ]; then echo "already trained"; else
  time python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" \
    --init_encoder_dir "$PHASE1" \
    --object_mask_path "$MASK" \
    --fine_seg_path "$FINE" \
    --pix_dino_features_path "$PIX_sdsquare_fuse16" \
    --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 1.0 --train_seed 0 \
    --geo_diag_2d 1175.75 \
    --geo_table_3d_path "$GEO3D" \
    --geo_table_2d_path "$GEO2D" \
    --out_dir "$P2DIR"
fi

echo "########## [2/3] analysis + eval ##########"
./scripts/analyze_isobbox_arm.sh "$RUN" "$P2DIR/pred_f.pth" "$PIX_sdsquare_fuse16" \
  isobbox_sdsquare_fuse16_truediag2d_T1_seed0

echo "########## [3/3] compare vs the baseline ##########"
./scripts/compare_isobbox_arms.sh \
  "$RUN" "true geo_diag_2d = 1175.75" \
  bsb_isobbox_sdsquare_olddiag2d_T1_seed0_phase2 "BASELINE, stored geo_diag_2d = 1171.65" \
  comparison_vs_baseline_T1 \
  isobbox_sdsquare_fuse16_truediag2d_T1_seed0 isobbox_sdsquare_fuse16_T1_seed0

echo "===== TRUE-DIAG-2D CHAIN DONE ====="
