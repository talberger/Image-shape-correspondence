#!/bin/bash
# PHASE MARGINALIZATION (--dino_phases 4) ON THE NEW BASELINE RECIPE.
# 2026-09-08, Tal: *"בוא נעשה ריצה של החלקה של הפיצ'רים עם 4 פאזות... אבל בוא נעשה את זה עם המתכון
# החדש ונשווה לעם ולבלי"*. The 2026-09-07 measurement was run on `squash`, which is no longer the
# framing; this repeats it on `sdsquare_fuse16` T=1, the single current baseline.
#
# PHOTO SIDE ONLY. Phase-1 is REUSED UNCHANGED and that is correct: the renders go through the same
# DINO path, but each vertex aggregates ~1000 random views with the object at a different grid phase
# in every one, i.e. the mesh side already marginalizes for free. Only `--pix_dino_features_path`
# differs from the baseline, everywhere.
#
# --dino_phases 4 keeps --sd_frame square_pad (the script's default, and the new recipe's framing);
# it REFUSES --fuse_mode common_grid / --fix_grid_alignment / --dino_patch_stride / --save_raw_path.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

FEATDIR=demo/guitar/features_analyze/dino_sd_sdsquare_phase4
PIX=$FEATDIR/guitar_01_pix_sdsquare_phase4.pth
P1=$PHASE1/pred_f.pth
P2DIR=./experiments/guitar/encoder_isobbox_sdsquare_fuse16_phase4_T1_seed0_phase2
ZS_RUN=bsb_isobbox_ZS_sdsquare_phase4
P2_RUN=bsb_isobbox_sdsquare_olddiag2d_phase4_T1_seed0_phase2

echo "########## [1/6] photo features, 4 DINO phases, sd_frame square_pad ##########"
if [ -f "$PIX" ]; then echo "already extracted: $PIX"; else
  mkdir -p "$FEATDIR"
  time python analysis/extract_dino_sd_features.py \
    --img_path images/guitar_01.png \
    --save_dir "$FEATDIR" \
    --output_path "$PIX" \
    --sd_frame square_pad \
    --dino_phases 4 \
    --cpu_postprocess
fi

echo "########## [2/6] ZS analysis (phase-1 reused unchanged) ##########"
./scripts/analyze_isobbox_arm.sh "$ZS_RUN" "$P1" "$PIX" isobbox_ZS_sdsquare_fuse16_phase4

echo "########## [3/6] phase-2, T=1 seed0 ##########"
if [ -f "$P2DIR/pred_f.pth" ]; then echo "already trained: $P2DIR"; else
  time python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" \
    --init_encoder_dir "$PHASE1" \
    --object_mask_path "$MASK" \
    --fine_seg_path "$FINE" \
    --pix_dino_features_path "$PIX" \
    --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 1.0 --train_seed 0 \
    --geo_table_3d_path "$GEO3D" \
    --geo_table_2d_path "$GEO2D" \
    --out_dir "$P2DIR"
fi

echo "########## [4/6] phase-2 analysis ##########"
./scripts/analyze_isobbox_arm.sh "$P2_RUN" "$P2DIR/pred_f.pth" "$PIX" isobbox_sdsquare_fuse16_phase4_T1_seed0

echo "########## [5/6] compare vs baseline AT ZS ##########"
./scripts/compare_isobbox_arms.sh \
  "$ZS_RUN" "ZS phase4 (4 DINO phases)" \
  bsb_isobbox_ZS_sdsquare "ZS BASELINE (1 phase)" \
  comparison_vs_ZS_baseline \
  isobbox_ZS_sdsquare_fuse16_phase4 isobbox_ZS_sdsquare_fuse16

echo "########## [6/6] compare vs baseline AT PHASE-2 ##########"
./scripts/compare_isobbox_arms.sh \
  "$P2_RUN" "phase-2 phase4 T=1 (4 DINO phases)" \
  bsb_isobbox_sdsquare_olddiag2d_T1_seed0_phase2 "phase-2 BASELINE T=1 (1 phase)" \
  comparison_vs_baseline_T1 \
  isobbox_sdsquare_fuse16_phase4_T1_seed0 isobbox_sdsquare_fuse16_T1_seed0

echo "===== PHASE4 CHAIN DONE ====="
