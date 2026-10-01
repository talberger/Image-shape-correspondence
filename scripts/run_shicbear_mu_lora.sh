#!/bin/bash
# The 2026-09-21 mu experiments (directional mu, grid, global, merged segments) re-run WITH LORA-W (Tal).
# Same recipe as run_shicbear_main_T.sh (MAIN: rkd_spatial_dist + 16 in-seg + cycle v2, T=0.01) but the phase-1
# encoder is FROZEN and only LORA-W trains, exactly the production arm: all layers, r=4, lr 1e-3, read at alpha 0.5.
# SMALL analysis at alpha 0.5 only.
# Env: MU = learn | aniso | grid | globalaniso | anisoxglobal ; MU_LR (1e-2) ; MU_GRID (6) ; FINE (fine segments) ; SUFFIX (tag)
# IMG_TAG (2026-09-22): default bear3 (unchanged). IMG_TAG=lvis<ann_id> runs on a DensePose-LVIS instance whose inputs were
# kept in data/densepose_lvis/runs/<tag> (KEEP_INTERMEDIATES=1): its mask / fine segments / 2D geo table / features / true
# diameter, cyc_return_stride 3 (the LVIS recipe, 400-px images; RET_STRIDE overrides), and the LVIS analysis
# (scripts/make_lvis_arm_summary.sh: soft back-projection + GT markers vs SHIC) instead of the bear3 one.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb

MU=${MU:-learn}; MU_LR=${MU_LR:-1e-2}; MU_GRID=${MU_GRID:-6}; SUFFIX=${SUFFIX:-}; ALPHA=0.5
# MU_FIXED (2026-09-28): a mu_aniso.pth of HAND-MADE ellipses, frozen (globalthenaniso: at the switch; anisoxglobal: all rows from step 0, file needs the global row)
# GCIRCLE=0.7 (2026-09-29): globalthenaniso, stage-A global frozen to that circle, stage B learned
# FIX_SEG="2:2.5" (2026-09-29): globalthenaniso, at the switch fix seg 2 to a circle 2.5, the rest learned
# STAGEA_FIXED=1 (2026-09-29): with SWITCH_HYBRID + MU_FIXED, the manual file is frozen from step 0 (stage A = manual global row)
# SWITCH_HYBRID=1 (2026-09-28): two-stage whose stage B is the HYBRID (8 in + 8 out); with MU_FIXED = a file WITH the global row, all frozen
NIN=${NIN:-16}; NOUT=${NOUT:-0}   # neighbours in / out of the centre's fine segment (NOUT>0: globalaniso / anisoxglobal only)
IMG_TAG=${IMG_TAG:-bear3}
case "$MU" in
  learn)       MUFLAGS="--mu_mode learned --mu_scope segment --mu_lr $MU_LR" ;;
  aniso)       MUFLAGS="--mu_mode learned --mu_scope segment_aniso --mu_lr $MU_LR" ;;
  grid)        MUFLAGS="--mu_mode learned --mu_scope grid_aniso --mu_grid $MU_GRID --mu_lr $MU_LR" ;;
  globalaniso) MUFLAGS="--mu_mode learned --mu_scope global_aniso --mu_lr $MU_LR" ;;
  globalthenaniso) MUFLAGS="--mu_mode learned --mu_scope global_then_segment_aniso --mu_switch_step ${SWITCH:-150} --mu_lr $MU_LR${MU_FIXED:+ --mu_aniso_fixed_path $MU_FIXED}${SWITCH_HYBRID:+ --switch_hybrid --switch_nin 8 --switch_nout 8}${STAGEA_FIXED:+ --stageA_fixed}${FIX_SEG:+ --mu_fix_seg_circle $FIX_SEG}${GCIRCLE:+ --stageA_fixed_circle $GCIRCLE}" ;;   # stage A global in+out, stage B per-seg 16 in
  anisoxglobal) MUFLAGS="--mu_mode learned --mu_scope segment_aniso_xglobal --mu_lr $MU_LR${MU_FIXED:+ --mu_aniso_fixed_path $MU_FIXED}" ;;   # per-seg ellipse in-seg, global ellipse cross-seg
  *) echo "unknown MU=$MU"; exit 1 ;;
esac
# CLASS (2026-09-22): bear (default) | horse | elephant; non-bear classes need IMG_TAG=lvis<id> (prep: scripts/run_lvis_prep.sh)
CLASS=${CLASS:-bear}
OBJ=./demo/shic/shapes/$CLASS.obj; NAME=shic_$CLASS
P1=./experiments/shic_$CLASS/encoder_dfd_crossview_phase1
GEO3D=./experiments/shic_$CLASS/geodesic_tables/shic_${CLASS}_geo_table_3d.pth
if [ "$IMG_TAG" = bear3 ]; then
  FINE=${FINE:-./demo/shic/bear3/bear3_fine_segments.pth}
  MASK=./demo/shic/masks/bear3_click_350_300_sam_mask_1.pth
  GEO2D=./experiments/shic_bear/geodesic_tables/bear3_click_350_300_sam_mask_1_geo_table_2d_seed0.pth
  PIX=demo/shic/features_analyze/sdsquare_fuse16/bear3_pix_sdsquare_fuse16.pth
  DIAG=967.21; RET_STRIDE=${RET_STRIDE:-5}
  S=sdsquare_truediag2d_mu_${MU}_perseg_cycv2T0.05_c1.676e-05_retstride${RET_STRIDE}_loraW_all_r4_lr1e-3_T0.01_seed0${SUFFIX}
  P2=./experiments/shic_bear/encoder_${S}_phase2
  RUN=demo/shic/bear3_${S}_phase2
else
  D=data/densepose_lvis/runs/$IMG_TAG
  FINE=${FINE:-$D/${IMG_TAG}_fine_segments.pth}
  MASK=$D/${IMG_TAG}_mask.pth
  GEO2D=$D/${IMG_TAG}_mask_geo_table_2d_seed0.pth
  PIX=$D/${IMG_TAG}_pix_sdsquare_fuse16.pth
  DIAG=$(python -c "import json; print(json.load(open('$D/${IMG_TAG}_mask_geo_table_2d_seed0_meta.json'))['true_diameter_2d'])")
  RET_STRIDE=${RET_STRIDE:-3}
  S=mu_${MU}_perseg_cycv2T0.05_c1.676e-05_retstride${RET_STRIDE}_loraW_all_r4_lr1e-3_T0.01_seed0${SUFFIX}
  P2=./experiments/shic_$CLASS/lvis/${IMG_TAG}/encoder_${S}_phase2
  RUN=demo/shic/${IMG_TAG}_mu/${S}
fi

if [ ! -f "$P2/pred_f.pth" ]; then
  python phase2/train_relational_loss_l2norm.py --obj_path "$OBJ" --name "$NAME" --init_encoder_dir "$P1" \
    --object_mask_path "$MASK" --fine_seg_path "$FINE" --pix_dino_features_path "$PIX" \
    --width 2304 --n_classes 2304 --geodesic --temperature 0.01 --train_seed 0 --geo_diag_2d $DIAG \
    --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
    $MUFLAGS --mu_fit_space abs \
    --num_neighbors_in_segment $NIN --num_neighbors_out_segment $NOUT \
    --cyc_weight 1.676e-05 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride $RET_STRIDE \
    --lora_w_rank 4 --lora_w_layers ${LORA_LAYERS:-all} --lora_w_lr 1e-3 --out_dir "$P2"
fi

python analysis/export_pred_f_lora_alpha.py --obj_path "$OBJ" --name "$NAME" \
  --phase1_checkpoint "$P1/encoder_checkpoint.pth" --lora_checkpoint "$P2/encoder_checkpoint.pth" \
  --alphas $ALPHA --out_dir "$P2/alpha"
if [ "$IMG_TAG" = bear3 ]; then
  P2=$P2/alpha/alpha$ALPHA RUN=$RUN/alpha$ALPHA PIX=$PIX READ_T=0.01 LABEL="LORA-W all r4 a$ALPHA, mu=$MU$SUFFIX" \
    bash scripts/analyze_shicbear_arm.sh
else
  CLASS=$CLASS ANN_ID=${IMG_TAG#lvis} PRED_F=$P2/alpha/alpha$ALPHA/pred_f.pth BASE=$RUN/alpha$ALPHA bash scripts/make_lvis_arm_summary.sh
fi
echo "MU-LORA DONE: $RUN/alpha$ALPHA"
