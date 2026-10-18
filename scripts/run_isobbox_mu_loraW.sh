#!/bin/bash
# GUITAR (isobbox) on the CURRENT recipe (Tal, 2026-09-27): MAIN (run_isobbox_main_T.sh, T=0.01, cycle v2 stride 5)
# + LORA-W all r4 lr 1e-3 read at alpha 0.5 + learned DIRECTIONAL mu at mu_lr 5e-2 -- the same arm the horse/bear LVIS
# runs use (run_shicbear_mu_lora.sh), only the guitar's own inputs.
# Env: MU = learn | aniso | globalaniso | anisoxglobal | globalthenaniso (SWITCH step, default 150) (default: the hybrid) ; NIN/NOUT (8/8) ; MU_LR (5e-2) ; SUFFIX
# MU_FIXED (2026-09-28): hand-made frozen ellipses (analysis/make_manual_mu_aniso.py) -- see run_shicbear_mu_lora.sh
# SWITCH_HYBRID=1 (2026-09-28): two-stage with a HYBRID stage B (8/8); MU_FIXED must then include the global row (--add_global)
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

MU=${MU:-anisoxglobal}; MU_LR=${MU_LR:-5e-2}; NIN=${NIN:-8}; NOUT=${NOUT:-8}; SUFFIX=${SUFFIX:-}; ALPHA=0.5; T=0.01
case "$MU" in
  learn)        SCOPE=segment ;;          # ISOTROPIC per-seg = MAIN's mu, plus LORA-W
  aniso)        SCOPE=segment_aniso ;;
  globalaniso)  SCOPE=global_aniso ;;
  anisoxglobal) SCOPE=segment_aniso_xglobal ;;
  globalthenaniso) SCOPE="global_then_segment_aniso --mu_switch_step ${SWITCH:-150}"; SUFFIX="_sw${SWITCH:-150}${SUFFIX}" ;;   # stage A global in+out, stage B per-seg 16 in
  *) echo "unknown MU=$MU"; exit 1 ;;
esac
export SMALL=1 READ_T=$T
S=isobbox_sdsquare_fuse16_truediag2d_mu_${MU}_perseg_cycv2T0.05_c1.676e-05_retstride5_loraW_all_r4_lr1e-3_T${T}_seed0_mulr${MU_LR}_in${NIN}out${NOUT}${SUFFIX}
D=./experiments/guitar/encoder_${S}_phase2
R=bsb_isobbox_sdsquare_mu_${MU}_perseg_cycv2T0.05_c1.676e-05_retstride5_loraW_all_r4_T${T}_seed0_mulr${MU_LR}_in${NIN}out${NOUT}${SUFFIX}_a${ALPHA}_phase2

if [ -f "$D/pred_f.pth" ]; then echo "already trained: $D"; else
  time python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" --init_encoder_dir "$PHASE1" \
    --object_mask_path "$MASK" --fine_seg_path "$FINE" --pix_dino_features_path "$PIX_sdsquare_fuse16" \
    --width 2304 --n_classes 2304 --geodesic --temperature $T --train_seed 0 \
    --geo_diag_2d 1175.75 \
    --num_neighbors_in_segment $NIN --num_neighbors_out_segment $NOUT \
    --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
    --mu_mode learned --mu_scope $SCOPE --mu_lr $MU_LR --mu_fit_space abs ${MU_FIXED:+--mu_aniso_fixed_path $MU_FIXED} ${SWITCH_HYBRID:+--switch_hybrid --switch_nin 8 --switch_nout 8} \
    --cyc_weight 1.676e-05 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5 \
    --lora_w_rank 4 --lora_w_layers all --lora_w_lr 1e-3 \
    --out_dir "$D"
fi
python analysis/export_pred_f_lora_alpha.py --obj_path "$OBJ" --name "$NAME" \
  --phase1_checkpoint "$PHASE1/encoder_checkpoint.pth" --lora_checkpoint "$D/encoder_checkpoint.pth" \
  --alphas $ALPHA --out_dir "$D/alpha"
./scripts/analyze_isobbox_arm.sh "$R" "$D/alpha/alpha$ALPHA/pred_f.pth" "$PIX_sdsquare_fuse16" "$S"
DEMO=$(run_dir "$R")
python analysis/plot_mu_ellipses.py --run_dir "$D" --fine_seg_path "$FINE" --img_path images/guitar_01.png \
  --min_frac 0.01 --output_path "$DEMO/ellipses.png"
echo "===== ISOBBOX MU-LORA $MU DONE: $DEMO ====="
