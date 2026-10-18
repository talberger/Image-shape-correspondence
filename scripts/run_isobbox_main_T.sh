#!/bin/bash
# MAIN's exact recipe at a chosen temperature (Tal, 2026-09-16). MAIN is T=1; this exists to run the
# T=0.01 control, so that the neck-rings arm (which Tal keeps at T=0.01) and the baseline differ by
# the MESH ALONE. Every flag below is copied from MAIN's own run_config.json.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

T=${T:-0.01}
export SMALL=1 READ_T=$T
S=isobbox_sdsquare_fuse16_truediag2d_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T${T}_seed0
D=./experiments/guitar/encoder_${S}_phase2
R=bsb_isobbox_sdsquare_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T${T}_seed0_phase2

if [ -f "$D/pred_f.pth" ]; then echo "already trained: $D"; else
  time python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" --init_encoder_dir "$PHASE1" \
    --object_mask_path "$MASK" --fine_seg_path "$FINE" --pix_dino_features_path "$PIX_sdsquare_fuse16" \
    --width 2304 --n_classes 2304 --geodesic --temperature $T --train_seed 0 \
    --geo_diag_2d 1175.75 \
    --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 \
    --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
    --mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs \
    --cyc_weight 1.676e-05 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5 \
    --out_dir "$D"
fi
./scripts/analyze_isobbox_arm.sh "$R" "$D/pred_f.pth" "$PIX_sdsquare_fuse16" "$S"
echo "===== ISOBBOX MAIN T=$T DONE ====="
