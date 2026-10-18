#!/bin/bash
# LORA-W on the bear (2026-09-17, Tal's decisions): MAIN recipe WITHOUT anchor (rkd_spatial_dist + learned per-seg mu
# + 16 in-seg + cycle v2, T=0.01), but the phase-1 encoder is FROZEN and only a rank-R update on the output Linear
# trains (modules/lora.py). Then SMALL analysis at alpha=1 and at read-time alphas (analysis/export_pred_f_lora_alpha.py).
# Env: RANK (16), LAYERS (last|all), LR (1e-3), ALPHAS ("0.25 0.5 0.75").
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

RANK=${RANK:-16}; LAYERS=${LAYERS:-last}; LR=${LR:-1e-3}; ALPHAS=${ALPHAS:-"0.25 0.5 0.75"}
OBJ=./demo/shic/shapes/bear.obj; NAME=shic_bear
P1=./experiments/shic_bear/encoder_dfd_crossview_phase1
MASK=./demo/shic/masks/bear3_click_350_300_sam_mask_1.pth
GEO3D=./experiments/shic_bear/geodesic_tables/shic_bear_geo_table_3d.pth
PIX=demo/shic/features_analyze/sdsquare_fuse16/bear3_pix_sdsquare_fuse16.pth
# SUFFIX_EXTRA (2026-09-20): append a tag so a re-run lands in its OWN arm dir instead of hitting the
# "pred_f.pth already exists -> skip training" guard below. Used to re-run an existing arm as a
# regression check without disturbing the original.
SUFFIX_EXTRA=${SUFFIX_EXTRA:-}
S=sdsquare_truediag2d_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_loraW_${LAYERS}_r${RANK}_lr${LR}_T0.01_seed0${SUFFIX_EXTRA}
P2=./experiments/shic_bear/encoder_${S}_phase2
RUN=demo/shic/bear3_${S}_phase2

if [ ! -f "$P2/pred_f.pth" ]; then
  python phase2/train_relational_loss_l2norm.py --obj_path "$OBJ" --name "$NAME" --init_encoder_dir "$P1" \
    --object_mask_path "$MASK" --fine_seg_path ./demo/shic/bear3/bear3_fine_segments.pth --pix_dino_features_path "$PIX" \
    --width 2304 --n_classes 2304 --geodesic --temperature 0.01 --train_seed 0 --geo_diag_2d 967.21 \
    --geo_table_3d_path "$GEO3D" --geo_table_2d_path ./experiments/shic_bear/geodesic_tables/bear3_click_350_300_sam_mask_1_geo_table_2d_seed0.pth \
    --mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs \
    --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 \
    --cyc_weight 1.676e-05 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5 \
    --lora_w_rank $RANK --lora_w_layers $LAYERS --lora_w_lr $LR --out_dir "$P2"
fi

P2=$P2 RUN=$RUN PIX=$PIX READ_T=0.01 LABEL="LORA-W $LAYERS r=$RANK, alpha 1" bash scripts/analyze_shicbear_arm.sh

python analysis/export_pred_f_lora_alpha.py --obj_path "$OBJ" --name "$NAME" \
  --phase1_checkpoint "$P1/encoder_checkpoint.pth" --lora_checkpoint "$P2/encoder_checkpoint.pth" \
  --alphas $ALPHAS --out_dir "$P2/alpha"
KNNS="$RUN/analyze/ours/bear3_click_350_300_sam_mask_1_pixel_to_vertex_nn.pth"; LABELS="alpha1"
for a in $ALPHAS; do
  P2=$P2/alpha/alpha$a RUN=$RUN/alpha$a PIX=$PIX READ_T=0.01 LABEL="LORA-W $LAYERS r=$RANK, alpha $a" \
    bash scripts/analyze_shicbear_arm.sh
  KNNS="$KNNS $RUN/alpha$a/analyze/ours/bear3_click_350_300_sam_mask_1_pixel_to_vertex_nn.pth"; LABELS="$LABELS alpha$a"
done
python analysis/topmatch_geo_extent.py --geo_table_3d_path $GEO3D --knn_paths $KNNS --labels $LABELS | tee $RUN/proportion_extent.txt
echo "LORA-W DONE: $RUN"
