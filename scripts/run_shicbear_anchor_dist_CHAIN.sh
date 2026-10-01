#!/bin/bash
# SHIC-bear, 2026-09-17 (Tal): (0) SMALL analysis of the MAIN bear arm at step 25;
# then the recipe WITH anchor + L_dist on the bear, SMALL analysis, two arms:
#   secondary : the SECONDARY baseline EXACTLY (run_config of the guitar arm): soft anchor T=0.01 w=3.5e-7,
#               cycle v2 T_cyc=0.05 c=1.05e-5 stride 5, L_dist T=1 g=2.7951e-6, mu FIXED 1, 8in+8out, T=1
#   mainplus  : MAIN (learned per-seg mu, 16 in-seg, cycle c=1.676e-5, T=0.01) + the same anchor + L_dist
# All weights are the guitar's, NOT recalibrated for the bear. Same bear assets as run_shicbear_main_T.sh.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb

OBJ=./demo/shic/shapes/bear.obj
NAME=shic_bear
P1=./experiments/shic_bear/encoder_dfd_crossview_phase1
MASK=./demo/shic/masks/bear3_click_350_300_sam_mask_1.pth
FINE=./demo/shic/bear3/bear3_fine_segments.pth
GEO3D=./experiments/shic_bear/geodesic_tables/shic_bear_geo_table_3d.pth
GEO2D=./experiments/shic_bear/geodesic_tables/bear3_click_350_300_sam_mask_1_geo_table_2d_seed0.pth
PIX=demo/shic/features_analyze/sdsquare_fuse16/bear3_pix_sdsquare_fuse16.pth
DIAG=967.21

echo "==================== 0: MAIN bear arm, step 25, SMALL analysis ===================="
S=sdsquare_truediag2d_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0
MP2=experiments/shic_bear/encoder_${S}_phase2
STEP25=experiments/shic_bear/encoder_${S}_phase2_step25
mkdir -p $STEP25
python analysis/export_pred_f_l2norm.py --obj_path $OBJ --name $NAME \
  --checkpoints $MP2/encoder_checkpoint_step25.pth --outputs $STEP25/pred_f.pth --width 2304 --n_classes 2304
P2=$STEP25 RUN=demo/shic/bear3_${S}_phase2/step25_small PIX=$PIX READ_T=0.01 LABEL="MAIN recipe, step 25 of 300" \
  bash scripts/analyze_shicbear_arm.sh

ANCHOR="--anchor_weight 3.5e-07 --anchor_temperature 0.01 --dist_weight 2.7951e-06 --dist_temperature 1.0"
for ARM in secondary mainplus; do
  if [ $ARM = secondary ]; then
    T=1
    S=sdsquare_truediag2d_anchorsoftT0.01_w3.5e-7_cycv2T0.05_c1.05e-5_retstride5_distT1_g2.7951e-06_mu1_nbr8in8out_T1_seed0
    FLAGS="--mu_3d_scale 1.0 --num_neighbors_in_segment 8 --num_neighbors_out_segment 8 --cyc_weight 1.05e-05"
  else
    T=0.01
    S=sdsquare_truediag2d_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_anchorsoftT0.01_w3.5e-7_distT1_g2.7951e-06_T0.01_seed0
    FLAGS="--mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 --cyc_weight 1.676e-05"
  fi
  P2=./experiments/shic_bear/encoder_${S}_phase2
  RUN=demo/shic/bear3_${S}_phase2
  echo "==================== $ARM: phase-2, T=$T ===================="
  if [ -f "$P2/pred_f.pth" ]; then echo "already trained: $P2"; else
    time python phase2/train_relational_loss_l2norm.py \
      --obj_path "$OBJ" --name "$NAME" --init_encoder_dir "$P1" \
      --object_mask_path "$MASK" --fine_seg_path "$FINE" --pix_dino_features_path "$PIX" \
      --width 2304 --n_classes 2304 --geodesic --temperature $T --train_seed 0 --geo_diag_2d $DIAG \
      --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
      $FLAGS $ANCHOR --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5 \
      --out_dir "$P2"
  fi
  P2=$P2 RUN=$RUN PIX=$PIX READ_T=$T LABEL="$ARM (anchor + L_dist), T=$T" bash scripts/analyze_shicbear_arm.sh
  python analysis/topmatch_geo_extent.py --geo_table_3d_path $GEO3D \
    --knn_paths $RUN/analyze/ours/bear3_click_350_300_sam_mask_1_pixel_to_vertex_nn.pth --labels "$ARM" \
    | tee $RUN/analyze/proportion_extent.txt
done
echo "CHAIN DONE"
