#!/usr/bin/env bash
# SHIC-comparison bear: phase-2, relational loss + geodesic (2026-09-03).
# Same recipe shape as the guitar's current baseline (--geodesic --mu_3d_scale 0.5 --train_seed 0),
# pointed at the bear. TEMPERATURE IS A FIRST-PASS CHOICE, T=1 -- unlike the guitar, nothing has
# been established for this mesh (today's whole T sweep on the guitar was mesh-specific and gave
# three different winners depending on the criterion). T=1 is the plain, unbiased default; revisit
# once there's something to compare it against.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb

OBJ=./demo/shic/shapes/bear.obj
NAME=shic_bear
P1=./experiments/shic_bear/encoder_dfd_crossview_phase1
P2=./experiments/shic_bear/encoder_relational_geodesic_mu3dscale0.5_T1_seed0_phase2
MASK=./demo/shic/masks/bear3_click_350_300_sam_mask_1.pth
FINE=./demo/shic/bear3/bear3_fine_segments.pth
PIX=./demo/shic/features_analyze/stride14_squash/bear3_pix_stride14_squash.pth
GEO3D=./experiments/shic_bear/geodesic_tables/shic_bear_geo_table_3d.pth
GEO2D=./experiments/shic_bear/geodesic_tables/bear3_click_350_300_sam_mask_1_geo_table_2d_seed0.pth

echo "########## phase-2: relational loss, geodesic, T=1, seed 0 ##########"
time python phase2/train_relational_loss_l2norm.py \
  --obj_path $OBJ --name $NAME \
  --init_encoder_dir $P1 \
  --object_mask_path $MASK \
  --fine_seg_path $FINE \
  --pix_dino_features_path $PIX \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 0.5 --temperature 1.0 --train_seed 0 \
  --geo_table_3d_path $GEO3D \
  --geo_table_2d_path $GEO2D \
  --out_dir $P2

echo "SHIC-BEAR PHASE-2 DONE"
