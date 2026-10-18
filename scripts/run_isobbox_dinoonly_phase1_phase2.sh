#!/usr/bin/env bash
# DINO-ONLY arm of the ISOBBOX baseline (Tal, 2026-09-02): same recipe, SD removed.
#
# Renders and barycentric xyz targets are REUSED from the SD+DINO isobbox data dir -- they depend
# only on the mesh, not on which 2D model is run over them (the 2026-09-01 dino_only arm did the
# same against the original mesh). Only the feature cache differs.
#
# THE ONE UNAVOIDABLE DIFFERENCE, same as 2026-09-01: width/n_classes 1024 instead of 2304, forced
# by the feature dimension. A 2304-wide encoder cannot consume 1024-channel features, so capacity
# rides along with the ablation. Flagged, not solved.
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

OBJ=./meshes/guitar_isotropic_bboxnorm.obj
NAME=guitar_isobbox
CACHE=dino_only_isobbox
DATA=./data/guitar/encoder_data_frontcentered_dinosd_isobbox
P1=./experiments/guitar/encoder_isobbox_dino_only_stride14_dfd_crossview_phase1
P2=./experiments/guitar/encoder_isobbox_dino_only_stride14_relational_geodesic_mu3dscale1_T0.01_seed0_phase2
FEAT=demo/guitar/features_analyze/dino_only_stride14

echo "########## 1/3 precompute 1000 views, DINO ONLY (no SD), reusing this mesh's renders ##########"
time python train_encoder_dino_sd.py \
  --obj_path $OBJ --name $NAME \
  --encoder_data_dir $DATA --encoder_model_dir $P1 \
  --view_sampling random --frontview_center 0 1.5707963267948966 --frontview_std 4 \
  --n_views 1000 --width 1024 --n_classes 1024 \
  --feature_cache_name $CACHE --dino_only \
  --generate_random_views 0 --encode_random_views 1 --start_training 0

echo "########## 2/3 phase-1 (DFD cross-view), DINO ONLY ##########"
time python train_encoder_dfd_crossview.py \
  --obj_path $OBJ --name $NAME \
  --encoder_data_dir $DATA --encoder_model_dir $P1 \
  --feature_model_type $CACHE \
  --n_views 1000 --width 1024 --n_classes 1024

echo "########## 3/3 phase-2, T=0.01, seed 0 ##########"
time python phase2/train_relational_loss_l2norm.py \
  --obj_path $OBJ --name $NAME \
  --init_encoder_dir $P1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path $FEAT/guitar_01_pix_dino_only_stride14.pth \
  --width 1024 --n_classes 1024 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
  --geo_table_3d_path ./experiments/guitar/geodesic_tables/guitar_isobbox_geo_table_3d.pth \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir $P2

echo "ISOBBOX DINO-ONLY DONE"
