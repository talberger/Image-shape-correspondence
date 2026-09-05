#!/usr/bin/env bash
# RESUME of run_iso_baseline_phase1_phase2.sh from stage 1b (2026-09-02).
# Stage 1 (renders + 36 GB DINO+SD cache) already completed; skipped here so it is not redone.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb

OBJ=./meshes/guitar_isotropic.obj
NAME=guitar_iso
CACHE=dino_sd_iso
DATA=./data/guitar/encoder_data_frontcentered_dinosd_iso
P1=./experiments/guitar/encoder_iso_dino_sd_dfd_crossview_phase1
P2=./experiments/guitar/encoder_iso_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T0.01_seed0_phase2
PIX=./demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth
MASK=./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
FINE=./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth
GEO3D=./experiments/guitar/geodesic_tables/guitar_iso_geo_table_3d.pth
GEO2D=./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth


echo "########## 1b/4 barycentric xyz targets (kaolin rasterizer, no external model) ##########"
# MISSING THE FIRST TIME THIS SCRIPT RAN (2026-09-02): train_encoder_dino_sd.py builds the renders
# and the dino_sd feature cache but NOT xyz_bary_f/, and every recent arm (the 2026-09-01 dino_only
# run included) reused the baseline data dir's existing xyz_bary_f/ rather than generating one. A
# new mesh needs its own. Cheap, and skips any target already on disk.
time python train_encoder_dfd.py \
  --obj_path $OBJ --name $NAME \
  --encoder_data_dir $DATA \
  --encoder_model_dir $P1 \
  --n_views 1000 --width 2304 --n_classes 2304 \
  --generate_xyz_targets 1 --start_training 0

echo "########## 2/4 phase-1: DFD barycentric distillation, cross-view batching ##########"
time python train_encoder_dfd_crossview.py \
  --obj_path $OBJ --name $NAME \
  --encoder_data_dir $DATA \
  --encoder_model_dir $P1 \
  --feature_model_type $CACHE \
  --n_views 1000 --width 2304 --n_classes 2304

echo "########## 3/4 phase-2: relational loss, geodesic, T=0.01, seed 0 ##########"
time python phase2/train_relational_loss_l2norm.py \
  --obj_path $OBJ --name $NAME \
  --init_encoder_dir $P1 \
  --object_mask_path $MASK \
  --fine_seg_path $FINE \
  --pix_dino_features_path $PIX \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
  --geo_table_3d_path $GEO3D \
  --geo_table_2d_path $GEO2D \
  --out_dir $P2

echo "ISO PHASE-1 + PHASE-2 DONE"
