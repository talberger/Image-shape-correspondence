#!/bin/bash
# THE MISSING CELL OF THE 2x2 (Tal, 2026-09-08): sd_frame=square_pad AND fuse at SD's native 32.
#
#                | fuse16 (SD downsampled onto DINO's grid) | fuse32 (SD at native res)
#   squash       | sdsquash_fuse16  (current baseline)      | impossible -- the code refuses it
#   square_pad   | sdsquare_fuse16  (was "arm A")           | sdsquare_fuse32  <-- THIS
#   aspect_nopad | --                                       | sdnopad_fuse32   (was "arm B")
#
# WHY A PHASE-1 REBUILD IS REQUIRED, and why the old "arm B" result is confounded:
# train_encoder_dino_sd.py:96-101 states it outright -- --fuse_mode common_grid changes the render
# cache (SD stops being downsampled onto DINO's 16x16 and is upsampled from its own 32x32 straight
# to the 64x64 phase-1 target), so "a cache built with a different --fuse_mode is NOT
# interchangeable with the production one". sdnopad_fuse32 was trained on the PRODUCTION dino_grid
# phase-1 while its photo features were cg32 -- photo@32 read against mesh@16. Its 75.4 / 76.0 is
# therefore not a clean read of the fuse resolution. This chain fixes that for both fuse32 arms.
#
# WHAT IS REUSED (all fuse-mode-INDEPENDENT, so the mesh side stays matched to every other arm):
#   - the 1000 renders            data/.../encoder_data_frontcentered_dinosd_isobbox/224/
#   - the barycentric xyz targets data/.../xyz_bary_f/
# Only the FEATURE CACHE is rebuilt (~36 GB, float32 to match the production cache bit-for-bit in
# dtype -- float16 would halve it but would confound the comparison with a precision change).
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

DATA=./data/guitar/encoder_data_frontcentered_dinosd_isobbox
CACHE=dino_sd_isobbox_cg32
P1=./experiments/guitar/encoder_isobbox_fuse32_phase1
PIXDIR=demo/guitar/features_analyze/dino_sd_sdsquare_cg32
PIX=$PIXDIR/guitar_01_pix_sdsquare_cg32.pth
P2=./experiments/guitar/encoder_isobbox_sdsquare_fuse32_T1_seed0_phase2

echo "########## 1/4 render-side feature cache at fuse32 (renders + xyz targets REUSED) ##########"
time python train_encoder_dino_sd.py \
  --obj_path $OBJ --name $NAME \
  --encoder_data_dir $DATA \
  --encoder_model_dir $P1 \
  --view_sampling random --frontview_center 0 1.5707963267948966 --frontview_std 4 \
  --n_views 1000 --width 2304 --n_classes 2304 \
  --feature_cache_name $CACHE \
  --fuse_mode common_grid \
  --generate_random_views 0 --encode_random_views 1 --start_training 0

echo "########## 2/4 phase-1: DFD barycentric distillation, cross-view batching ##########"
time python train_encoder_dfd_crossview.py \
  --obj_path $OBJ --name $NAME \
  --encoder_data_dir $DATA \
  --encoder_model_dir $P1 \
  --feature_model_type $CACHE \
  --n_views 1000 --width 2304 --n_classes 2304

echo "########## 3/4 photo features: square_pad + common_grid 32 ##########"
mkdir -p "$PIXDIR"
time python analysis/extract_dino_sd_features.py \
  --img_path images/guitar_01.png \
  --save_dir "$PIXDIR" \
  --output_path "$PIX" \
  --save_raw_path "$PIXDIR/guitar_01_pix_sdsquare_cg32_RAW_pregrid.pth" \
  --sd_frame square_pad --fuse_mode common_grid --common_grid 32 \
  --cpu_postprocess

echo "########## 4/4 phase-2: relational loss, geodesic, T=1, seed 0 ##########"
time python phase2/train_relational_loss_l2norm.py \
  --obj_path $OBJ --name $NAME \
  --init_encoder_dir $P1 \
  --object_mask_path $MASK \
  --fine_seg_path $FINE \
  --pix_dino_features_path $PIX \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 1.0 --train_seed 0 \
  --geo_table_3d_path $GEO3D \
  --geo_table_2d_path $GEO2D \
  --out_dir $P2

echo "===== sdsquare_fuse32 CHAIN DONE ====="
