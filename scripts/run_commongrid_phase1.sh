#!/bin/bash
# SHARED phase-1 for both common-grid arms (B and C), 2026-08-30.
# stride 14 (no gridalign -- at stride 14 the un-fixed upsample is already correct), but with
# --fuse_mode common_grid: SD is no longer downsampled onto DINO's 16x16 grid, it goes from its own
# 32x32 straight to the 64x64 phase-1 target.
# Renders are SQUARE, so square_pad and aspect_nopad are identical here -- which is why arms B and C
# can share this one phase-1 and only differ in how the PHOTO is framed.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
CACHE=dino_sd_commongrid
DATA=./data/guitar/encoder_data_frontcentered_dinosd
DIR=./experiments/guitar/encoder_dino_sd_commongrid_dfd_crossview_phase1

echo "########## [$CACHE] precompute 1000 views (stride 14, common-grid fuse, fp16 cache) ##########"
time python train_encoder_dino_sd.py \
  --encoder_data_dir $DATA --encoder_model_dir $DIR \
  --view_sampling random --frontview_center 0 1.5707963267948966 --frontview_std 4 \
  --n_views 1000 --width 2304 --n_classes 2304 \
  --feature_cache_name $CACHE \
  --fuse_mode common_grid \
  --cache_dtype float16 \
  --generate_random_views 0 --encode_random_views 1 --start_training 0

echo "########## [$CACHE] phase-1 (DFD cross-view) ##########"
time python train_encoder_dfd_crossview.py \
  --encoder_data_dir $DATA --encoder_model_dir $DIR \
  --feature_model_type $CACHE \
  --n_views 1000 --width 2304 --n_classes 2304
