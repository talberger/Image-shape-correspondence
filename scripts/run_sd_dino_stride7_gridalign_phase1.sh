#!/bin/bash
# Phase-1 for the stride-7 + patch-grid-alignment recipe, re-run 2026-08-28 with the two speed
# changes: SD UNet early exit (bit-identical output) and a float16 target cache (18 GB instead of
# 36 GB). Same recipe as the promoted `stride7_fixA` run in every other respect -- this is the
# end-to-end validation that the speed changes did not change the result.
# NOTE: SD's noise is UNSEEDED (modules/dino_sd_fusion.py), so this will NOT reproduce the promoted
# run bit-for-bit even in principle; the run-to-run spread it shows is itself the measurement.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb

CACHE=dino_sd_stride7_gridalign
DATA=./data/guitar/encoder_data_frontcentered_dinosd

echo "########## [$CACHE] precompute 1000 views (stride 7 + gridalign + early exit + fp16 cache) ##########"
time python train_encoder_dino_sd.py \
  --encoder_data_dir $DATA \
  --encoder_model_dir ./experiments/guitar/encoder_dino_sd_stride7_gridalign_dfd_crossview_phase1 \
  --view_sampling random --frontview_center 0 1.5707963267948966 --frontview_std 4 \
  --n_views 1000 --width 2304 --n_classes 2304 \
  --feature_cache_name $CACHE \
  --dino_patch_stride 7 --fix_grid_alignment \
  --cache_dtype float16 \
  --generate_random_views 0 --encode_random_views 1 --start_training 0

echo "########## [$CACHE] phase-1 (DFD cross-view) ##########"
time python train_encoder_dfd_crossview.py \
  --encoder_data_dir $DATA \
  --encoder_model_dir ./experiments/guitar/encoder_dino_sd_stride7_gridalign_dfd_crossview_phase1 \
  --feature_model_type $CACHE \
  --n_views 1000 --width 2304 --n_classes 2304
