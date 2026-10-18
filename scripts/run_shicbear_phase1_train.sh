#!/usr/bin/env bash
# SHIC-comparison bear: phase-1 training, DFD barycentric distillation with cross-view batching
# (2026-09-03). Same recipe as the guitar's current baseline, script 2/4 of
# scripts/run_isobbox_baseline_phase1_phase2.sh, pointed at the bear instead.
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

OBJ=./demo/shic/shapes/bear.obj
NAME=shic_bear
DATA=./data/shic_bear/encoder_data
P1=./experiments/shic_bear/encoder_dfd_crossview_phase1

echo "########## phase-1: DFD barycentric distillation, cross-view batching ##########"
time python train_encoder_dfd_crossview.py \
  --obj_path $OBJ --name $NAME \
  --encoder_data_dir $DATA \
  --encoder_model_dir $P1 \
  --feature_model_type dino_sd \
  --n_views 1000 --width 2304 --n_classes 2304

echo "SHIC-BEAR PHASE-1 TRAINING DONE"
