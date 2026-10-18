#!/usr/bin/env bash
# SHIC-comparison bear: phase-1 renders + DINO+SD precompute, 1000 views (2026-09-03).
# Mirrors the guitar's current-recipe phase-1 build (scripts/run_isobbox_baseline_phase1_phase2.sh)
# on a NEW object. Two deliberate departures from that script, both explained in
# PROJECT_NOTES.md's SHIC-comparison section:
#   - --obj_path is the RAW demo/shic/shapes/bear.obj, NOT bear_shic_normalized.obj. Tal's call:
#     no isotropic remesh happened here (unlike the guitar), so the bbox-vs-centroid frame issue
#     that motivated a .normalizer.json sidecar does not apply -- plain loadmesh (centroid,
#     bounding-sphere) is used, same as this project's own original default. Vertex order is
#     unchanged from SHIC's own file (verified: faces identical, coords an exact affine transform),
#     so the DensePose UV (texcoords_bear_4936.pkl) and SHIC's own inference stay valid, indexed by
#     the SAME vertex ids -- only the numeric frame differs, which no correspondence depends on.
#   - --frontview_center is left at the script's own default (0, 0), NOT the guitar's tuned
#     (0, pi/2). That value encodes which way the GUITAR happens to face after its own frontview
#     fix; nothing equivalent has been established for the bear yet. Using the default is an
#     honest first pass, not a guess dressed as a decision -- revisit if the trained renders look
#     oddly weighted toward one angle.
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

OBJ=./demo/shic/shapes/bear.obj
NAME=shic_bear
DATA=./data/shic_bear/encoder_data
P1=./experiments/shic_bear/encoder_dfd_crossview_phase1

echo "########## 1/2 renders + DINO+SD precompute, 1000 views ##########"
time python train_encoder_dino_sd.py \
  --obj_path $OBJ --name $NAME \
  --encoder_data_dir $DATA \
  --encoder_model_dir $P1 \
  --view_sampling random --frontview_std 4 \
  --n_views 1000 --width 2304 --n_classes 2304 \
  --feature_cache_name dino_sd \
  --generate_random_views 1 --encode_random_views 1 --start_training 0

echo "########## 2/2 barycentric xyz targets ##########"
time python train_encoder_dfd.py \
  --obj_path $OBJ --name $NAME \
  --encoder_data_dir $DATA \
  --encoder_model_dir $P1 \
  --n_views 1000 --width 2304 --n_classes 2304 \
  --generate_xyz_targets 1 --start_training 0

echo "SHIC-BEAR PHASE-1 PRECOMPUTE DONE"
