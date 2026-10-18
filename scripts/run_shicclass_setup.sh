#!/usr/bin/env bash
# Per-CLASS setup for a new SHIC class (horse / elephant), 2026-09-22 -- everything that does not depend on the photo.
# Mirrors the bear: run_shicbear_phase1_renders.sh + run_shicbear_phase1_train.sh (raw SHIC .obj, frontview default),
# DensePose UV checker16 colouring + reference renders, and SHIC's own LVIS eval (reproduces Table 1 + caches the
# heat geodesics the per-instance eval needs). The 3D geodesic table is built by the first per-instance run.
# Usage: CLASS=horse bash scripts/run_shicclass_setup.sh
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
CLASS=${CLASS:?}
case "$CLASS" in horse) MESH=horse_5004 ;; elephant) MESH=elephant_5002 ;; *) echo "unknown CLASS"; exit 1 ;; esac
OBJ=./demo/shic/shapes/$CLASS.obj; NAME=shic_$CLASS
DATA=./data/$NAME/encoder_data; P1=./experiments/$NAME/encoder_dfd_crossview_phase1
CM=demo/shic/colored_mesh

echo "########## SHIC's own LVIS eval ($CLASS) ##########"
[ -f data/densepose_lvis/eval_shic_$CLASS/results.json ] || \
  python analysis/eval_densepose_lvis.py --class_name $CLASS --out_dir data/densepose_lvis/eval_shic_$CLASS

echo "########## checker16 UV colouring + reference renders ##########"
python analysis/fetch_densepose_uv.py --mesh_name $MESH --save_dir demo/shic/densepose_cse
python analysis/mesh_coloring.py --obj_path $OBJ --mode checker --checker_coords uv \
  --uv_path demo/shic/densepose_cse/texcoords_$MESH.pkl --grid 16 --save_dir $CM \
  --output_path $CM/${CLASS}_mesh_checker16_uv_color.ply
for az in 180 90 135; do
  python analysis/render_colored_mesh.py --obj_path $OBJ --ply_path $CM/${CLASS}_mesh_checker16_uv_color.ply \
    --save_dir $CM --output_path $CM/${CLASS}_checker16_uv_azim$az.png --render_res 400 --elev_deg 10 --azim_deg $az --save_rendered_views_grid
done

echo "########## phase-1: renders + DINO+SD precompute, 1000 views ##########"
if [ ! -f "$P1/encoder_checkpoint.pth" ]; then
  python train_encoder_dino_sd.py --obj_path $OBJ --name $NAME --encoder_data_dir $DATA --encoder_model_dir $P1 \
    --view_sampling random --frontview_std 4 --n_views 1000 --width 2304 --n_classes 2304 \
    --feature_cache_name dino_sd --generate_random_views 1 --encode_random_views 1 --start_training 0
  python train_encoder_dfd.py --obj_path $OBJ --name $NAME --encoder_data_dir $DATA --encoder_model_dir $P1 \
    --n_views 1000 --width 2304 --n_classes 2304 --generate_xyz_targets 1 --start_training 0
  echo "########## phase-1 training ##########"
  python train_encoder_dfd_crossview.py --obj_path $OBJ --name $NAME --encoder_data_dir $DATA --encoder_model_dir $P1 \
    --feature_model_type dino_sd --n_views 1000 --width 2304 --n_classes 2304
fi
echo "SHIC-CLASS SETUP DONE: $CLASS"
