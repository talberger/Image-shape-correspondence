#!/usr/bin/env bash
# THE BASELINE RECIPE, REBUILT ON THE ISOTROPIC MESH (2026-09-02, Tal's request).
#
# The mesh changed (meshes/guitar.obj 4036v -> meshes/guitar_isotropic.obj 3886v, isotropic
# remesh: edge-length CV 1.497 -> 0.230, area max/median 47 -> 1.8), so every mesh-dependent
# artifact has to be rebuilt. Recipe is otherwise UNCHANGED from the current baseline
# (encoder_dino_sd_stride14_nosdsquare_..._T0.01_seed0_phase2): stride 14, SD+DINO, squash photo
# framing (NO sd_square -- the known bug Tal deliberately keeps in), phase-2 --geodesic
# --mu_3d_scale 1 --temperature 0.01 --train_seed 0.
#
# EVERY MESH-CONSUMING CALL PASSES --obj_path EXPLICITLY. The default is './meshes/guitar.obj'
# in ~30 scripts across this repo, so a missing flag would silently run the OLD mesh.
#
# REUSED UNCHANGED (all mesh-INDEPENDENT, verified):
#   - photo features  demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/...  (photo only)
#   - 2D geodesic table  ..._geo_table_2d_seed0.pth  (pixel space: mask + fine segments only)
#   - coarse mask, fine segments, position coloring, the 9 tracked pixels  (all photo-side)
#
# REBUILT FOR THIS MESH (before this script, already done):
#   - 3D geodesic table  guitar_iso_geo_table_3d.pth  at t_coef=1.0, RE-SWEPT (the old 20.0 was
#     tuned to the old mesh's 500x edge ratio; worst undershoot 0.061 -> 0.0104)
#   - eval pairs  demo/guitar/eval_labels/iso/pixel_vertex_eval_pairs.pth  (the same 100 hand
#     -picked .pp points re-snapped; needed a real fix in join_eval_pixel_labels.py, see notes)
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

echo "########## 1/4 renders + DINO+SD precompute, 1000 views, ISOTROPIC MESH ##########"
echo "(--generate_random_views 1: the renders themselves change with the mesh, unlike the"
echo " dino_only arm of 2026-09-01 which could reuse the baseline's 224/ and xyz_bary_f/)"
time python train_encoder_dino_sd.py \
  --obj_path $OBJ --name $NAME \
  --encoder_data_dir $DATA \
  --encoder_model_dir $P1 \
  --view_sampling random --frontview_center 0 1.5707963267948966 --frontview_std 4 \
  --n_views 1000 --width 2304 --n_classes 2304 \
  --feature_cache_name $CACHE \
  --generate_random_views 1 --encode_random_views 1 --start_training 0

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
