#!/bin/bash
# PHASE-2 for the phase-marginalized PHOTO features (--dino_phases 4), isobbox mesh, 2026-09-07.
# Identical to the isobbox baseline recipe in every respect EXCEPT --pix_dino_features_path.
# Phase-1 is REUSED unchanged: the change is photo-side only, so the mesh side needs nothing.
# Both temperatures: T=1 (Tal's call on this mesh) and T=0.01 (the eval leader).
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

OBJ=./meshes/guitar_isotropic_bboxnorm.obj
P1=./experiments/guitar/encoder_isobbox_dino_sd_dfd_crossview_phase1
PIX=./demo/guitar/features_analyze/dino_sd_stride14_nosdsquare_phase4/guitar_01_pix_stride14_nosdsquare_phase4.pth
MASK=./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
FINE=./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth
GEO3D=./experiments/guitar/geodesic_tables/guitar_isobbox_geo_table_3d.pth
GEO2D=./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth

for T in 1 0.01; do
  OUT=./experiments/guitar/encoder_isobbox_dino_sd_stride14_nosdsquare_dinophases4_relational_geodesic_mu3dscale1_T${T}_seed0_phase2
  echo "########## phase-2, dino_phases=4, T=$T ##########"
  time python phase2/train_relational_loss_l2norm.py \
    --obj_path $OBJ --name guitar_isobbox \
    --init_encoder_dir $P1 \
    --object_mask_path $MASK \
    --fine_seg_path $FINE \
    --pix_dino_features_path $PIX \
    --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature $T --train_seed 0 \
    --geo_table_3d_path $GEO3D \
    --geo_table_2d_path $GEO2D \
    --out_dir $OUT
done
echo "DINOPHASES4 PHASE-2 QUEUE DONE"
