#!/bin/bash
# SOFT feature-RKD with --feat_detach_features (Tal, 2026-08-31): the vertex FEATURES are detached
# inside the student sum, so the gradient reaches the encoder ONLY through the soft-map weights.
# This is structurally what the SPATIAL relational loss does (its Sum w*X uses fixed positions), and
# it is the mirror of the 2026-08-29 --feat_detach_weights test. beta=0.5, mu=mean -- the best soft
# config -- so the detach is the only change from run_stride14_featrkd_soft_beta_sweep's 0.5 arm.
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs
EL=demo/guitar/eval_labels
PIX=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth
TAG=dist0.5_ang0_detachfeat
OUT=./experiments/guitar/encoder_dino_sd_stride14_nosdsquare_featrkd_soft_${TAG}_T0.01_seed0_phase2

echo "===== [train] detach FEATURES, beta=0.5 ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python phase2/train_relational_loss_l2norm.py \
    --init_encoder_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1 \
    --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
    --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
    --pix_dino_features_path ./$PIX \
    --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
    --feat_student_mode soft --rkd_feat_dist_weight 0.5 --rkd_feat_angle3_weight 0.0 \
    --feat_detach_features \
    --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
    --out_dir $OUT ) > $L/featrkd_soft_${TAG}.log 2>&1
grep -E "^step (25|300)/300" $L/featrkd_soft_${TAG}.log | sed 's/(mu_2D[^)]*)//' | cut -c1-120
grep "final drift" $L/featrkd_soft_${TAG}.log

echo "===== [eval] ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python analysis/eval_geodesic_error.py \
    --run_label "featrkd SOFT beta=0.5, features DETACHED" \
    --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth --features_path $PIX \
    --vertex_features_path ${OUT#./}/pred_f.pth --k_vertices_per_pixel 10 \
    --save_path $EL/eval_featrkd_soft_${TAG}.pth \
    --plot_path $EL/eval_featrkd_soft_${TAG}.png ) > $L/featrkd_soft_${TAG}_eval.log 2>&1
echo "===== DETACHFEAT DONE ====="
