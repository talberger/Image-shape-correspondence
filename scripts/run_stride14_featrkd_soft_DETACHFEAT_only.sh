#!/bin/bash
# detach-FEATURES + feature-distance ONLY (Tal, 2026-08-31): --feat_detach_features together with
# --rkd_spatial_dist_weight 0, so the ONLY objective is featD and the ONLY gradient route is the soft-map
# weights. Companion to run_stride14_featrkd_soft_DETACHFEAT.sh, which is the same detach ON TOP of
# the existing spatial loss (that one collapsed: AUC 24.2/26.5/17.7, drift -0.335).
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs
EL=demo/guitar/eval_labels
PIX=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth
TAG=featonly_dist1.0_ang0_detachfeat
OUT=./experiments/guitar/encoder_dino_sd_stride14_nosdsquare_featrkd_soft_${TAG}_T0.01_seed0_phase2

echo "===== [train] detach FEATURES + featD ONLY ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python phase2/train_relational_loss_l2norm.py \
    --init_encoder_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1 \
    --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
    --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
    --pix_dino_features_path ./$PIX \
    --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
    --rkd_spatial_dist_weight 0 \
    --feat_student_mode soft --rkd_feat_dist_weight 1.0 --rkd_feat_angle3_weight 0.0 \
    --feat_detach_features \
    --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
    --out_dir $OUT ) > $L/featrkd_soft_${TAG}.log 2>&1
grep -E "^step (25|300)/300" $L/featrkd_soft_${TAG}.log | sed 's/(mu_2D[^)]*)//' | cut -c1-120
grep "final drift" $L/featrkd_soft_${TAG}.log

echo "===== [eval] ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python analysis/eval_geodesic_error.py \
    --run_label "featrkd SOFT, featD only, features DETACHED" \
    --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth --features_path $PIX \
    --vertex_features_path ${OUT#./}/pred_f.pth --k_vertices_per_pixel 10 \
    --save_path $EL/eval_featrkd_soft_${TAG}.pth \
    --plot_path $EL/eval_featrkd_soft_${TAG}.png ) > $L/featrkd_soft_${TAG}_eval.log 2>&1

echo "########## small_analyze: featrkd_soft_${TAG} ##########"
./small_analyze.sh "featrkd_soft_${TAG}" "featrkd_soft_${TAG}" \
  > $L/small_analyze_featrkd_soft_${TAG}.log 2>&1 || echo "FAILED"
grep -E "coverage \(read T=1\)" $L/small_analyze_featrkd_soft_${TAG}.log || true
echo "===== DETACHFEAT-ONLY DONE ====="
