#!/bin/bash
# SOFT feature-RKD, distance term ONLY and NO spatial term (Tal, 2026-08-31):
# --rkd_spatial_dist_weight 0, so the feature distance loss is the entire objective.
# beta is then just an overall scale, which Adam cancels -- kept at 1.0 for readability.
set -e
cd /workspace/bsb/scripts
L=/workspace/logs
EL=demo/guitar/eval_labels
PIX=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth
TAG=featonly_dist1.0_ang0
OUT=./experiments/guitar/encoder_dino_sd_stride14_nosdsquare_featrkd_soft_${TAG}_T0.01_seed0_phase2

echo "===== [train] feature-distance ONLY (relational_weight 0) ====="
( source /workspace/activate_bsb.sh; cd /workspace/bsb
  python phase2/train_relational_loss_l2norm.py \
    --init_encoder_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1 \
    --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
    --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
    --pix_dino_features_path ./$PIX \
    --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
    --rkd_spatial_dist_weight 0 \
    --feat_student_mode soft --rkd_feat_dist_weight 1.0 --rkd_feat_angle3_weight 0.0 \
    --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
    --out_dir $OUT ) > $L/featrkd_soft_${TAG}.log 2>&1
grep -E "^step (25|300)/300" $L/featrkd_soft_${TAG}.log | sed 's/(mu_2D[^)]*)//' | cut -c1-120
grep "final drift" $L/featrkd_soft_${TAG}.log

echo "===== [eval] ====="
( source /workspace/activate_bsb.sh; cd /workspace/bsb
  python analysis/eval_geodesic_error.py \
    --run_label "featrkd SOFT, feature-distance ONLY" \
    --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth --features_path $PIX \
    --vertex_features_path ${OUT#./}/pred_f.pth --k_vertices_per_pixel 10 \
    --save_path $EL/eval_featrkd_soft_${TAG}.pth \
    --plot_path $EL/eval_featrkd_soft_${TAG}.png ) > $L/featrkd_soft_${TAG}_eval.log 2>&1
echo "===== FEATONLY DONE ====="
