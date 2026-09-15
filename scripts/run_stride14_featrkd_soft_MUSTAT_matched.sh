#!/bin/bash
# beta-MATCHED companions to the mu-stat runs (Tal, 2026-08-31).
# max/p95 make the normalized distances smaller, so beta=0.5 there is a WEAKER feature term than
# beta=0.5 with mean (measured loss ratio at step 25: mean 1.69, max 1.01). These arms raise beta so
# the feature/spatial ratio matches the mean run's, isolating the NORMALIZER from its strength.
set -e
cd /workspace/bsb/scripts
L=/workspace/logs
EL=demo/guitar/eval_labels
PIX=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth

run () {  # $1 = stat, $2 = beta
  local STAT=$1 BETA=$2
  local TAG=dist${BETA}_ang0_mu${STAT}
  local OUT=./experiments/guitar/encoder_dino_sd_stride14_nosdsquare_featrkd_soft_${TAG}_T0.01_seed0_phase2
  echo "===== [train] mu=$STAT beta=$BETA ====="
  ( source /workspace/activate_bsb.sh; cd /workspace/bsb
    python phase2/train_relational_loss_l2norm.py \
      --init_encoder_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1 \
      --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
      --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
      --pix_dino_features_path ./$PIX \
      --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
      --feat_student_mode soft --rkd_feat_dist_weight $BETA --rkd_feat_angle3_weight 0.0 \
      --feat_mu_stat $STAT \
      --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
      --out_dir $OUT ) > $L/featrkd_soft_${TAG}.log 2>&1
  grep "mu references" $L/featrkd_soft_${TAG}.log; grep "final drift" $L/featrkd_soft_${TAG}.log
  echo "===== [eval] mu=$STAT beta=$BETA ====="
  ( source /workspace/activate_bsb.sh; cd /workspace/bsb
    python analysis/eval_geodesic_error.py \
      --run_label "featrkd SOFT beta=$BETA, mu=$STAT" \
      --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth --features_path $PIX \
      --vertex_features_path ${OUT#./}/pred_f.pth --k_vertices_per_pixel 10 \
      --save_path $EL/eval_featrkd_soft_${TAG}.pth \
      --plot_path $EL/eval_featrkd_soft_${TAG}.png ) > $L/featrkd_soft_${TAG}_eval.log 2>&1
}

run max 0.85
run p95 0.8
echo "===== MATCHED MU-STAT DONE ====="
