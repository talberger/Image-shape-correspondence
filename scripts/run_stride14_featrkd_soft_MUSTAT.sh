#!/bin/bash
# Alternative FIXED normalizers for the feature-RKD distance term (Tal, 2026-08-31):
# each space divided by its own MAX pair distance, and by its own 95th percentile, instead of the
# mean. beta held at 0.5 (the best soft arm) so the normalizer is the only change -- but note
# max/p95 >> mean, so the normalized distances and the loss SHRINK; the printed loss ratios tell us
# how much beta would need rescaling for a like-for-like comparison.
set -e
cd /workspace/bsb/scripts
L=/workspace/logs
EL=demo/guitar/eval_labels
PIX=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth

for STAT in max p95; do
  TAG=dist0.5_ang0_mu${STAT}
  OUT=./experiments/guitar/encoder_dino_sd_stride14_nosdsquare_featrkd_soft_${TAG}_T0.01_seed0_phase2
  echo "===== [train] mu stat = $STAT ====="
  ( source /workspace/activate_bsb.sh; cd /workspace/bsb
    python phase2/train_relational_loss_l2norm.py \
      --init_encoder_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1 \
      --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
      --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
      --pix_dino_features_path ./$PIX \
      --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
      --feat_student_mode soft --feat_dist_weight 0.5 --feat_angle_weight 0.0 \
      --feat_mu_stat $STAT \
      --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
      --out_dir $OUT ) > $L/featrkd_soft_${TAG}.log 2>&1
  grep "mu references" $L/featrkd_soft_${TAG}.log
  grep "final drift" $L/featrkd_soft_${TAG}.log

  echo "===== [eval] mu stat = $STAT ====="
  ( source /workspace/activate_bsb.sh; cd /workspace/bsb
    python analysis/eval_geodesic_error.py \
      --run_label "featrkd SOFT beta=0.5, mu=$STAT" \
      --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth --features_path $PIX \
      --vertex_features_path ${OUT#./}/pred_f.pth --k_vertices_per_pixel 10 \
      --save_path $EL/eval_featrkd_soft_${TAG}.pth \
      --plot_path $EL/eval_featrkd_soft_${TAG}.png ) > $L/featrkd_soft_${TAG}_eval.log 2>&1
done
echo "===== MU-STAT SWEEP DONE ====="
