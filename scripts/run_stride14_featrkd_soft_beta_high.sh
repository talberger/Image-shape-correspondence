#!/bin/bash
# SOFT-student feature-RKD, distance term ONLY, on the CURRENT baseline (Tal, 2026-08-31).
# The soft student was only ever tested on the OLD stride7+gridalign baseline at beta=0.2, where it
# failed and the circularity test closed it. It has never been run on the current recipe
# (stride14 + squash + T=0.01 + seed 0), and never as a beta sweep. Angle term stays 0 -- Tal wants
# dist working first, angle added afterwards.
# Everything else is identical to the argmax sweep: same phase-1, same photo features, same seed.
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs
EL=demo/guitar/eval_labels
PIX=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth

for BETA in 0.5 1.0; do
  TAG=dist${BETA}_ang0
  OUT=./experiments/guitar/encoder_dino_sd_stride14_nosdsquare_featrkd_soft_${TAG}_T0.01_seed0_phase2
  echo "===== [train] soft beta=$BETA ====="
  ( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
    python phase2/train_relational_loss_l2norm.py \
      --init_encoder_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1 \
      --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
      --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
      --pix_dino_features_path ./$PIX \
      --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
      --feat_student_mode soft --rkd_feat_dist_weight $BETA --rkd_feat_angle3_weight 0.0 \
      --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
      --out_dir $OUT ) > $L/featrkd_soft_${TAG}.log 2>&1
  grep -E "^step (25|300)/300" $L/featrkd_soft_${TAG}.log | sed 's/(mu_2D.*featD/featD/'

  echo "===== [eval] soft beta=$BETA ====="
  ( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
    python analysis/eval_geodesic_error.py \
      --run_label "featrkd SOFT beta=$BETA" \
      --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth --features_path $PIX \
      --vertex_features_path ${OUT#./}/pred_f.pth --k_vertices_per_pixel 10 \
      --save_path $EL/eval_featrkd_soft_${TAG}.pth \
      --plot_path $EL/eval_featrkd_soft_${TAG}.png ) > $L/featrkd_soft_${TAG}_eval.log 2>&1
done

echo "===== [compare] high-beta soft vs BOTH baselines ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_stride14_nosdsquare_T0.01_seed0.pth \
           $EL/eval_stride14_nosdsquare_T1_seed0.pth \
           $EL/eval_featrkd_soft_dist0.05_ang0.pth \
           $EL/eval_featrkd_soft_dist0.1_ang0.pth \
           $EL/eval_featrkd_soft_dist0.5_ang0.pth \
           $EL/eval_featrkd_soft_dist1.0_ang0.pth \
    --labels "BASELINE T=0.01" "BASELINE companion T=1" "soft 0.05" "soft 0.1" "soft 0.5" "soft 1.0" \
    --save_path $EL/eval_compare_featrkd_soft_beta_HIGH.png ) > $L/featrkd_soft_high_compare.log 2>&1
tail -28 $L/featrkd_soft_high_compare.log
echo "===== HIGH-BETA SWEEP DONE ====="
