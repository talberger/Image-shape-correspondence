#!/bin/bash
# beta sweep for the ARGMAX feature-RKD student, angle term OFF (2026-09-01, Tal approved).
# Context: at beta=0.2 the distance term moved features for real (featD -49%, vs -4% with the old
# detached soft student) and geometric snap rose +2.2 AUC over the baseline -- above the 1.6 AUC
# seed floor, the first positive signal this direction has produced. feature argmax and soft mean
# each slipped ~1 AUC, inside the floor. The angle term at gamma=0.2 destroyed the run (AUC 54/49/48),
# so it stays at 0 here.
# Arms: beta = 0.05, 0.1, 0.5. Everything else identical to the baseline and to the beta=0.2 arm:
# stride14 + squash photo features, phase-1 dfd_crossview, T=0.01, seed 0, same GPU.
set -e
cd /workspace/bsb/scripts
L=/workspace/logs
EL=demo/guitar/eval_labels
PIX=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth

for BETA in 0.05 0.1 0.5; do
  TAG=dist${BETA}_ang0
  OUT=./experiments/guitar/encoder_dino_sd_stride14_nosdsquare_featrkd_argmax_${TAG}_T0.01_seed0_phase2
  echo "===== [train] beta=$BETA ====="
  ( source /workspace/activate_bsb.sh; cd /workspace/bsb
    python phase2/train_relational_loss_l2norm.py \
      --init_encoder_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1 \
      --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
      --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
      --pix_dino_features_path ./$PIX \
      --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
      --feat_student_mode argmax --feat_dist_weight $BETA --feat_angle_weight 0.0 \
      --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
      --out_dir $OUT ) > $L/featrkd_argmax_${TAG}.log 2>&1
  grep -E "^step (25|300)/300" $L/featrkd_argmax_${TAG}.log | sed 's/(mu_2D.*featD/featD/'

  echo "===== [eval] beta=$BETA ====="
  ( source /workspace/activate_bsb.sh; cd /workspace/bsb
    python analysis/eval_geodesic_error.py \
      --run_label "featrkd ARGMAX beta=$BETA" \
      --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth --features_path $PIX \
      --vertex_features_path ${OUT#./}/pred_f.pth --k_vertices_per_pixel 10 \
      --save_path $EL/eval_featrkd_argmax_${TAG}.pth \
      --plot_path $EL/eval_featrkd_argmax_${TAG}.png ) > $L/featrkd_argmax_${TAG}_eval.log 2>&1
done

echo "===== [compare] full beta sweep vs baseline ====="
( source /workspace/activate_bsb.sh; cd /workspace/bsb
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_stride14_nosdsquare_T0.01_seed0.pth \
           $EL/eval_featrkd_argmax_dist0.05_ang0.pth \
           $EL/eval_featrkd_argmax_dist0.1_ang0.pth \
           $EL/eval_featrkd_argmax_dist0.2_ang0.pth \
           $EL/eval_featrkd_argmax_dist0.5_ang0.pth \
    --labels "BASELINE (beta=0)" "beta=0.05" "beta=0.1" "beta=0.2" "beta=0.5" \
    --save_path $EL/eval_compare_featrkd_argmax_beta_sweep.png ) > $L/featrkd_argmax_beta_compare.log 2>&1
tail -28 $L/featrkd_argmax_beta_compare.log
echo "===== SWEEP DONE ====="
