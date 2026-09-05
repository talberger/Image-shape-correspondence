#!/bin/bash
# Feature-space RKD with the ARGMAX student (--feat_student_mode argmax), on the NEW baseline
# (stride 14 + squash + T=0.01 + seed 0). 2026-09-01, Tal: "let's try, nothing to lose".
#
# WHY THIS VARIANT. The 2026-08-29 circularity test showed the soft (weighted-average) student lets
# the loss fall by re-mixing the softmax weights rather than by moving features (-64% attached vs
# -4% detached), and that detaching does not rescue it -- the gradient is still smeared across k
# vertices with frozen coefficients. argmax removes the mixing entirely: no sharpening knob exists,
# and a pixel's whole gradient lands on one vertex. Judge it on the eval metrics, NOT on the loss
# curve -- the assignment is self-supplied, so a wrong match can reinforce itself.
#
# Control = the baseline itself (same features, same phase-1, same seed, same GPU, both weights 0).
set -e
cd /workspace/bsb/scripts
L=/workspace/logs
EL=demo/guitar/eval_labels
PIX=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth

run_arm () {
  TAG=$1; BETA=$2; GAMMA=$3
  OUT=./experiments/guitar/encoder_dino_sd_stride14_nosdsquare_featrkd_argmax_${TAG}_T0.01_seed0_phase2
  echo "===== [train] $TAG  beta=$BETA gamma=$GAMMA ====="
  ( source /workspace/activate_bsb.sh; cd /workspace/bsb
    python phase2/train_relational_loss_l2norm.py \
      --init_encoder_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1 \
      --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
      --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
      --pix_dino_features_path ./$PIX \
      --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
      --feat_student_mode argmax --feat_dist_weight $BETA --feat_angle_weight $GAMMA \
      --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
      --out_dir $OUT ) > $L/featrkd_argmax_${TAG}.log 2>&1
  grep -E "feature-RKD student|feature-RKD weights" $L/featrkd_argmax_${TAG}.log
  grep -E "^step (0|300)\b" $L/featrkd_argmax_${TAG}.log | tail -2

  echo "===== [eval] $TAG ====="
  ( source /workspace/activate_bsb.sh; cd /workspace/bsb
    python analysis/eval_geodesic_error.py \
      --run_label "featrkd ARGMAX $TAG (stride14 squash T=0.01 seed0)" \
      --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth --features_path $PIX \
      --vertex_features_path ${OUT#./}/pred_f.pth --k_vertices_per_pixel 10 \
      --save_path $EL/eval_featrkd_argmax_${TAG}.pth \
      --plot_path $EL/eval_featrkd_argmax_${TAG}.png ) > $L/featrkd_argmax_${TAG}_eval.log 2>&1
}

run_arm dist0.2_ang0   0.2 0.0
run_arm dist0.2_ang0.2 0.2 0.2

echo "===== [compare] vs the baseline control ====="
( source /workspace/activate_bsb.sh; cd /workspace/bsb
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_stride14_nosdsquare_T0.01_seed0.pth \
           $EL/eval_featrkd_argmax_dist0.2_ang0.pth \
           $EL/eval_featrkd_argmax_dist0.2_ang0.2.pth \
    --labels "BASELINE (no feat terms)" "argmax featD 0.2" "argmax featD 0.2 + featA 0.2" \
    --save_path $EL/eval_compare_featrkd_argmax_vs_baseline.png ) > $L/featrkd_argmax_compare.log 2>&1
tail -22 $L/featrkd_argmax_compare.log
echo "===== DONE ====="
