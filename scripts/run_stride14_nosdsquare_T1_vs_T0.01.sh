#!/bin/bash
# The training-temperature decision for the NEW baseline (Tal, 2026-09-01): stride 14 + squash
# (no sd_square), T=1 vs T=0.01. Matched in everything else -- same photo features, same phase-1,
# same seed 0, same GPU (A4500), same pod. The T=0.01 arm already exists from this pod's
# isolation run; this trains and analyses the T=1 arm and compares the two.
set -e
cd /workspace/bsb/scripts
L=/workspace/logs
EL=demo/guitar/eval_labels
D=demo/guitar
RUN=experiments/guitar/encoder_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T1_seed0_phase2
T1DIR=$D/bsb_sd_dino_stride14_nosdsquare_T1_seed0_phase2
T001DIR=$D/bsb_sd_dino_stride14_nosdsquare_T0.01_seed0_phase2
PIX=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth
BP=back_projection/guitar_01_click_270_370_sam_mask_2_backproj_guitar_mesh_checker16_color_soft_snap_k1_side_by_side.png

echo "===== [1/5] train T=1 seed0 ====="
./run_stride14_nosdsquare_T1_seed0_phase2_train.sh > $L/stride14_nosdsquare_T1_seed0_train.log 2>&1

echo "===== [2/5] eval ====="
( source /workspace/activate_bsb.sh; cd /workspace/bsb
  python analysis/eval_geodesic_error.py \
    --run_label "stride14 no-sd_square T=1 seed0" \
    --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth --features_path $PIX \
    --vertex_features_path $RUN/pred_f.pth --k_vertices_per_pixel 10 \
    --save_path $EL/eval_stride14_nosdsquare_T1_seed0.pth \
    --plot_path $EL/eval_stride14_nosdsquare_T1_seed0.png
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_stride14_nosdsquare_T1_seed0.pth $EL/eval_stride14_nosdsquare_T0.01_seed0.pth $EL/eval_T0.01_phase2.pth \
    --labels "stride14 T=1 seed0" "stride14 T=0.01 seed0" "LIKED stride14 T=0.01 (08-27)" \
    --save_path $EL/eval_compare_stride14_nosdsquare_T1_vs_T0.01.png ) > $L/stride14_nosdsquare_T1_seed0_eval.log 2>&1
grep -A5 "^geometric snap\|^feature argmax\|^soft mean" $L/stride14_nosdsquare_T1_seed0_eval.log | head -20

echo "===== [3/5] step progression ====="
./run_stride14_nosdsquare_T1_seed0_analysis.sh > $L/stride14_nosdsquare_T1_seed0_analysis.log 2>&1

echo "===== [4/5] full analyze (read T=1) ====="
./run_stride14_nosdsquare_T1_seed0_full_analyze.sh > $L/stride14_nosdsquare_T1_seed0_full_analyze.log 2>&1
grep -h "vertex coverage" $L/stride14_nosdsquare_T1_seed0_full_analyze.log | tail -1

echo "===== [5/5] comparisons ====="
( source /workspace/activate_bsb.sh; cd /workspace/bsb
  # back-projection: T=1 read at 1 vs T=0.01 read at 0.01 -- each arm at its own native read T,
  # which is how each would actually be used as a baseline.
  python analysis/stack_images_labeled.py \
    --top_path "$T1DIR/analyze/$BP" \
    --top_label "stride14 + squash, train T=1, read T=1 (seed0)" \
    --bottom_path "$T001DIR/analyze_readT/$BP" \
    --bottom_label "stride14 + squash, train T=0.01, read T=0.01 (seed0)" \
    --title "NEW BASELINE, training-temperature A/B -- each arm at its own read temperature" \
    --output_path "$T1DIR/comparison_vs_T0.01/backproj_T1_vs_T0.01_native_readT.png"
  # and against the run Tal likes
  python analysis/stack_images_labeled.py \
    --top_path "$T1DIR/analyze/$BP" \
    --top_label "stride14 + squash, train T=1, read T=1 (seed0)" \
    --bottom_path "$D/bsb_sd_dino_dfd_crossview_relational_geodesic_mu3dscale1_T0.01_phase2/analyze/$BP" \
    --bottom_label "LIKED: stride14 + squash T=0.01, read T=0.01 (2026-08-27)" \
    --title "stride14 + squash T=1 vs the liked run" \
    --output_path "$T1DIR/comparison_vs_liked_T0.01/backproj_T1_vs_liked.png" ) \
  > $L/stride14_nosdsquare_T1_seed0_comparisons.log 2>&1
tail -2 $L/stride14_nosdsquare_T1_seed0_comparisons.log
echo "===== T=1 ARM DONE ====="
