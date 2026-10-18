#!/bin/bash
# stride 14 + sd_square + T=0.01, then compared against the SAME recipe WITHOUT sd_square
# (demo/guitar/runs/legacy/bsb_sd_dino_dfd_crossview_relational_geodesic_mu3dscale1_T0.01_phase2), which is
# the run whose back-projection Tal singled out. Isolates sd_square at T=0.01.
# Sequential -- one GPU job at a time (the 2026-08-29 OOM rule).
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs; EL=demo/guitar/eval_labels
D=$DISC_ROOT/experiments/guitar/encoder_dino_sd_dfd_crossview_sdaligned_T0.01_seed0_phase2
rm -f $L/sdaligned_T0.01_seed0_*.log

echo "===== [1/5] phase-2 ====="
./run_sdaligned_T0.01_seed0_phase2_train.sh > $L/sdaligned_T0.01_seed0_train.log 2>&1

echo "===== [2/5] eval ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python analysis/eval_geodesic_error.py \
    --run_label "stride14 + sd_square, T=0.01 seed0" \
    --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth \
    --features_path demo/guitar/features_analyze/dino_sd_aligned/guitar_01_pix_dino_sd_features_aligned.pth \
    --vertex_features_path $D/pred_f.pth --k_vertices_per_pixel 10 \
    --save_path $EL/eval_sdaligned_T0.01_seed0.pth \
    --plot_path $EL/eval_sdaligned_T0.01_seed0.png ) > $L/sdaligned_T0.01_seed0_eval.log 2>&1

echo "===== [3/5] step progression ====="
./run_sdaligned_T0.01_seed0_analysis.sh > $L/sdaligned_T0.01_seed0_analysis.log 2>&1

echo "===== [4/5] full analyze ====="
./run_sdaligned_T0.01_seed0_full_analyze.sh > $L/sdaligned_T0.01_seed0_full_analyze.log 2>&1
COV=$(grep -E "vertex coverage" $L/sdaligned_T0.01_seed0_full_analyze.log | tail -1 \
      | sed -E "s#.*coverage: ([0-9]+/4036) vertices \(([0-9.]+%)\).*#\1 = \2#")

echo "===== [5/5] compare vs the SAME recipe WITHOUT sd_square (coverage $COV) ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python analysis/make_run_vs_baseline_comparison.py \
    --run_label "stride 14, T=0.01, WITH sd_square" \
    --run_phase2_dir demo/guitar/runs/legacy/bsb_sd_dino_dfd_crossview_sdaligned_T0.01_seed0_phase2 \
    --run_phase2_coverage "$COV" \
    --baseline_label "stride 14, T=0.01, NO sd_square (the one Tal liked)" \
    --baseline_phase2_dir demo/guitar/runs/legacy/bsb_sd_dino_dfd_crossview_relational_geodesic_mu3dscale1_T0.01_phase2 \
    --baseline_phase2_coverage "n/a" \
    --phase2_out_dir demo/guitar/runs/legacy/bsb_sd_dino_dfd_crossview_sdaligned_T0.01_seed0_phase2/comparison_vs_no_sd_square \
  ) > $L/sdaligned_T0.01_seed0_compare.log 2>&1
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_T0.01_phase2.pth $EL/eval_sdaligned_T0.01_seed0.pth \
    --labels "stride 14, T=0.01, NO sd_square" "stride 14, T=0.01, WITH sd_square" \
    --save_path $EL/eval_compare_sd_square_at_T0.01.png ) > $L/sdaligned_T0.01_seed0_evalcmp.log 2>&1

echo "===== SDALIGNED T0.01 ALL DONE  (coverage $COV) ====="
