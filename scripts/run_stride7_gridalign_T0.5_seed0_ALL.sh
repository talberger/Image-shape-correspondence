#!/bin/bash
# T=0.5 on the current baseline recipe (stride 7 + gridalign + sd_square), --train_seed 0.
# Tal's request 2026-08-30: the sweep had 1 / 0.1 / 0.01 / 0.001 and nothing between 1 and 0.1.
# Same five stages every other arm got, so it drops straight into the comparison.
# Sequential -- one GPU job at a time (the 2026-08-29 OOM rule).
set -e
cd /workspace/bsb/scripts
L=/workspace/logs
EL=demo/guitar/eval_labels
D=/workspace/bsb/experiments/guitar/encoder_dino_sd_stride7_gridalign_relational_geodesic_mu3dscale1_T0.5_seed0_phase2
rm -f $L/gridalign_T0.5_seed0_*.log

echo "===== [1/5] phase-2 T=0.5 ====="
./run_stride7_gridalign_T0.5_seed0_phase2_train.sh > $L/gridalign_T0.5_seed0_train.log 2>&1

echo "===== [2/5] eval ====="
( source /workspace/activate_bsb.sh; cd /workspace/bsb
  python analysis/eval_geodesic_error.py \
    --run_label "stride7 gridalign T=0.5 seed0" \
    --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth \
    --features_path demo/guitar/features_analyze/dino_sd_stride7_gridalign/guitar_01_pix_stride7_gridalign.pth \
    --vertex_features_path $D/pred_f.pth \
    --k_vertices_per_pixel 10 \
    --save_path $EL/eval_stride7_gridalign_T0.5_seed0.pth \
    --plot_path $EL/eval_stride7_gridalign_T0.5_seed0.png ) > $L/gridalign_T0.5_seed0_eval.log 2>&1

echo "===== [3/5] step progression ====="
./run_stride7_gridalign_T0.5_seed0_analysis.sh > $L/gridalign_T0.5_seed0_analysis.log 2>&1

echo "===== [4/5] full analyze ====="
./run_stride7_gridalign_T0.5_seed0_full_analyze.sh > $L/gridalign_T0.5_seed0_full_analyze.log 2>&1
COV=$(grep -E "vertex coverage" $L/gridalign_T0.5_seed0_full_analyze.log | tail -1 \
      | sed -E "s#.*coverage: ([0-9]+/4036) vertices \(([0-9.]+%)\).*#\1 = \2#")

echo "===== [5/5] compare vs baseline (coverage $COV) + 5-way T sweep ====="
( source /workspace/activate_bsb.sh; cd /workspace/bsb
  python analysis/make_run_vs_baseline_comparison.py \
    --run_label "s7 + gridalign, T=0.5, train_seed 0" \
    --run_phase2_dir demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_gridalign_T0.5_seed0_phase2 \
    --run_phase2_coverage "$COV" ) > $L/gridalign_T0.5_seed0_compare.log 2>&1
( source /workspace/activate_bsb.sh; cd /workspace/bsb
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_stride7_gridalign_T1_seed0.pth $EL/eval_stride7_gridalign_T0.5_seed0.pth \
           $EL/eval_stride7_gridalign_T0.1_seed0.pth $EL/eval_stride7_gridalign_T0.01_seed0.pth \
           $EL/eval_stride7_gridalign_T0.001_seed0.pth \
    --labels "T=1" "T=0.5" "T=0.1" "T=0.01" "T=0.001" \
    --save_path $EL/eval_compare_stride7_T_sweep_seed0.png ) > $L/gridalign_T_sweep5_compare.log 2>&1

echo "===== T0.5 ALL DONE  (coverage $COV) ====="
