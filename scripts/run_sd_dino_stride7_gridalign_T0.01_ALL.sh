#!/bin/bash
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs
echo "===== [1/5] phase-2 T=0.01 ====="  ; ./run_sd_dino_stride7_gridalign_T0.01_phase2_train.sh > $L/gridalign_T0.01_phase2_train.log 2>&1
echo "===== [2/5] step progression ====="; ./run_sd_dino_stride7_gridalign_T0.01_analysis.sh     > $L/gridalign_T0.01_analysis.log 2>&1
echo "===== [3/5] full analyze ====="    ; ./run_sd_dino_stride7_gridalign_T0.01_full_analyze.sh > $L/gridalign_T0.01_full_analyze.log 2>&1
echo "===== [4/5] eval + T sweep ====="  ; ./run_sd_dino_stride7_gridalign_T0.01_eval.sh         > $L/gridalign_T0.01_eval.log 2>&1
COV=$(grep -E "vertex coverage" $L/gridalign_T0.01_full_analyze.log | tail -1 | sed -E "s#.*coverage: ([0-9]+/4036) vertices \\(([0-9.]+%)\\).*#\\1 = \\2#")
echo "===== [5/5] compare vs best (coverage $COV) ====="
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
python analysis/make_run_vs_baseline_comparison.py \
  --run_label "s7 both fixes, T=0.01" \
  --run_phase2_dir demo/guitar/bsb_sd_dino_stride7_gridalign_T0.01_phase2 \
  --run_phase2_coverage "$COV" \
  --baseline_label "s7 both fixes, T=1 -- PROMOTED BEST" \
  --baseline_phase2_dir demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_fixA_phase2 \
  --baseline_phase2_coverage "1162/4036 = 28.79%" > $L/gridalign_T0.01_compare.log 2>&1
echo "===== DONE ====="
