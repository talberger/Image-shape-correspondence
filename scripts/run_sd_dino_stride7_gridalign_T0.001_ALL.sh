#!/bin/bash
set -e
cd ${DISC_ROOT:-/workspace/disc}/scripts
L=${DISC_WS:-/workspace}/logs
echo "===== [1/3] phase-2 T=0.001 ====="   ; ./run_sd_dino_stride7_gridalign_T0.001_phase2_train.sh > $L/gridalign_T0.001_phase2_train.log 2>&1
echo "===== [2/3] step progression ====="  ; ./run_sd_dino_stride7_gridalign_T0.001_analysis.sh     > $L/gridalign_T0.001_analysis.log 2>&1
echo "===== [3/3] eval + comparison =====" ; ./run_sd_dino_stride7_gridalign_T0.001_eval.sh         > $L/gridalign_T0.001_eval.log 2>&1
echo "===== DONE ====="
