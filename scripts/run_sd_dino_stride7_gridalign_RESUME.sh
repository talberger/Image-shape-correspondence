#!/bin/bash
# Resume the end-to-end validation run from phase-2 (steps 1-3 completed successfully).
set -e
cd ${DISC_ROOT:-/workspace/disc}/scripts
L=${DISC_WS:-/workspace}/logs
echo "===== [4/7] phase-2 training ====="                     ; ./run_sd_dino_stride7_gridalign_phase2_train.sh    > $L/gridalign_phase2_train.log 2>&1
echo "===== [5/7] step progression ====="                     ; ./run_sd_dino_stride7_gridalign_analysis.sh        > $L/gridalign_analysis.log 2>&1
echo "===== [6/7] full analyze ====="                         ; ./run_sd_dino_stride7_gridalign_full_analyze.sh    > $L/gridalign_full_analyze.log 2>&1
echo "===== [7/7] ground-truth eval + 4-way comparison ====="; ./run_sd_dino_stride7_gridalign_eval.sh             > $L/gridalign_eval.log 2>&1
echo "===== PIPELINE DONE ====="
