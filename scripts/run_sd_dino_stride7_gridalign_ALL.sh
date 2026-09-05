#!/bin/bash
# End-to-end validation run: stride 7 + gridalign, with the 2026-08-28 speed changes
# (SD UNet early exit, float16 target cache). Everything from phase-1 to the ground-truth eval.
set -e
cd /workspace/bsb/scripts
L=/workspace/logs
echo "===== [1/7] phase-1 (precompute + DFD cross-view) ====="; ./run_sd_dino_stride7_gridalign_phase1.sh          > $L/gridalign_phase1.log 2>&1
echo "===== [2/7] photo features ====="                       ; ./run_sd_dino_stride7_gridalign_photo_features.sh  > $L/gridalign_photo_features.log 2>&1
echo "===== [3/7] ZS analysis ====="                          ; ./run_sd_dino_stride7_gridalign_ZS.sh              > $L/gridalign_ZS.log 2>&1
echo "===== [4/7] phase-2 training ====="                     ; ./run_sd_dino_stride7_gridalign_phase2_train.sh    > $L/gridalign_phase2_train.log 2>&1
echo "===== [5/7] step progression ====="                     ; ./run_sd_dino_stride7_gridalign_analysis.sh        > $L/gridalign_analysis.log 2>&1
echo "===== [6/7] full analyze ====="                         ; ./run_sd_dino_stride7_gridalign_full_analyze.sh    > $L/gridalign_full_analyze.log 2>&1
echo "===== [7/7] ground-truth eval + 4-way comparison ====="; ./run_sd_dino_stride7_gridalign_eval.sh             > $L/gridalign_eval.log 2>&1
echo "===== PIPELINE DONE ====="
