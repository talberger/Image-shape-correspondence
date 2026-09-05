#!/usr/bin/env bash
# ISOBBOX at T=1, end to end with the FULL analysis (Tal, 2026-09-02).
set -uo pipefail
cd /workspace/bsb; L=/workspace/logs; fail=0
run(){ echo "===== $1"; shift; bash -c "$*" && echo "  OK" || { echo "  FAILED"; fail=1; }; }
run "1/4 phase-2 train T=1"      "./scripts/run_isobbox_T1_phase2_train.sh  > $L/isobbox_T1_train.log 2>&1"
run "2/4 step progression"        "./scripts/run_isobbox_T1_analysis.sh      > $L/isobbox_T1_stepprog.log 2>&1"
run "3/4 full analyze"            "./scripts/run_isobbox_T1_full_analyze.sh  > $L/isobbox_T1_full_analyze.log 2>&1"
run "4/4 eval + comparisons"      "./scripts/small_analyze_isobbox.sh encoder_isobbox_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T1_seed0_phase2 bsb_isobbox_sd_dino_stride14_nosdsquare_T1_seed0_phase2 isobbox_stride14_nosdsquare_T1_seed0 > $L/isobbox_T1_small_analyze.log 2>&1"
echo "T=1 QUEUE FINISHED (fail=$fail)"
