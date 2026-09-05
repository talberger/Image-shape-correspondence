#!/usr/bin/env bash
# ISOBBOX + sd_square photo features at T=1, end to end (Tal, 2026-09-03). THE FUSE EXPERIMENT.
# small_analyze gets the aligned photo features as its 4th argument -- without it the analysis would
# read the run's mesh features against the SQUASH photo features and silently measure a mismatch.
set -uo pipefail
cd /workspace/bsb; L=/workspace/logs; fail=0
R=encoder_isobbox_dino_sd_stride14_sdsquare_relational_geodesic_mu3dscale1_T1_seed0_phase2
D=bsb_isobbox_sd_dino_stride14_sdsquare_T1_seed0_phase2
T=isobbox_stride14_sdsquare_T1_seed0
PF=demo/guitar/features_analyze/dino_sd_aligned/guitar_01_pix_dino_sd_features_aligned.pth
run(){ echo "===== $(date +%H:%M:%S) $1"; shift; bash -c "$*" && echo "  OK" || { echo "  FAILED"; fail=1; }; }
run "1/3 phase-2 train sd_square T=1" "./scripts/run_isobbox_sdsquare_T1_phase2_train.sh > $L/isobbox_sdsquare_T1_train.log 2>&1"
run "2/3 step progression"            "./scripts/run_isobbox_sdsquare_T1_analysis.sh     > $L/isobbox_sdsquare_T1_stepprog.log 2>&1"
run "3/3 small analyze"               "./scripts/small_analyze_isobbox_readT1.sh $R $D $T $PF > $L/isobbox_sdsquare_T1_small_analyze.log 2>&1"
echo "===== $(date +%H:%M:%S) sd_square T=1 QUEUE FINISHED (fail=$fail)"
