#!/usr/bin/env bash
# ISOBBOX, T=1, mu_3d_scale=0.5, end to end (Tal, 2026-09-03).
# small_analyze only -- no full_analyze, and read T=1 throughout (trained at T=1).
set -uo pipefail
cd ${DISC_ROOT:-/workspace/disc}; L=${DISC_WS:-/workspace}/logs; fail=0
R=encoder_isobbox_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale0.5_T1_seed0_phase2
D=bsb_isobbox_sd_dino_stride14_nosdsquare_mu3dscale0.5_T1_seed0_phase2
T=isobbox_stride14_nosdsquare_mu3dscale0.5_T1_seed0
run(){ echo "===== $(date +%H:%M:%S) $1"; shift; bash -c "$*" && echo "  OK" || { echo "  FAILED"; fail=1; }; }
run "1/3 phase-2 train T=1 mu_3d_scale=0.5" "./scripts/run_isobbox_mu3dscale0.5_T1_phase2_train.sh > $L/isobbox_mu0.5_T1_train.log 2>&1"
run "2/3 step progression"                  "./scripts/run_isobbox_mu3dscale0.5_T1_analysis.sh   > $L/isobbox_mu0.5_T1_stepprog.log 2>&1"
run "3/3 small analyze (read T=1 only)"      "./scripts/small_analyze_isobbox_readT1.sh $R $D $T  > $L/isobbox_mu0.5_T1_small_analyze.log 2>&1"
echo "===== $(date +%H:%M:%S) mu_3d_scale=0.5 T=1 QUEUE FINISHED (fail=$fail)"
