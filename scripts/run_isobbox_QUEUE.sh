#!/usr/bin/env bash
# Tal's queue, 2026-09-02, in order:
#   1. FULL analyze of the isobbox baseline (step progression first, then the full suite)
#   2. isobbox at --temperature 0.1 + small_analyze
#   3. isobbox DINO-ONLY (no SD) + small_analyze
set -uo pipefail
cd /workspace/bsb
L=/workspace/logs
fail=0
run () { echo "===== $1 ====="; shift; "$@" && echo "  OK" || { echo "  FAILED: $*"; fail=1; }; }

run "1a/5 step progression (isobbox T=0.01)"  ./scripts/run_isobbox_T0.01_analysis.sh      > $L/isobbox_T0.01_stepprog.log 2>&1
run "1b/5 full analyze (isobbox T=0.01)"      ./scripts/run_isobbox_T0.01_full_analyze.sh  > $L/isobbox_T0.01_full_analyze.log 2>&1
run "2a/5 phase-2 train T=0.1"                ./scripts/run_isobbox_T0.1_phase2_train.sh   > $L/isobbox_T0.1_train.log 2>&1
run "2b/5 small analyze T=0.1"                ./scripts/small_analyze_isobbox.sh \
      encoder_isobbox_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T0.1_seed0_phase2 \
      bsb_isobbox_sd_dino_stride14_nosdsquare_T0.1_seed0_phase2 \
      isobbox_stride14_nosdsquare_T0.1_seed0                                               > $L/isobbox_T0.1_small_analyze.log 2>&1
run "3a/5 DINO-only phase-1 + phase-2"        ./scripts/run_isobbox_dinoonly_phase1_phase2.sh > $L/isobbox_dinoonly_p1p2.log 2>&1
run "3b/5 small analyze DINO-only"            ./scripts/small_analyze_isobbox.sh \
      encoder_isobbox_dino_only_stride14_relational_geodesic_mu3dscale1_T0.01_seed0_phase2 \
      bsb_isobbox_dino_only_stride14_T0.01_seed0_phase2 \
      isobbox_dino_only_stride14_T0.01_seed0 \
      demo/guitar/features_analyze/dino_only_stride14/guitar_01_pix_dino_only_stride14.pth  > $L/isobbox_dinoonly_small_analyze.log 2>&1

echo "QUEUE FINISHED (fail=$fail)"
