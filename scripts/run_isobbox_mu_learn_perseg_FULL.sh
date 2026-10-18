#!/bin/bash
# FULL analysis of the NEW MAIN BASELINE of 2026-09-15 (in-segment-only neighbours + LEARNED per-segment mu):
# the per-arm battery WITH mesh renders (SMALL unset), then the step progression.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh
unset SMALL

SUFFIX=sdsquare_fuse16_truediag2d_mu_learn_perseg_T1_seed0
DIR=./experiments/guitar/encoder_isobbox_${SUFFIX}_phase2
RUN=bsb_isobbox_sdsquare_mu_learn_perseg_T1_seed0_phase2

echo "===== 1/2 full per-arm analysis (with mesh renders) ====="
./scripts/analyze_isobbox_arm.sh "$RUN" "$DIR/pred_f.pth" "$PIX_sdsquare_fuse16" "isobbox_${SUFFIX}"

echo "===== 2/2 step progression ====="
./scripts/run_isobbox_mu_learn_perseg_analysis.sh
echo "===== FULL DONE: $RUN ====="
