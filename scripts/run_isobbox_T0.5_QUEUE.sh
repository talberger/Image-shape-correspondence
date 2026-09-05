#!/usr/bin/env bash
# ISOBBOX at T=0.5, end to end with the FULL analysis (Tal, 2026-09-02).
# Waits on a PID, never a pgrep -f pattern (2026-08-31 section 9).
set -uo pipefail
cd /workspace/bsb; L=/workspace/logs; fail=0
if [ -n "${WAIT_PID:-}" ]; then
  echo "waiting for pid $WAIT_PID (the T=1 queue) ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi
run(){ echo "===== $1"; shift; bash -c "$*" && echo "  OK" || { echo "  FAILED"; fail=1; }; }
run "1/4 phase-2 train T=0.5"  "./scripts/run_isobbox_T0.5_phase2_train.sh  > $L/isobbox_T0.5_train.log 2>&1"
run "2/4 step progression"      "./scripts/run_isobbox_T0.5_analysis.sh      > $L/isobbox_T0.5_stepprog.log 2>&1"
run "3/4 full analyze"          "./scripts/run_isobbox_T0.5_full_analyze.sh  > $L/isobbox_T0.5_full_analyze.log 2>&1"
run "4/4 eval + comparisons"    "./scripts/small_analyze_isobbox.sh encoder_isobbox_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T0.5_seed0_phase2 bsb_isobbox_sd_dino_stride14_nosdsquare_T0.5_seed0_phase2 isobbox_stride14_nosdsquare_T0.5_seed0 > $L/isobbox_T0.5_small_analyze.log 2>&1"
run "5/5 matched comparison vs SD+DINO reference" "./scripts/compare_isobbox_arm_vs_sd_dino.sh bsb_isobbox_sd_dino_stride14_nosdsquare_T0.5_seed0_phase2 'ISOBBOX SD+DINO T=0.5 seed0' isobbox_stride14_nosdsquare_T0.5_seed0 > $L/isobbox_T0.5_cmp_ref.log 2>&1"
echo "T=0.5 QUEUE FINISHED (fail=$fail)"
