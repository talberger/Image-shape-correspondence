#!/usr/bin/env bash
# ISOBBOX at T=10, end to end (Tal, 2026-09-03). Extends the temperature sweep upward.
# EVAL IS READ AT T=1, like every other isobbox arm on record -- comparing evals across different
# read temperatures is the 2026-08-31 confound, so the eval read T is held at 1 deliberately.
# The matched read-T=10 back-projection is produced separately as a visual only.
set -uo pipefail
cd /workspace/bsb; L=/workspace/logs; fail=0
R=encoder_isobbox_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T10_seed0_phase2
D=bsb_isobbox_sd_dino_stride14_nosdsquare_T10_seed0_phase2
T=isobbox_stride14_nosdsquare_T10_seed0
run(){ echo "===== $(date +%H:%M:%S) $1"; shift; bash -c "$*" && echo "  OK" || { echo "  FAILED"; fail=1; }; }
run "1/3 phase-2 train T=10"           "./scripts/run_isobbox_T10_phase2_train.sh > $L/isobbox_T10_train.log 2>&1"
run "2/3 step progression"             "./scripts/run_isobbox_T10_analysis.sh     > $L/isobbox_T10_stepprog.log 2>&1"
run "3/3 small analyze (eval read T=1)" "./scripts/small_analyze_isobbox_readT1.sh $R $D $T > $L/isobbox_T10_small_analyze.log 2>&1"
echo "===== $(date +%H:%M:%S) T=10 QUEUE FINISHED (fail=$fail)"
