#!/usr/bin/env bash
# mu_3d_scale 0.99 / 0.95 / 0.9 on the isobbox mesh, squash, T=1, seed 0 (Tal, 2026-09-03).
# Train + small_analyze only, no step progression. Matched control in every comparison is the
# isobbox squash T=1 run -- i.e. mu_3d_scale=1, the same recipe with only this knob changed.
set -uo pipefail
cd ${DISC_ROOT:-/workspace/disc}; L=${DISC_WS:-/workspace}/logs; fail=0
run(){ echo "===== $(date +%H:%M:%S) $1"; shift; bash -c "$*" && echo "  OK" || { echo "  FAILED"; fail=1; }; }
i=0
for MU in 0.99 0.95 0.9; do
  i=$((i+1))
  R=encoder_isobbox_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale${MU}_T1_seed0_phase2
  D=bsb_isobbox_sd_dino_stride14_nosdsquare_mu3dscale${MU}_T1_seed0_phase2
  T=isobbox_stride14_nosdsquare_mu3dscale${MU}_T1_seed0
  run "$i/3 mu=$MU train + small analyze" \
    "./scripts/run_isobbox_mu3dscale${MU}_T1_phase2_train.sh > $L/isobbox_mu${MU}_T1_train.log 2>&1 && ./scripts/small_analyze_isobbox_readT1.sh $R $D $T > $L/isobbox_mu${MU}_T1_small_analyze.log 2>&1"
done
echo "===== $(date +%H:%M:%S) mu-near-1 QUEUE FINISHED (fail=$fail)"
