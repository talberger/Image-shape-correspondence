#!/bin/bash
# T=0.001 vs the promoted best (same recipe, T=1). Only the softmax temperature differs.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
python analysis/make_run_vs_baseline_comparison.py \
  --run_label "s7 both fixes, T=0.001" \
  --run_phase2_dir demo/guitar/bsb_sd_dino_stride7_gridalign_T0.001_phase2 \
  --run_phase2_coverage "${1:?usage: $0 '<phase2 coverage>'}" \
  --baseline_label "s7 both fixes, T=1 -- PROMOTED BEST" \
  --baseline_phase2_dir demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_fixA_phase2 \
  --baseline_phase2_coverage "1162/4036 = 28.79%"
