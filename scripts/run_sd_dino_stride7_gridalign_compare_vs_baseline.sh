#!/bin/bash
# Side-by-side render comparison: the stride-7 + gridalign validation re-run against the run it is
# meant to reproduce -- the PROMOTED stride7 fixA run, not the older dfd_crossview default baked
# into make_run_vs_baseline_comparison.py's defaults.
# NOTE: these two are the SAME RECIPE. Any visible difference is the (previously unseeded) SD noise,
# not a recipe effect -- see PROJECT_NOTES.md 2026-08-28, section 4.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
python analysis/make_run_vs_baseline_comparison.py \
  --run_label "stride7 gridalign (re-run, early-exit + fp16)" \
  --run_zs_dir demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_gridalign_ZS \
  --run_zs_coverage "1006/4036 = 24.93%" \
  --run_phase2_dir demo/guitar/bsb_sd_dino_stride7_gridalign_phase2 \
  --run_phase2_coverage "1137/4036 = 28.17%" \
  --baseline_label "stride7 fixA -- PROMOTED BEST" \
  --baseline_zs_dir demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_fixA_ZS \
  --baseline_zs_coverage "995/4036 = 24.65%" \
  --baseline_phase2_dir demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_fixA_phase2 \
  --baseline_phase2_coverage "1162/4036 = 28.79%"
