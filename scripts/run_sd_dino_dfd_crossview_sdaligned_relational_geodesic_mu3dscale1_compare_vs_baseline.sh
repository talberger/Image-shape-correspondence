#!/usr/bin/env bash
# SD-frame-aligned run vs the promoted baseline (identical recipe; only the photo features differ).
# ZS half included: the fix changes the photo features, so ZS differs even though the phase-1
# checkpoint is bit-identical.
set -e
cd ${DISC_ROOT:-/workspace/disc}
python analysis/make_run_vs_baseline_comparison.py \
  --run_label "SD-frame FIX" \
  --run_zs_dir demo/guitar/runs/legacy/bsb_sd_dino_dfd_crossview_frontcentered_sdaligned_ZS \
  --run_zs_coverage "697/4036 = 17.27%" \
  --run_phase2_dir demo/guitar/runs/legacy/bsb_sd_dino_dfd_crossview_sdaligned_relational_geodesic_mu3dscale1_phase2 \
  --run_phase2_coverage "1101/4036 = 27.28%"
