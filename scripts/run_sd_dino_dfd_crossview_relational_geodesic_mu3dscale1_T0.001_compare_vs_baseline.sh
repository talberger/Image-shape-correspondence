#!/usr/bin/env bash
# Stacks the T=0.001 phase-2 run's analysis renders directly above the current-best-recipe
# baseline's (same recipe, temperature=1.0), so both can be compared visually in one image.
#
# ZS comparisons are deliberately skipped: --temperature only affects phase-2 relational-loss
# training. Both runs share the SAME phase-1 init (encoder_dino_sd_dfd_crossview_phase1), so
# their ZS outputs are byte-identical and stacking them would compare a run against itself.
#
# Coverage numbers are passed in explicitly (the tool never recomputes them) -- 1239/4036 = 30.70%
# is the soft-snap coverage printed by color_mesh_from_soft_mapping.py in
# ${DISC_WS:-/workspace}/logs/T0.001_full_analyze.log.
set -e
cd ${DISC_ROOT:-/workspace/disc}

python analysis/make_run_vs_baseline_comparison.py \
  --run_label "T=0.001 softmax" \
  --run_phase2_dir demo/guitar/runs/legacy/bsb_sd_dino_dfd_crossview_relational_geodesic_mu3dscale1_T0.001_phase2 \
  --run_phase2_coverage "1239/4036 = 30.70%"
