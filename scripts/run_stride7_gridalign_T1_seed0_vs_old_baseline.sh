#!/bin/bash
# The NEW seeded baseline (T=1, --train_seed 0, 2026-08-30) against the OLD unseeded one
# (2026-08-28, the currently promoted reference run). Same recipe in every respect -- the ONLY
# difference is that phase-2's per-step neighbour resampling is now seeded. So this comparison is
# a direct picture of what one seed draw is worth, with nothing else varying.
# Output lands in the NEW run's own comparison_vs_old_baseline/ so it does not collide with the
# comparison_vs_baseline/ dirs that point at the new baseline.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
python analysis/make_run_vs_baseline_comparison.py \
  --run_label "s7 + gridalign, T=1, train_seed 0 -- NEW SEEDED BASELINE" \
  --run_phase2_dir demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_gridalign_T1_seed0_phase2 \
  --run_phase2_coverage "1064/4036 = 26.36%" \
  --baseline_label "s7 + gridalign, T=1, UNSEEDED -- old promoted baseline (2026-08-28)" \
  --baseline_phase2_dir demo/guitar/bsb_sd_dino_stride7_gridalign_phase2 \
  --baseline_phase2_coverage "1137/4036 = 28.17%" \
  --phase2_out_dir demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_gridalign_T1_seed0_phase2/comparison_vs_old_baseline
echo "OLD-VS-NEW BASELINE COMPARISON DONE"
