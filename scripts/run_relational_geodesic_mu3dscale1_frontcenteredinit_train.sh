#!/bin/bash
set -e
source /workspace/activate_bsb.sh

# Phase-2 training, "current best" recipe (--geodesic --mu_3d_scale 1, no cyc_weight/
# --balance_weight -- see PROJECT_NOTES.md's "Current best recipe" line), but initialized from
# the 2026-08-16 --frontview_center-bug-fixed phase-1 run (encoder_dino_frontcentered) instead
# of the usual bsb_retrained. Mirrors
# run_relational_geodesic_mu3dscale1_fib1000init_analysis.sh's training step (that run's own
# training command was never saved as a script -- this one is, this time).

python phase2/train_relational_loss.py \
  --object_mask_path demo/guitar/bsb_origin/guitar_01_click_130_140_sam_mask_2.pth \
  --pix_dino_features_path demo/guitar/bsb_origin/guitar_01_pix_dino_features.pth \
  --init_encoder_dir experiments/guitar/encoder_dino_frontcentered \
  --out_dir experiments/guitar/encoder_dino_relational_geodesic_mu3dscale1_frontcenteredinit \
  --geodesic --mu_3d_scale 1

echo "=== training exit=$? ==="
