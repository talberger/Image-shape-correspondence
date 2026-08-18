#!/bin/bash
set -e
source /workspace/activate_bsb.sh

# Same recipe/init as run_relational_geodesic_mu3dscale1_frontcenteredinit_train.sh
# (--geodesic --mu_3d_scale 1, init from encoder_dino_frontcentered -- see PROJECT_NOTES.md's
# "Current best recipe" line), but using the 2026-08-17 hole-free coarse mask + fine map
# (segment_masks_pixel270_370/, see PROJECT_NOTES.md's "Current coarse/fine segmentation
# masks" line) instead of the original holey bsb_origin mask. Requires
# experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth
# to exist first -- built once via analysis/build_geodesic_tables.py (see that script's own
# docstring for the run command); --geo_table_3d_path defaults to the shared mesh-only table,
# unaffected by the mask change.

python phase2/train_relational_loss.py \
  --object_mask_path demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path demo/guitar/bsb_origin/guitar_01_pix_dino_features.pth \
  --init_encoder_dir experiments/guitar/encoder_dino_frontcentered \
  --out_dir experiments/guitar/encoder_dino_relational_geodesic_mu3dscale1_frontcenteredinit_newmask \
  --geodesic --mu_3d_scale 1

echo "=== training exit=$? ==="
