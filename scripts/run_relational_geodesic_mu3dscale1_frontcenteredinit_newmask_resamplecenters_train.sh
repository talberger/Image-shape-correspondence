#!/bin/bash
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"

# Same recipe/init/mask as run_relational_geodesic_mu3dscale1_frontcenteredinit_newmask_train.sh
# (--geodesic --mu_3d_scale 1, init from encoder_dino_frontcentered, 2026-08-17 hole-free mask),
# but with --resample_centers (2026-08-18): instead of sample_center_pixels's single fixed draw
# of 2048 centers reused for the whole run, a fresh random 2048-pixel subset of a candidate grid
# (stride 5, ~8.5k candidates -- still ~4x the old fixed 2048) is drawn every step -- the
# neighbours-side counterpart of this already existed (resampled every step by default); this
# extends the same idea to centers. Stride 5, not the denser stride 3 first tried: a stride-3
# grid (~23.5k candidates) OOM-killed the table build at this container's 58GB memory limit the
# first time (see build_2d_table's docstring in analysis/build_geodesic_tables.py for the fix
# applied since -- chunked dijkstra, bounds peak memory regardless of stride) and stride 5 is
# also ~3x faster to build. Requires
# experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_gridstride5.pth
# to exist first -- built via analysis/build_geodesic_tables.py --center_mode grid --grid_stride 5
# --skip_3d (the 3D table is unaffected, reused from the existing run's build).

python phase2/train_relational_loss.py \
  --object_mask_path demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path demo/guitar/bsb_origin/guitar_01_pix_dino_features.pth \
  --init_encoder_dir experiments/guitar/encoder_dino_frontcentered \
  --out_dir experiments/guitar/encoder_dino_relational_geodesic_mu3dscale1_frontcenteredinit_newmask_resamplecenters \
  --geo_table_2d_path experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_gridstride5.pth \
  --geodesic --mu_3d_scale 1 --resample_centers --center_grid_stride 5

echo "=== training exit=$? ==="
