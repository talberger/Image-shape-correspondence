#!/usr/bin/env bash
# PROBE ONLY (2026-09-04): 25 steps of the isobbox T=1 baseline with --log_feature_terms, to read
# the RAW magnitude of the new pairwise-angle term (featPA) against the spatial loss. Weight is 0,
# so this is the BASELINE trajectory -- it trains nothing new and its output is thrown away. The
# only purpose is to choose --feat_pair_angle_weight from a measurement instead of a guess.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb
# 16 GB GPU on this pod (earlier pods had 24 GB): 4.4 GB was lost to allocator fragmentation
# (13.53 GiB reserved vs 9.13 GiB allocated) on the first attempt. Capping split size recovers it.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python phase2/train_relational_loss_l2norm.py \
  --obj_path ./meshes/guitar_isotropic_bboxnorm.obj --name guitar_isobbox \
  --init_encoder_dir ./experiments/guitar/encoder_isobbox_dino_sd_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 1.0 --train_seed 0 \
  --log_feature_terms --feat_pair_angle_weight 0.0 --feat_pair_angle_neighbors in_segment --num_iters 25 \
  --geo_table_3d_path ./experiments/guitar/geodesic_tables/guitar_isobbox_geo_table_3d.pth \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir /tmp/claude-0/-workspace/9b25fa26-e3a2-4bec-a606-37fd764be2fd/scratchpad/pairangle_probe_inseg
