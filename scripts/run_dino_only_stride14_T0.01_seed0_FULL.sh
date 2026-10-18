#!/usr/bin/env bash
# DINO-ONLY arm of the current baseline recipe, end to end (Tal, 2026-09-01).
#
# THE QUESTION: what does Stable Diffusion actually contribute, measured on the n=100
# ground-truth eval, ON THE RECIPE WE ACTUALLY USE?
#
# WHY A NEW RUN WAS NECESSARY. Every DINO-only run already on disk predates DFD barycentric
# distillation, the L2-normalize fix and the current phase-2, and the photo features every
# 2026-08-20..26 DINO+SD run trained on were deleted (SD unseeded until 2026-08-28, so they
# cannot be regenerated). The best historical comparison available -- DINO-only 2026-08-17 vs
# DINO+SD sdaligned 2026-08-27, run earlier today -- is confounded by DFD cross-view and
# sd_square on top of SD itself:
#     geometric snap 78.3 vs 77.9 | feature argmax 74.2 vs 76.5 | soft mean 73.0 vs 75.4
# This run removes every one of those confounds.
#
# WHAT IS HELD IDENTICAL TO THE BASELINE (encoder_dino_sd_stride14_nosdsquare_..._T0.01_seed0):
#   - the same 1000 renders and the same DFD barycentric targets: this cache is written INTO
#     data/guitar/encoder_data_frontcentered_dinosd/, reusing 224/ and xyz_bary_f/ untouched
#   - stride 14, no gridalign (a no-op at stride 14), float32 cache (the baseline's dtype)
#   - phase-1: train_encoder_dfd_crossview.py, same 1000 views
#   - phase-2: --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0, same mask, same fine
#     segments, same 2D geodesic table
#   - eval read temperature 0.01, matching the baseline's own recorded read T
#
# THE ONE INTENDED DIFFERENCE: modules/dino_sd_fusion.py::dino_only() replaces fuse_dino_sd(),
# i.e. the 1280 SD channels are deleted and the DINO branch is treated bit-identically
# (F.normalize(dino) * 0.5, exactly as inside the fusion). SD is never loaded or run.
#
# THE ONE UNAVOIDABLE DIFFERENCE: width/n_classes 1024 instead of 2304, forced by the feature
# dimension. It cannot be held fixed -- a 2304-wide encoder cannot consume 1024-ch features --
# so a gap here is "DINO-only recipe vs DINO+SD recipe", with capacity riding along. Flagged,
# not solved.
#
# SEEDING: phase-2 is --train_seed 0, same as the baseline. Phase-1 and the DINO forward pass are
# deterministic (no SD noise draw exists in this arm at all), so this run is reproducible in a way
# no DINO+SD run is. Note the baseline it is compared against still carries its own SD draw, so
# the ~1.6-2.4 AUC seed floor applies to the COMPARISON even though this arm is itself clean.
#
# NO ANALYSES (Tal: "אין צורך בכל האנליזות") -- eval only.
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

CACHE=dino_only_stride14
DATA=./data/guitar/encoder_data_frontcentered_dinosd
P1=./experiments/guitar/encoder_dino_only_stride14_dfd_crossview_phase1
P2=./experiments/guitar/encoder_dino_only_stride14_relational_geodesic_mu3dscale1_T0.01_seed0_phase2
FEAT=demo/guitar/features_analyze/dino_only_stride14
OUT=demo/guitar/eval_labels

echo "########## 1/5 precompute 1000 views, DINO ONLY (no SD) ##########"
time python train_encoder_dino_sd.py \
  --encoder_data_dir $DATA \
  --encoder_model_dir $P1 \
  --view_sampling random --frontview_center 0 1.5707963267948966 --frontview_std 4 \
  --n_views 1000 --width 1024 --n_classes 1024 \
  --feature_cache_name $CACHE \
  --dino_only \
  --generate_random_views 0 --encode_random_views 1 --start_training 0

echo "########## 2/5 phase-1 (DFD cross-view), DINO ONLY ##########"
time python train_encoder_dfd_crossview.py \
  --encoder_data_dir $DATA \
  --encoder_model_dir $P1 \
  --feature_model_type $CACHE \
  --n_views 1000 --width 1024 --n_classes 1024

echo "########## 3/5 photo features, DINO ONLY, stride 14 ##########"
mkdir -p "$FEAT"
time python analysis/extract_dino_sd_features.py \
  --img_path images/guitar_01.png \
  --save_dir "$FEAT" \
  --output_path "$FEAT/guitar_01_pix_dino_only_stride14.pth" \
  --save_raw_path "$FEAT/guitar_01_pix_dino_only_stride14_RAW_pregrid.pth" \
  --dino_only \
  --cpu_postprocess

echo "########## 4/5 phase-2, T=0.01 seed 0 ##########"
time python phase2/train_relational_loss_l2norm.py \
  --init_encoder_dir $P1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./$FEAT/guitar_01_pix_dino_only_stride14.pth \
  --width 1024 --n_classes 1024 --geodesic --mu_3d_scale 1 --temperature 0.01 --train_seed 0 \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir $P2

if [ ! -f $P2/pred_f.pth ]; then
  echo "---------- exporting pred_f ----------"
  python analysis/export_pred_f_l2norm.py \
    --checkpoint $P2/encoder_checkpoint.pth --output $P2/pred_f.pth \
    --width 1024 --n_classes 1024
fi

echo "########## 5/5 EVAL, read T=0.01, vs BOTH baselines ##########"
python analysis/eval_geodesic_error.py \
  --eval_pairs_path $OUT/pixel_vertex_eval_pairs.pth \
  --features_path "$FEAT/guitar_01_pix_dino_only_stride14.pth" \
  --vertex_features_path $P2/pred_f.pth \
  --temperature 0.01 \
  --run_label "DINO only, stride14 T=0.01 seed0" \
  --plot_path $OUT/eval_dino_only_stride14_T0.01_seed0.png \
  --save_path $OUT/eval_dino_only_stride14_T0.01_seed0.pth

python analysis/compare_eval_runs.py \
  --runs $OUT/eval_dino_only_stride14_T0.01_seed0.pth \
         $OUT/eval_stride14_nosdsquare_T0.01_seed0.pth \
         $OUT/eval_stride14_nosdsquare_T1_seed0.pth \
  --labels "DINO only T=0.01" "baseline DINO+SD T=0.01" "companion DINO+SD T=1" \
  --save_path $OUT/eval_compare_dino_only_vs_dinosd_currentrecipe.png
