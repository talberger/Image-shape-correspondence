#!/bin/bash
# SHIC-bear, 2026-09-17 (Tal): FULL analysis of the MAIN bear arm, then phase-1 retrained with the view
# distribution centred on the PHOTO's viewpoint (azim 135 deg) instead of the default azim 0 (the far side).
# ONE change vs experiments/shic_bear/encoder_dfd_crossview_phase1: --frontview_center 0 2.356194490.
# Then ZS with the MAIN photo features (sd_square fuse16), old vs new phase-1, same read T, + extent diagnostic.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb

S=sdsquare_truediag2d_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0
PIX=demo/shic/features_analyze/sdsquare_fuse16/bear3_pix_sdsquare_fuse16.pth

echo "==================== 1/3 FULL analysis, MAIN bear arm ===================="
P1=experiments/shic_bear/encoder_dfd_crossview_phase1 P2=experiments/shic_bear/encoder_${S}_phase2 \
RUN=demo/shic/bear3_${S}_phase2 PIX=$PIX READ_T=0.01 LABEL="MAIN recipe, T=0.01" \
  bash scripts/analyze_shicbear_arm_FULL.sh || echo "!!! FULL analysis FAILED -- continuing to phase-1"

echo "==================== 2/3 phase-1, views centred at azim 135 ===================="
OBJ=./demo/shic/shapes/bear.obj
NAME=shic_bear
DATA=./data/shic_bear/encoder_data_az135
P1N=./experiments/shic_bear/encoder_dfd_crossview_az135_phase1
if [ -f "$P1N/pred_f.pth" ]; then echo "exists: $P1N"; else
  python train_encoder_dino_sd.py --obj_path $OBJ --name $NAME \
    --encoder_data_dir $DATA --encoder_model_dir $P1N \
    --view_sampling random --frontview_center 0 2.356194490192345 --frontview_std 4 \
    --n_views 1000 --width 2304 --n_classes 2304 --feature_cache_name dino_sd \
    --generate_random_views 1 --encode_random_views 1 --start_training 0
  python train_encoder_dfd.py --obj_path $OBJ --name $NAME \
    --encoder_data_dir $DATA --encoder_model_dir $P1N \
    --n_views 1000 --width 2304 --n_classes 2304 --generate_xyz_targets 1 --start_training 0
  python train_encoder_dfd_crossview.py --obj_path $OBJ --name $NAME \
    --encoder_data_dir $DATA --encoder_model_dir $P1N \
    --feature_model_type dino_sd --n_views 1000 --width 2304 --n_classes 2304
fi

echo "==================== 3/3 ZS old vs new phase-1 (sd_square photo features, read T=0.01) ===================="
for tag in old new; do
  if [ $tag = old ]; then PF=experiments/shic_bear/encoder_dfd_crossview_phase1; else PF=$P1N; fi
  P2=$PF RUN=demo/shic/bear3_ZS_sdsquare_fuse16_phase1${tag} PIX=$PIX READ_T=0.01 \
    LABEL="ZS, phase-1 $tag ($([ $tag = old ] && echo 'views azim 0' || echo 'views azim 135'))" \
    bash scripts/analyze_shicbear_arm.sh
done
python analysis/topmatch_geo_extent.py \
  --geo_table_3d_path experiments/shic_bear/geodesic_tables/shic_bear_geo_table_3d.pth \
  --knn_paths demo/shic/bear3_ZS_sdsquare_fuse16_phase1old/analyze/ours/bear3_click_350_300_sam_mask_1_pixel_to_vertex_nn.pth \
              demo/shic/bear3_ZS_sdsquare_fuse16_phase1new/analyze/ours/bear3_click_350_300_sam_mask_1_pixel_to_vertex_nn.pth \
  --labels "ZS phase-1 old (azim 0)" "ZS phase-1 new (azim 135)" | tee demo/shic/bear3_ZS_sdsquare_fuse16_phase1new/proportion_extent_old_vs_new.txt
echo "CHAIN DONE"
