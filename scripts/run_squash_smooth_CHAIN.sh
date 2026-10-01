#!/bin/bash
# WHY IS SQUASH SMOOTH -- starting from the MAIN baseline (Tal, 2026-09-15). SMALL analysis per arm,
# each compared against MAIN and the PROTECTED smooth squash run (compare_vs_main_and_squash.sh).
#   arm 1: MAIN, photo features sd_square -> squash (bug B back in), train T = read T = 0.01
#   arm 2: MAIN, SD+DINO -> DINO only, train T = read T = 0.01
#          needs the isobbox DINO-only phase-1 + DINO-only photo features, both deleted in an earlier
#          cleanup -> rebuilt here with the 2026-09-02 recipe (run_isobbox_dinoonly_phase1_phase2.sh).
#          width/n_classes 1024 instead of 2304 (forced by the feature dim -- capacity rides along).
# Everything else = MAIN: 16 in-segment neighbours, learned per-segment mu, truediag2d, cycle v2
# T_cyc=0.05 c=1.676e-5 return stride 5 (c NOT recalibrated for T=0.01).
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh
export SMALL=1 READ_T=0.01
EXP=./experiments/guitar
CYC="--cyc_weight 1.676e-05 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5"
REC="-mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0"

train () {  # <out_dir> <phase1_dir> <pix_features> <width>
  if [ -f "$1/pred_f.pth" ]; then echo "already trained: $1"; return; fi
  time python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" --init_encoder_dir "$2" \
    --object_mask_path "$MASK" --fine_seg_path "$FINE" --pix_dino_features_path "$3" \
    --width $4 --n_classes $4 --geodesic --temperature 0.01 --train_seed 0 \
    --geo_diag_2d 1175.75 --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 \
    --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
    --mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs \
    $CYC --out_dir "$1"
}

echo "##################### ARM 1: squash + T=0.01 #####################"
S1=sdsquash_fuse16_truediag2d_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0
D1=$EXP/encoder_isobbox_${S1}_phase2
R1=bsb_isobbox_sdsquash_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0_phase2
train "$D1" "$PHASE1" "$PIX_sdsquash_fuse16" 2304
./scripts/analyze_isobbox_arm.sh "$R1" "$D1/pred_f.pth" "$PIX_sdsquash_fuse16" "isobbox_${S1}"
./scripts/compare_vs_main_and_squash.sh "$R1" "ARM 1: MAIN + squash + T=0.01, read T=0.01" "isobbox_${S1}"

echo "##################### ARM 2: DINO only + T=0.01 #####################"
DATA=./data/guitar/encoder_data_frontcentered_dinosd_isobbox
CACHE=dino_only_isobbox
P1D=$EXP/encoder_isobbox_dino_only_stride14_dfd_crossview_phase1
FEATD=demo/guitar/features_analyze/dino_only_stride14
PIXD=$FEATD/guitar_01_pix_dino_only_stride14.pth
if [ ! -f "$P1D/pred_f.pth" ]; then
  echo "=== DINO-only precompute (1000 views) ==="
  time python train_encoder_dino_sd.py --obj_path "$OBJ" --name "$NAME" \
    --encoder_data_dir $DATA --encoder_model_dir $P1D \
    --view_sampling random --frontview_center 0 1.5707963267948966 --frontview_std 4 \
    --n_views 1000 --width 1024 --n_classes 1024 \
    --feature_cache_name $CACHE --dino_only \
    --generate_random_views 0 --encode_random_views 1 --start_training 0
  echo "=== DINO-only phase-1 ==="
  time python train_encoder_dfd_crossview.py --obj_path "$OBJ" --name "$NAME" \
    --encoder_data_dir $DATA --encoder_model_dir $P1D \
    --feature_model_type $CACHE --n_views 1000 --width 1024 --n_classes 1024
fi
if [ ! -f "$PIXD" ]; then
  echo "=== DINO-only photo features, stride 14 ==="
  mkdir -p "$FEATD"
  time python analysis/extract_dino_sd_features.py --img_path images/guitar_01.png \
    --save_dir "$FEATD" --output_path "$PIXD" \
    --save_raw_path "$FEATD/guitar_01_pix_dino_only_stride14_RAW_pregrid.pth" \
    --dino_only --cpu_postprocess
fi
S2=dinoonly_truediag2d_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0
D2=$EXP/encoder_isobbox_${S2}_phase2
R2=bsb_isobbox_dinoonly_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0_phase2
train "$D2" "$P1D" "$PIXD" 1024
./scripts/analyze_isobbox_arm.sh "$R2" "$D2/pred_f.pth" "$PIXD" "isobbox_${S2}"
./scripts/compare_vs_main_and_squash.sh "$R2" "ARM 2: MAIN + DINO only + T=0.01, read T=0.01" "isobbox_${S2}"
echo "===== SQUASH SMOOTH CHAIN DONE ====="
