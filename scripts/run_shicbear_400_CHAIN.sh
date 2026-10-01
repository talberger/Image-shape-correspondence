#!/bin/bash
# bear3 at long_side 400 instead of 808 (2026-09-20, Tal's call) -- the resolution experiment.
# Rationale: DINO resizes its input to 224, SD to 512, SAM to 1024, so feeding them an 800px image
# adds no information; it only inflates the mask pixel count, which is what the expensive 2D
# geodesic Dijkstra scales with. Three hyperparameters are pixel-count-dependent and are scaled with
# it here so they keep meaning the same thing:
#   min_pixels_in_object  200 -> 50   (an AREA threshold; area drops 4x when the long side halves)
#   cyc_return_stride       5 -> 2    (a LINEAR spacing; exact ratio would be 2.5, 2 is the denser side)
#   num_center_pixels    2048 -> 2048 (left alone, Tal's call)
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb

OBJ=./demo/shic/shapes/bear.obj; NAME=shic_bear
P1=./experiments/shic_bear/encoder_dfd_crossview_phase1
GEO3D=./experiments/shic_bear/geodesic_tables/shic_bear_geo_table_3d.pth
D=demo/shic/bear3_400
IMG=$D/bear3_400.png; MASK=$D/bear3_400_mask.pth; MASKBASE=bear3_400_mask
# build_fine_segments.py names its output from the IMAGE basename, not the mask's.
FSEG=$D/bear3_400_fine_segments.pth
S=res400_sdsquare_truediag2d_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride2_loraW_all_r4_lr1e-3_T0.01_seed0
P2=./experiments/shic_bear/encoder_${S}_phase2
RUN=demo/shic/bear3_400_${S}_phase2
t() { local s=$(date +%s); "$@"; echo "[time] $(( $(date +%s) - s )) s : $1 $2"; }

[ -f "$FSEG" ] || t python analysis/build_fine_segments.py --img_path $IMG --mask_path $MASK --save_dir $D \
    --min_pixels_in_object 50
t python analysis/build_geodesic_tables.py --obj_path $OBJ --name $NAME --skip_3d \
    --object_mask_path $MASK --fine_seg_path $FSEG \
    --num_center_pixels 2048 --seed 0 --true_diameter --out_dir $D 2>&1 | tee $D/geo_build.log
DIAG=$(awk '/TRUE 2D geodesic diameter/ {print $(NF-1)}' $D/geo_build.log)
[ -n "$DIAG" ] || { echo "FAILED to read geo_diag_2d" >&2; exit 1; }
echo "geo_diag_2d = $DIAG"
t python analysis/extract_dino_sd_features.py --img_path $IMG --save_dir $D \
    --output_path $D/bear3_400_pix_sdsquare_fuse16.pth --sd_frame square_pad --cpu_postprocess
t python analysis/pixel_coloring.py --img_path $IMG --mask_path $MASK --mode position --save_dir $D \
    --output_path $D/${MASKBASE}_position_color.png

t python phase2/train_relational_loss_l2norm.py --obj_path $OBJ --name $NAME --init_encoder_dir $P1 \
    --object_mask_path $MASK --fine_seg_path $FSEG \
    --pix_dino_features_path $D/bear3_400_pix_sdsquare_fuse16.pth \
    --width 2304 --n_classes 2304 --geodesic --temperature 0.01 --train_seed 0 --geo_diag_2d $DIAG \
    --geo_table_3d_path $GEO3D --geo_table_2d_path $D/${MASKBASE}_geo_table_2d_seed0.pth \
    --mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs \
    --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 \
    --cyc_weight 1.676e-05 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 2 \
    --lora_w_rank 4 --lora_w_layers all --lora_w_lr 1e-3 --no_step_checkpoints --out_dir $P2

t python analysis/export_pred_f_lora_alpha.py --obj_path $OBJ --name $NAME \
    --phase1_checkpoint $P1/encoder_checkpoint.pth --lora_checkpoint $P2/encoder_checkpoint.pth \
    --alphas 0.5 --out_dir $P2/alpha

POSCOLOR=$D/${MASKBASE}_position_color.png
MASK=$MASK IMG=$IMG MASKBASE=$MASKBASE POSCOLOR=$POSCOLOR \
  P2=$P2/alpha/alpha0.5 RUN=$RUN/alpha0.5 PIX=$D/bear3_400_pix_sdsquare_fuse16.pth \
  READ_T=0.01 LABEL="res400 LORA-W all r=4, alpha 0.5" bash scripts/analyze_shicbear_arm.sh
echo "BEAR3-400 CHAIN DONE: $RUN"
