#!/bin/bash
# WHY IS SQUASH SMOOTH, arm 3 (Tal, 2026-09-15): arm 1 (MAIN + squash + T=0.01) + the OLD mesh.
# Old mesh = guitar.obj 4036v + its own phase-1 (encoder_dino_sd_dfd_crossview_phase1, the one the smooth
# squash run started from) + old 3D geodesic table + old 2D diameter (no --geo_diag_2d).
# Kept from MAIN: 16 in-segment neighbours, learned per-segment mu, cycle v2 c=1.676e-5 (not recalibrated).
# SMALL analysis, read T=0.01, no eval (no v2 labels on the old mesh). Waits for the arm 1/2 chain first.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh
export OLDMESH=1 SMALL=1 READ_T=0.01
source scripts/lib_oldmesh_override.sh

until grep -qE "SQUASH SMOOTH CHAIN DONE" /workspace/logs/squash_smooth_chain.log || ! pgrep -f run_squash_smooth_CHAIN.sh >/dev/null; do sleep 30; done
echo "chain finished, starting arm 3"

S3=oldmesh_sdsquash_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0
D3=./experiments/guitar/encoder_${S3}_phase2
R3=bsb_${S3}_phase2
if [ -f "$D3/pred_f.pth" ]; then echo "already trained: $D3"; else
  time python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" --init_encoder_dir "$PHASE1" \
    --object_mask_path "$MASK" --fine_seg_path "$FINE" --pix_dino_features_path "$PIX_sdsquash_fuse16" \
    --width 2304 --n_classes 2304 --geodesic --temperature 0.01 --train_seed 0 \
    --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 \
    --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
    --mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs \
    --cyc_weight 1.676e-05 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5 \
    --out_dir "$D3"
fi
./scripts/analyze_isobbox_arm.sh "$R3" "$D3/pred_f.pth" "$PIX_sdsquash_fuse16"
ARM_OLDMESH=1 ./scripts/compare_vs_main_and_squash.sh "$R3" "ARM 3: MAIN + squash + T=0.01 + OLD MESH, read T=0.01" "none"
echo "===== ARM 3 DONE ====="
