#!/bin/bash
# GUITAR on the CURRENT LVIS recipe (Tal, 2026-10-10): two-stage mu (global ellipse 8 in + 8 out, then per-segment 16 in, switch 150)
# + LORA-W all r4 read at alpha 0.5, with the fine segments rebuilt as on LVIS since 10-07: merge 1.5% + unlabelled pixels joined
# (MERGE / FILLSEG). The guitar keeps its own inputs (SAM click mask, features, mesh, phase 1, cycle return stride 5); no amodal
# (the mask is in one piece). The end of stage 1 is saved and ZS / stage 1 / final are scored on eval v2.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

MERGE=${MERGE:-0.015}; FILLSEG=${FILLSEG:-1}; SWITCH=${SWITCH:-150}; MU_LR=5e-2; T=0.01; ALPHA=0.5
# MU=aniso (2026-10-10): NO stage 1 -- per-segment directional ellipses from step 1, 16 in-segment neighbours (same segments)
MU=${MU:-globalthenaniso}
TAG=merge${MERGE}; [ "$FILLSEG" = 1 ] && TAG=${TAG}_fillseg
SEGDIR=demo/guitar/segment_masks_pixel270_370/$TAG
GEODIR=experiments/guitar/geodesic_tables/$TAG
FINE=$SEGDIR/guitar_01_fine_segments.pth
GEO2D=$GEODIR/${MB}_geo_table_2d_seed0.pth
if [ "$MU" = aniso ]; then
  MUARGS="--num_neighbors_in_segment 16 --num_neighbors_out_segment 0 --mu_scope segment_aniso"; MUTAG=in16out0
else
  MUARGS="--num_neighbors_in_segment 8 --num_neighbors_out_segment 8 --mu_scope global_then_segment_aniso --mu_switch_step $SWITCH --switch_nin 16 --switch_nout 0 --save_stageA_end"
  MUTAG=in8out8_sw${SWITCH}
fi
S=isobbox_sdsquare_fuse16_truediag2d_mu_${MU}_perseg_cycv2T0.05_c1.676e-05_retstride5_loraW_all_r4_lr1e-3_T${T}_seed0_mulr${MU_LR}_${MUTAG}_${TAG}
D=./experiments/guitar/encoder_${S}_phase2
DEMO=$PROP_RUNS/bsb_isobbox_sdsquare_mu_${MU}_${MUTAG#in8out8_}_${TAG}_a${ALPHA}_phase2
mkdir -p "$SEGDIR" "$GEODIR" "$DEMO"

if [ ! -f "$FINE" ]; then
  python analysis/build_fine_segments.py --img_path images/guitar_01.png --mask_path "$MASK" --save_dir "$SEGDIR" \
    --merge_below_frac $MERGE $([ "$FILLSEG" = 1 ] && echo --fill_unlabelled)
fi
if [ ! -f "$GEO2D" ]; then
  python analysis/build_geodesic_tables.py --obj_path "$OBJ" --name "$NAME" --skip_3d --object_mask_path "$MASK" \
    --fine_seg_path "$FINE" --num_center_pixels 2048 --seed 0 --true_diameter --out_dir "$GEODIR"
fi
DIAG=$(python -c "import json;print(json.load(open('$GEODIR/${MB}_geo_table_2d_seed0_meta.json'))['true_diameter_2d'])")
echo "geo_diag_2d = $DIAG"

if [ -f "$D/pred_f.pth" ]; then echo "already trained: $D"; else
  time python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" --init_encoder_dir "$PHASE1" \
    --object_mask_path "$MASK" --fine_seg_path "$FINE" --pix_dino_features_path "$PIX_sdsquare_fuse16" \
    --width 2304 --n_classes 2304 --geodesic --temperature $T --train_seed 0 \
    --geo_diag_2d $DIAG \
    --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
    --mu_mode learned $MUARGS \
    --mu_lr $MU_LR --mu_fit_space abs \
    --cyc_weight 1.676e-05 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5 \
    --lora_w_rank 4 --lora_w_layers all --lora_w_lr 1e-3 \
    --no_step_checkpoints --out_dir "$D"
fi
python analysis/eval_guitar_stage_ladder.py --run_dir "$D" --phase1_dir "$PHASE1" --obj_path "$OBJ" --name "$NAME" \
  --features_path "$PIX_sdsquare_fuse16" --eval_pairs_path "$EVALPAIRS" --geo_table_path "$GEO3D" \
  --alpha $ALPHA --temperature $T --out_dir "$DEMO/eval"
echo "===== GUITAR TWO-STAGE CURRENT DONE: $DEMO ====="
