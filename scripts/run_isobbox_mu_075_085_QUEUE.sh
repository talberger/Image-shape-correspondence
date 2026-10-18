#!/bin/bash
# mu_3d_scale = 0.75 / 0.80 / 0.85 ON THE NEW BASELINE (sdsquare_fuse16, T=1, seed0,
# geo_diag_2d = 1175.75). 2026-09-08, Tal.
#
# The earlier near-1 sweep (0.99/0.95/0.9, 2026-09-03) ran on SQUASH framing and a much worse ZS
# (68.8 vs 87.9), and Tal rejected all three on the visual gate while flagging 0.75 as the next
# value to try. This is that follow-up, on the current recipe, extended to bracket it.
# mu_3d_scale divides norm_d_3D AFTER normalization, so <1 asks neighbouring pixels to map FURTHER
# apart on the mesh. 0.5 was catastrophic (51.4 AUC, coverage 13.7%); 0.9 was fine (75.5). This
# fills the gap between them.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

for MU in 0.75 0.8 0.85; do
  P2DIR=./experiments/guitar/encoder_isobbox_sdsquare_fuse16_truediag2d_mu${MU}_T1_seed0_phase2
  RUN=bsb_isobbox_sdsquare_mu${MU}_T1_seed0_phase2
  TAG=isobbox_sdsquare_fuse16_truediag2d_mu${MU}_T1_seed0

  echo "########## mu_3d_scale=$MU : [1/3] phase-2 ##########"
  if [ -f "$P2DIR/pred_f.pth" ]; then echo "already trained"; else
    time python phase2/train_relational_loss_l2norm.py \
      --obj_path "$OBJ" --name "$NAME" \
      --init_encoder_dir "$PHASE1" \
      --object_mask_path "$MASK" \
      --fine_seg_path "$FINE" \
      --pix_dino_features_path "$PIX_sdsquare_fuse16" \
      --width 2304 --n_classes 2304 --geodesic --mu_3d_scale $MU \
      --temperature 1.0 --train_seed 0 --geo_diag_2d 1175.75 \
      --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
      --out_dir "$P2DIR"
  fi

  echo "########## mu_3d_scale=$MU : [2/3] analysis + eval ##########"
  ./scripts/analyze_isobbox_arm.sh "$RUN" "$P2DIR/pred_f.pth" "$PIX_sdsquare_fuse16" "$TAG"

  echo "########## mu_3d_scale=$MU : [3/3] compare vs baseline ##########"
  ./scripts/compare_isobbox_arms.sh \
    "$RUN" "mu_3d_scale = $MU" \
    bsb_isobbox_sdsquare_T1_seed0_phase2 "BASELINE, mu_3d_scale = 1" \
    comparison_vs_baseline \
    "$TAG" isobbox_sdsquare_fuse16_truediag2d_T1_seed0
done

echo "===== MU 0.75/0.8/0.85 QUEUE DONE ====="
