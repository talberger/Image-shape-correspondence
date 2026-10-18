#!/bin/bash
# The stride-7 + gridalign T sweep on the NO-sd_square (squashed SD) photo features, all seed 0.
# 2026-08-31, Tal's call: sd_square comes OFF the baseline for now -- across the 2026-08-30
# investigation the back-projection he judged best is still the pre-sd_square one, and neither
# proposed explanation for that survived testing.
# Four arms: T = 1, 0.1, 0.01, 0.001. Sequential, one GPU job at a time (the 2026-08-29 OOM rule).
# NOTE this pod runs on an RTX 2000 Ada; the sd_square arms it will be compared against were
# trained on an A4500, so cross-framing gaps carry a hardware confound on top of the 1.6 AUC
# seed-noise floor. Within this sweep all four arms share this GPU and this seed.
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs
E=$DISC_ROOT/experiments/guitar
EL=demo/guitar/eval_labels
PIX=demo/guitar/features_analyze/dino_sd_stride7_gridalign_nosdsquare/guitar_01_pix_stride7_gridalign_nosdsquare.pth

for T in 1 0.1 0.01 0.001; do
  D=$E/encoder_dino_sd_stride7_gridalign_nosdsquare_relational_geodesic_mu3dscale1_T${T}_seed0_phase2
  rm -f $L/nosdsquare_T${T}_seed0_train.log $L/nosdsquare_T${T}_seed0_eval.log
  echo "===== [train] T=$T seed 0, no sd_square ====="
  ./run_stride7_gridalign_nosdsquare_T${T}_seed0_phase2_train.sh > $L/nosdsquare_T${T}_seed0_train.log 2>&1

  echo "===== [eval] T=$T seed 0, no sd_square ====="
  ( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
    python analysis/eval_geodesic_error.py \
      --run_label "stride7 gridalign no-sd_square T=$T seed0" \
      --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth \
      --features_path $PIX \
      --vertex_features_path $D/pred_f.pth \
      --k_vertices_per_pixel 10 \
      --save_path $EL/eval_stride7_gridalign_nosdsquare_T${T}_seed0.pth \
      --plot_path $EL/eval_stride7_gridalign_nosdsquare_T${T}_seed0.png ) > $L/nosdsquare_T${T}_seed0_eval.log 2>&1
  du -sh $D
done

echo "===== [compare] 4-way T sweep, no sd_square ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_stride7_gridalign_nosdsquare_T1_seed0.pth \
           $EL/eval_stride7_gridalign_nosdsquare_T0.1_seed0.pth \
           $EL/eval_stride7_gridalign_nosdsquare_T0.01_seed0.pth \
           $EL/eval_stride7_gridalign_nosdsquare_T0.001_seed0.pth \
    --labels "T=1" "T=0.1" "T=0.01" "T=0.001" \
    --save_path $EL/eval_compare_stride7_gridalign_nosdsquare_T_sweep.png ) > $L/nosdsquare_T_sweep_compare.log 2>&1

echo "===== [compare] sd_square A/B at matched T, all seed 0 ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  for T in 1 0.1 0.01 0.001; do
    python analysis/compare_eval_runs.py \
      --runs $EL/eval_stride7_gridalign_T${T}_seed0.pth \
             $EL/eval_stride7_gridalign_nosdsquare_T${T}_seed0.pth \
      --labels "sd_square T=$T" "no sd_square T=$T" \
      --save_path $EL/eval_compare_sd_square_ab_stride7_T${T}.png
  done ) > $L/nosdsquare_sd_square_ab_compare.log 2>&1

echo "SWEEP DONE"
