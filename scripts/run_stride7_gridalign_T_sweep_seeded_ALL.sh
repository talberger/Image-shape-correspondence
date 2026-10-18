#!/bin/bash
# The T sweep, RE-RUN SEEDED so all four arms share --train_seed 0 (2026-08-30, Tal's request).
# T=0.1 already exists seeded from 2026-08-29 and is reused as-is; T=1 / 0.01 / 0.001 are re-run here.
# Sequential -- one GPU job at a time (the 2026-08-29 OOM rule).
# After each run, the 12 intermediate step checkpoints are deleted (~5.4 GB/run); the final
# encoder_checkpoint.pth, pred_f.pth, run_config.json and relational_loss.png are kept, which is
# everything the eval needs. Full loss/drift trace stays in the log.
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs
E=$DISC_ROOT/experiments/guitar
EL=demo/guitar/eval_labels

for T in 1 0.01 0.001; do
  D=$E/encoder_dino_sd_stride7_gridalign_relational_geodesic_mu3dscale1_T${T}_seed0_phase2
  rm -f $L/gridalign_T${T}_seed0_train.log $L/gridalign_T${T}_seed0_eval.log
  echo "===== [train] T=$T seed 0 ====="
  ./run_stride7_gridalign_T${T}_seed0_phase2_train.sh > $L/gridalign_T${T}_seed0_train.log 2>&1

  echo "===== [eval] T=$T seed 0 ====="
  ( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
    python analysis/eval_geodesic_error.py \
      --run_label "stride7 gridalign T=$T seed0" \
      --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth \
      --features_path demo/guitar/features_analyze/dino_sd_stride7_gridalign/guitar_01_pix_stride7_gridalign.pth \
      --vertex_features_path $D/pred_f.pth \
      --k_vertices_per_pixel 10 \
      --save_path $EL/eval_stride7_gridalign_T${T}_seed0.pth \
      --plot_path $EL/eval_stride7_gridalign_T${T}_seed0.png ) > $L/gridalign_T${T}_seed0_eval.log 2>&1

  echo "===== [prune] T=$T seed 0 ====="
  : # prune OFF 2026-08-30 keep step ckpts for review      
  du -sh $D
done

echo "===== [compare] 4-way, all --train_seed 0 ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_stride7_gridalign_T1_seed0.pth $EL/eval_stride7_gridalign_T0.1_seed0.pth \
           $EL/eval_stride7_gridalign_T0.01_seed0.pth $EL/eval_stride7_gridalign_T0.001_seed0.pth \
    --labels "T=1" "T=0.1" "T=0.01" "T=0.001" \
    --save_path $EL/eval_compare_stride7_T_sweep_seed0.png ) > $L/gridalign_T_sweep_seed0_compare.log 2>&1

echo "===== T SWEEP SEEDED ALL DONE ====="
