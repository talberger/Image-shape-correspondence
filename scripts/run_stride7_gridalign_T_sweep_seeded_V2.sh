#!/bin/bash
# The T sweep, re-run SEEDED so all four arms share --train_seed 0 (2026-08-30, Tal's request).
# T=0.1 already exists seeded from 2026-08-29 and is reused as-is; T=1 / 0.01 / 0.001 are run here.
#
# V2, and why: V1 pruned each run's 12 step checkpoints to save disk. Tal then asked to review the
# new seeded baseline VISUALLY, which needs those checkpoints (build_step_grid consumes
# pred_f_step*.pth). Patching the running V1 did NOT work -- bash parses an entire `for` compound
# command into memory before executing it, so an in-flight edit to the loop body has no effect
# (byte-offset preservation only helps for top-level sequential commands). Hence a clean relaunch
# with NO prune. Disk: 3 runs x 5.9 GB = 18 GB, 272 -> ~290 GB against the 350 GB ceiling.
#
# T=1 is re-trained even though its eval already ran, purely to regenerate the deleted step
# checkpoints; being seeded it must reproduce bit-identically, which is checked below.
# Sequential -- one GPU job at a time (the 2026-08-29 OOM rule).
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
  du -sh $D
done

echo "===== [check] does the seeded T=1 re-run reproduce the first run bit-identically? ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python - <<'PY'
import torch
a = torch.load('/tmp/claude-0/-workspace/feba119c-8a68-4f63-bdb4-de0a599d1fd3/scratchpad/pred_f_T1_seed0_run1.pth', map_location='cpu')
b = torch.load('experiments/guitar/encoder_dino_sd_stride7_gridalign_relational_geodesic_mu3dscale1_T1_seed0_phase2/pred_f.pth', map_location='cpu')
print('max abs diff run1 vs run2:', (a - b).abs().max().item())
print('bit-identical:', bool(torch.equal(a, b)))
PY
) 2>&1 | tail -3

echo "===== [compare] 4-way, all --train_seed 0 ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_stride7_gridalign_T1_seed0.pth $EL/eval_stride7_gridalign_T0.1_seed0.pth \
           $EL/eval_stride7_gridalign_T0.01_seed0.pth $EL/eval_stride7_gridalign_T0.001_seed0.pth \
    --labels "T=1" "T=0.1" "T=0.01" "T=0.001" \
    --save_path $EL/eval_compare_stride7_T_sweep_seed0.png ) > $L/gridalign_T_sweep_seed0_compare.log 2>&1

echo "===== [step progression] T=1 seed0 -- the visual review Tal gated on ====="
rm -f $L/gridalign_T1_seed0_analysis.log
./run_stride7_gridalign_T1_seed0_analysis.sh > $L/gridalign_T1_seed0_analysis.log 2>&1

echo "===== T SWEEP SEEDED V2 ALL DONE ====="
