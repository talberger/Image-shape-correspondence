#!/bin/bash
# The common-grid experiment, 2026-08-30, Tal's design.
#   arm B = common-grid fuse, SD still zero-padded   -> isolates "stop downsampling SD"
#   arm C = common-grid fuse, SD unpadded true aspect -> adds "remove the padding"
# Both at stride 14 and T=0.01, from ONE shared phase-1 (renders are square, so the two framings
# are identical there). Control already on disk: bsb_sd_dino_dfd_crossview_sdaligned_T0.01_seed0.
# Sequential -- one GPU job at a time (the 2026-08-29 OOM rule).
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs; EL=demo/guitar/eval_labels
# NOT `rm -f $L/commongrid_*.log` -- that glob matches this driver's OWN output file and
# deleted it mid-run on 2026-08-30. Delete the per-stage logs BY NAME.
rm -f $L/commongrid_phase1.log $L/commongrid_photo_features.log \
      $L/commongrid_padded_*.log $L/commongrid_nopad_*.log $L/commongrid_evalcmp.log

echo "===== [1/8] SHARED phase-1 (precompute + DFD cross-view) ====="
./run_commongrid_phase1.sh > $L/commongrid_phase1.log 2>&1

echo "===== [2/8] photo features for both arms + overlay agreement check ====="
./run_commongrid_photo_features.sh > $L/commongrid_photo_features.log 2>&1
grep -E "centroid offset|IoU|dino grid" $L/commongrid_photo_features.log || true

for N in padded nopad; do
  case $N in
    padded) FP=dino_sd_commongrid_padded/guitar_01_pix_commongrid_padded.pth; ARM=B ;;
    nopad)  FP=dino_sd_commongrid_nopad/guitar_01_pix_commongrid_nopad.pth;   ARM=C ;;
  esac
  D=$DISC_ROOT/experiments/guitar/encoder_dino_sd_commongrid_${N}_T0.01_seed0_phase2

  echo "===== [arm $ARM / $N] phase-2 ====="
  ./run_commongrid_${N}_T0.01_phase2_train.sh > $L/commongrid_${N}_train.log 2>&1

  echo "===== [arm $ARM / $N] eval ====="
  ( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
    python analysis/eval_geodesic_error.py \
      --run_label "common-grid $N, T=0.01 seed0" \
      --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth \
      --features_path demo/guitar/features_analyze/$FP \
      --vertex_features_path $D/pred_f.pth --k_vertices_per_pixel 10 \
      --save_path $EL/eval_commongrid_${N}_T0.01.pth \
      --plot_path $EL/eval_commongrid_${N}_T0.01.png ) > $L/commongrid_${N}_eval.log 2>&1

  echo "===== [arm $ARM / $N] step progression ====="
  ./run_commongrid_${N}_T0.01_analysis.sh > $L/commongrid_${N}_analysis.log 2>&1

  echo "===== [arm $ARM / $N] full analyze ====="
  ./run_commongrid_${N}_T0.01_full_analyze.sh > $L/commongrid_${N}_full_analyze.log 2>&1
  COV=$(grep -E "vertex coverage" $L/commongrid_${N}_full_analyze.log | tail -1 \
        | sed -E "s#.*coverage: ([0-9]+/4036) vertices \(([0-9.]+%)\).*#\1 = \2#")

  echo "===== [arm $ARM / $N] compare vs the current sd_square control (coverage $COV) ====="
  ( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
    python analysis/make_run_vs_baseline_comparison.py \
      --run_label "arm $ARM: common-grid fuse, SD $N, T=0.01" \
      --run_phase2_dir demo/guitar/runs/legacy/bsb_sd_dino_commongrid_${N}_T0.01_seed0_phase2 \
      --run_phase2_coverage "$COV" \
      --baseline_label "control: sd_square + dino-grid fuse, T=0.01" \
      --baseline_phase2_dir demo/guitar/runs/legacy/bsb_sd_dino_dfd_crossview_sdaligned_T0.01_seed0_phase2 \
      --baseline_phase2_coverage "1232/4036 = 30.53%" \
      --phase2_out_dir demo/guitar/runs/legacy/bsb_sd_dino_commongrid_${N}_T0.01_seed0_phase2/comparison_vs_control \
    ) > $L/commongrid_${N}_compare.log 2>&1
done

echo "===== [8/8] 4-way eval graph ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_T0.01_phase2.pth $EL/eval_sdaligned_T0.01_seed0.pth \
           $EL/eval_commongrid_padded_T0.01.pth $EL/eval_commongrid_nopad_T0.01.pth \
    --labels "no sd_square (old)" "sd_square + dino-grid fuse" \
             "arm B: common-grid, padded" "arm C: common-grid, no padding" \
    --save_path $EL/eval_compare_commongrid_T0.01.png ) > $L/commongrid_evalcmp.log 2>&1

echo "===== COMMON-GRID BOTH ARMS DONE ====="
