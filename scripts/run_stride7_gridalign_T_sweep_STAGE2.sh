#!/bin/bash
# Stage 2 of the seeded T sweep (2026-08-30, Tal's request): for every arm, the FULL analysis
# battery on top of the step progression, plus a comparison against the NEW seeded baseline
# (T=1 --train_seed 0) rather than the old unseeded one.
#
# Arms and what each still needs:
#   T=1     seed0 -- step progression comes from the V2 driver; needs full_analyze. It IS the
#                    baseline, so it gets no comparison against itself.
#   T=0.01  seed0 -- step progression + full_analyze + comparison vs new baseline
#   T=0.001 seed0 -- step progression + full_analyze + comparison vs new baseline
#   T=0.1         -- step progression + full_analyze already exist (2026-08-29); it only needs the
#                    comparison re-made against the NEW baseline (its old one used the old baseline).
#
# Sequential -- one GPU job at a time (the 2026-08-29 OOM rule: the analysis battery allocates a
# [chunk, V] tensor in color_mesh_from_soft_mapping and needs most of the 20 GB card).
set -e
cd /workspace/bsb/scripts
L=/workspace/logs
BASE_DEMO=demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_gridalign_T1_seed0_phase2
BASE_LABEL="s7 + gridalign, T=1, train_seed 0 -- NEW SEEDED BASELINE"

cov_of () {  # pull "N/4036 = P%" out of a full_analyze log
  grep -E "vertex coverage" "$1" | tail -1 \
    | sed -E "s#.*coverage: ([0-9]+/4036) vertices \(([0-9.]+%)\).*#\1 = \2#"
}

# ---- 1. T=1 seed0: full analysis (step progression already done by the V2 driver) ----
echo "===== [full_analyze] T=1 seed0 (the new baseline) ====="
rm -f $L/gridalign_T1_seed0_full_analyze.log
./run_stride7_gridalign_T1_seed0_full_analyze.sh > $L/gridalign_T1_seed0_full_analyze.log 2>&1
BASE_COV=$(cov_of $L/gridalign_T1_seed0_full_analyze.log)
echo "      new baseline phase-2 coverage: $BASE_COV"

# ---- 2. T=0.01 and T=0.001 seed0: step progression, full analysis, comparison vs new baseline ----
for T in 0.01 0.001; do
  echo "===== [step progression] T=$T seed0 ====="
  rm -f $L/gridalign_T${T}_seed0_analysis.log
  ./run_stride7_gridalign_T${T}_seed0_analysis.sh > $L/gridalign_T${T}_seed0_analysis.log 2>&1

  echo "===== [full_analyze] T=$T seed0 ====="
  rm -f $L/gridalign_T${T}_seed0_full_analyze.log
  ./run_stride7_gridalign_T${T}_seed0_full_analyze.sh > $L/gridalign_T${T}_seed0_full_analyze.log 2>&1
  COV=$(cov_of $L/gridalign_T${T}_seed0_full_analyze.log)

  echo "===== [compare vs NEW baseline] T=$T seed0  (coverage $COV) ====="
  rm -f $L/gridalign_T${T}_seed0_compare.log
  ( source /workspace/activate_bsb.sh; cd /workspace/bsb
    python analysis/make_run_vs_baseline_comparison.py \
      --run_label "s7 + gridalign, T=$T, train_seed 0" \
      --run_phase2_dir demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_gridalign_T${T}_seed0_phase2 \
      --run_phase2_coverage "$COV" \
      --baseline_label "$BASE_LABEL" \
      --baseline_phase2_dir $BASE_DEMO \
      --baseline_phase2_coverage "$BASE_COV" ) > $L/gridalign_T${T}_seed0_compare.log 2>&1
done

# ---- 3. T=0.1: analysis already exists from 2026-08-29; only redo the comparison vs the NEW baseline ----
echo "===== [compare vs NEW baseline] T=0.1 (reusing its 2026-08-29 analysis) ====="
rm -f $L/gridalign_T0.1_vs_newbaseline_compare.log
( source /workspace/activate_bsb.sh; cd /workspace/bsb
  python analysis/make_run_vs_baseline_comparison.py \
    --run_label "s7 + gridalign, T=0.1, train_seed 0" \
    --run_phase2_dir demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_gridalign_T0.1_seed0_phase2 \
    --run_phase2_coverage "1180/4036 = 29.24%" \
    --baseline_label "$BASE_LABEL" \
    --baseline_phase2_dir $BASE_DEMO \
    --baseline_phase2_coverage "$BASE_COV" ) > $L/gridalign_T0.1_vs_newbaseline_compare.log 2>&1

echo "===== STAGE2 ALL DONE ====="
