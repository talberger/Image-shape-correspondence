#!/bin/bash
# The two comparison sets Tal asked for, 2026-08-31 -- THREE-ARM version (T=1, 0.1, 0.01).
# T=0.001 was deliberately left for the next pod (Tal, 2026-08-31); its training and eval ARE done,
# only its analysis rendering is missing. Re-run the four-arm version once that is filled in.
#   (A) each no-sd_square arm vs T=1 from the SAME sweep -- isolates temperature, everything else
#       identical (same phase-1, same photo features, same seed, same GPU).
#   (B) each no-sd_square arm vs the run Tal likes, dfd_crossview T=0.01 (2026-08-27, coverage
#       1162/4036 = 28.79%). NOTE that run differs in THREE ways, only two of which this sweep
#       matches: framing (no sd_square -- now matched), read temperature (0.01 -- matched by the
#       analyze_readT pass), and STRIDE (it is stride 14 with no gridalign -- NOT matched). If the
#       preference survives (A) and (B) it points at stride/smoothness, which is what the
#       2026-08-30 effective-rank numbers already suggested.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
L=/workspace/logs
D=demo/guitar

# Wait on a SENTINEL, not on pgrep: a pgrep -f pattern broad enough to match the analysis
# driver also matches any shell command that merely mentions it, which stalled this same
# wait earlier today. _stop_after_T0.01.sh writes STOPPED when the T=0.01 arm is done.
until grep -q STOPPED /workspace/logs/stop_after_T0.01.log 2>/dev/null; do sleep 15; done
echo "===== analysis finished, building comparisons ====="

LIKED=$D/bsb_sd_dino_dfd_crossview_relational_geodesic_mu3dscale1_T0.01_phase2
LIKED_COV="1162/4036 = 28.79%"
T1DIR=$D/bsb_sd_dino_stride7_gridalign_nosdsquare_T1_seed0_phase2

cov () {  # coverage of a run's soft-snap correspondence, computed from the file itself
  python - "$1" <<'PY'
import sys, torch, glob, os
p = glob.glob(os.path.join(sys.argv[1], 'analyze', 'color_mesh_from_soft_mapping',
                           '*_soft_snap_correspondence.pth'))
if not p:
    print('n/a'); raise SystemExit
c = torch.load(p[0], map_location='cpu')
n = int(c['num_vertices']); k = int(torch.unique(c['top1_vertex_idx']).numel())
print(f'{k}/{n} = {100*k/n:.2f}%')
PY
}

T1COV=$(cov "$T1DIR")
echo "T=1 (no sd_square) coverage: $T1COV"

for T in 1 0.1 0.01; do
  RUN=$D/bsb_sd_dino_stride7_gridalign_nosdsquare_T${T}_seed0_phase2
  RCOV=$(cov "$RUN")
  echo "===== T=$T coverage: $RCOV ====="

  # (A) vs T=1 of this same sweep -- skip for T=1 itself
  if [ "$T" != "1" ]; then
    python analysis/make_run_vs_baseline_comparison.py \
      --run_label "no sd_square T=$T (seed0)" \
      --run_phase2_dir "$RUN" --run_phase2_coverage "$RCOV" \
      --baseline_label "no sd_square T=1 (seed0)" \
      --baseline_phase2_dir "$T1DIR" --baseline_phase2_coverage "$T1COV" \
      --phase2_out_dir "$RUN/comparison_vs_T1" > $L/nosdsquare_cmp_T${T}_vs_T1.log 2>&1
    echo "  (A) -> $RUN/comparison_vs_T1"
  fi

  # (B) vs the run Tal likes
  python analysis/make_run_vs_baseline_comparison.py \
    --run_label "no sd_square stride7 T=$T (seed0)" \
    --run_phase2_dir "$RUN" --run_phase2_coverage "$RCOV" \
    --baseline_label "LIKED: dfd_crossview stride14 T=0.01 (read T=0.01)" \
    --baseline_phase2_dir "$LIKED" --baseline_phase2_coverage "$LIKED_COV" \
    --phase2_out_dir "$RUN/comparison_vs_liked_T0.01" > $L/nosdsquare_cmp_T${T}_vs_liked.log 2>&1
  echo "  (B) -> $RUN/comparison_vs_liked_T0.01"
done

echo "===== COMPARISONS DONE ====="
