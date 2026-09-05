#!/bin/bash
# Eval of the argmax feature-RKD beta=0.05 arm against BOTH baselines (Tal, 2026-08-31):
# the promoted T=0.01 baseline and its protected T=1 companion.
# NEW CONVENTION (Tal, 2026-08-31): a new run's own evaluation output goes in that run's folder,
# not in the shared demo/guitar/eval_labels/ dump. Cross-run comparisons Tal asks for explicitly
# still go to eval_labels/.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb

EL=demo/guitar/eval_labels
OUT=demo/guitar/runs/rkd_feature/bsb_sd_dino_stride14_nosdsquare_featrkd_argmax_dist0.05_ang0_T0.01_seed0_phase2/eval
mkdir -p "$OUT"

python analysis/compare_eval_runs.py \
  --runs $EL/eval_featrkd_argmax_dist0.05_ang0.pth \
         $EL/eval_stride14_nosdsquare_T0.01_seed0.pth \
         $EL/eval_stride14_nosdsquare_T1_seed0.pth \
  --labels "featrkd argmax beta=0.05 (T=0.01)" \
           "BASELINE stride14 squash T=0.01 seed0" \
           "BASELINE companion, same recipe T=1" \
  --save_path "$OUT/eval_vs_both_baselines.png"

echo "SAVED: $OUT/eval_vs_both_baselines.png"
