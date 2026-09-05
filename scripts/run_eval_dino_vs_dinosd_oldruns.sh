#!/bin/bash
# DINO-only vs DINO+SD on the n=100 ground-truth eval, USING ONLY RUNS AND FEATURE FILES THAT
# ALREADY EXIST ON DISK (Tal, 2026-09-01: "בעזרת ריצות ישנות" -- no new training, no re-extraction).
#
# WHY NOT THE 2026-08-20 PAIR, which is the comparison actually wanted: the photo features every
# DINO+SD run from 2026-08-20..2026-08-26 trained on
# (demo/guitar/features_analyze/dino_sd_poc/guitar_01_pix_dino_sd_features.pth, sha256 c089e1e0...)
# were deleted in an earlier disk cleanup. EIGHT phase-2 runs reference that one file. SD's noise
# was unseeded until 2026-08-28, so it cannot be regenerated bit-exactly -- Tal's call was that a
# regenerated substitute is not acceptable here, and this script therefore does not use one.
#
# THE ARMS -- both feature files verified sha256-identical to their own run_config.json before use:
#   A  encoder_dino_relational_geodesic_mu3dscale1_frontcenteredinit_newmask        (2026-08-17)
#      demo/guitar/bsb_origin/guitar_01_pix_dino_features.pth      08bce355... verified
#   B  encoder_dino_sd_dfd_crossview_sdaligned_relational_geodesic_mu3dscale1_phase2 (2026-08-27)
#      .../dino_sd_aligned/guitar_01_pix_dino_sd_features_aligned.pth  f24a95cd... verified
#
# BOTH ARE READ AT T=1, AND THAT IS MATCHED BY FACT, NOT BY DEFAULT: arm A predates the
# --temperature flag entirely, arm B recorded "temperature": 1.0. The sibling
# encoder_dino_sd_dfd_crossview_sdaligned_T0.01_seed0_phase2 is deliberately NOT used -- it would
# reintroduce the read-temperature confound the 2026-08-31 notes flagged.
#
# WHAT THE GAP DOES **NOT** MEASURE. Arm B is ten days downstream of arm A, and SD is only one of
# the things that changed in between. Do not read this as "the effect of SD":
#   1. DFD barycentric distillation with cross-view point batching (2026-08-22) -- arm A has none.
#   2. the sd_square SD/DINO framing fix (2026-08-27).
#   3. the L2-normalize fix, which landed with the SD recipe. There is no plain-DINO l2norm
#      phase-1 anywhere in experiments/guitar/, so SD and L2norm cannot be separated at all.
#   4. width/n_classes 1024 -> 2304, forced by the fused feature dimension. A capacity change.
#   5. BOTH runs predate phase-2 seeding (2026-08-29). Each is ONE draw from a distribution the
#      notes measure at up to 2.0 AUC wide -- a gap smaller than that is not a result.
# It is best read as "the recipe before SD vs the recipe after SD had bedded in", which is a real
# question, just a coarser one than the isolated ablation.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
OUT=demo/guitar/eval_labels
PAIRS=$OUT/pixel_vertex_eval_pairs.pth

echo "=== ARM A: DINO only (2026-08-17) ==="
python analysis/eval_geodesic_error.py \
  --eval_pairs_path "$PAIRS" \
  --features_path demo/guitar/bsb_origin/guitar_01_pix_dino_features.pth \
  --vertex_features_path experiments/guitar/encoder_dino_relational_geodesic_mu3dscale1_frontcenteredinit_newmask/pred_f.pth \
  --temperature 1.0 \
  --run_label "DINO only (2026-08-17)" \
  --plot_path "$OUT/eval_oldruns_dino_only.png" \
  --save_path "$OUT/eval_oldruns_dino_only.pth"

echo "=== ARM B: DINO + SD sdaligned (2026-08-27) ==="
python analysis/eval_geodesic_error.py \
  --eval_pairs_path "$PAIRS" \
  --features_path demo/guitar/features_analyze/dino_sd_aligned/guitar_01_pix_dino_sd_features_aligned.pth \
  --vertex_features_path experiments/guitar/encoder_dino_sd_dfd_crossview_sdaligned_relational_geodesic_mu3dscale1_phase2/pred_f.pth \
  --temperature 1.0 \
  --run_label "DINO + SD sdaligned (2026-08-27)" \
  --plot_path "$OUT/eval_oldruns_dino_sd.png" \
  --save_path "$OUT/eval_oldruns_dino_sd.pth"

echo "=== COMPARISON ==="
python analysis/compare_eval_runs.py \
  --runs "$OUT/eval_oldruns_dino_only.pth" "$OUT/eval_oldruns_dino_sd.pth" \
  --labels "DINO only (2026-08-17)" "DINO + SD (2026-08-27)" \
  --save_path "$OUT/eval_compare_dino_vs_dinosd_oldruns.png"
