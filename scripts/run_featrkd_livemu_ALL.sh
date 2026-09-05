#!/bin/bash
# Three feature-RKD runs, 2026-08-29, testing Tal's normalization hypothesis: does the RKD paper's
# LIVE mu (recomputed from the batch every step) behave differently from our FIXED mu (measured
# once from phase-1)?
#
#   dist0.2_ang0_fixed   CONTROL. Same recipe as 2026-08-28's dist0.2_ang0, re-run today seeded and on
#                        this pod's A4500. Any gap vs that run = hardware + phase-2 RNG, not mu.
#   dist0.2_ang0_live    RKD-D only, live mu.
#   dist0.2_ang0.2_live  RKD-D + RKD-A, live mu.
#
# Only TWO arms change under live mu: RKD-A has no normalizer at all (psi_A is a cosine, already
# scale-free -- see modules/feature_rkd_loss.py), so the angle-only arm is mathematically
# unaffected and is NOT re-run. Its 2026-08-28 numbers stand.
#
# WHAT TO WATCH: mu_msh_live in the logs. Fixed reference = 0.74032. On 2026-08-28 the live value
# stayed within 2% of it whenever the distance term was active, but drifted to 0.672 (-12%) in the
# angle-only run where nothing anchored the scale. If the live-mu runs drift the same way, the
# anchor is gone -- which is the whole mechanical difference between the two modes.
set -e
cd /workspace/bsb/scripts
L=/workspace/logs
EL=demo/guitar/eval_labels
PIXFEAT=demo/guitar/features_analyze/dino_sd_stride7_gridalign/guitar_01_pix_stride7_gridalign.pth

for cfg in dist0.2_ang0_fixed dist0.2_ang0_live dist0.2_ang0.2_live; do
  echo "########## $cfg : phase-2 ##########"
  ./run_featrkd_${cfg}_phase2_train.sh > $L/featrkd_${cfg}_phase2_train.log 2>&1
  echo "########## $cfg : step progression ##########"
  ./run_featrkd_${cfg}_analysis.sh     > $L/featrkd_${cfg}_analysis.log 2>&1
  echo "########## $cfg : eval ##########"
  ( source /workspace/activate_bsb.sh; cd /workspace/bsb
    python analysis/eval_geodesic_error.py \
      --run_label "featRKD $cfg" \
      --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth \
      --features_path $PIXFEAT \
      --vertex_features_path experiments/guitar/encoder_dino_sd_stride7_gridalign_relational_featrkd_${cfg}_phase2/pred_f.pth \
      --k_vertices_per_pixel 10 \
      --save_path $EL/eval_featrkd_${cfg}.pth \
      --plot_path $EL/eval_featrkd_${cfg}.png ) > $L/featrkd_${cfg}_eval.log 2>&1
done

echo "########## comparison: baseline + yesterday's fixed + today's three ##########"
( source /workspace/activate_bsb.sh; cd /workspace/bsb
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_stride7_gridalign.pth \
           $EL/eval_featrkd_dist0.2_ang0.pth \
           $EL/eval_featrkd_dist0.2_ang0_fixed.pth \
           $EL/eval_featrkd_dist0.2_ang0_live.pth \
           $EL/eval_featrkd_dist0.2_ang0.2_live.pth \
    --labels "baseline (no feat RKD)" \
             "dist0.2 fixed mu (08-28, unseeded)" \
             "dist0.2 fixed mu (08-29, seeded)" \
             "dist0.2 LIVE mu" \
             "dist0.2+ang0.2 LIVE mu" \
    --save_path $EL/eval_compare_featrkd_livemu.png ) > $L/featrkd_livemu_compare.log 2>&1
echo "########## ALL DONE ##########"
