#!/bin/bash
# Post-chain analysis for the fuse32 arms (2026-09-08). Runs the FULL per-arm battery
# (features_analyze included, always) on both the ZS and the phase-2 read of sdsquare_fuse32,
# then every comparison that isolates one axis of the 2x2.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh

P1_32=./experiments/guitar/encoder_isobbox_fuse32_phase1
P2_32=./experiments/guitar/encoder_isobbox_sdsquare_fuse32_T1_seed0_phase2

echo "############ arm 1/2: ZS sdsquare_fuse32 (phase-1 only, the new fuse32 encoder) ############"
./scripts/analyze_isobbox_arm.sh bsb_isobbox_ZS_sdsquare_fuse32 \
  "$P1_32/pred_f.pth" "$PIX_sdsquare_fuse32" isobbox_ZS_sdsquare_fuse32

echo "############ arm 2/2: phase-2 sdsquare_fuse32 T=1 seed0 ############"
./scripts/analyze_isobbox_arm.sh bsb_isobbox_sdsquare_fuse32_T1_seed0_phase2 \
  "$P2_32/pred_f.pth" "$PIX_sdsquare_fuse32" isobbox_sdsquare_fuse32_T1_seed0

# The two fuse16 phase-2 arms predate analyze_isobbox_arm.sh: they have a sim heatmap but NO PCA,
# so the feature-analysis half of every comparison against them would silently be skipped. Rebuild
# them through the same script so both sides of each comparison are produced identically, and
# re-emit their evals under the new <framing>_<fusegrid> tag names.
echo "############ backfill: phase-2 sdsquare_fuse16 T=1 (adds PCA, renames eval tag) ############"
./scripts/analyze_isobbox_arm.sh bsb_isobbox_sdsquare_olddiag2d_T1_seed0_phase2 \
  ./experiments/guitar/encoder_isobbox_dino_sd_stride14_sdsquare_relational_geodesic_mu3dscale1_T1_seed0_phase2/pred_f.pth \
  "$PIX_sdsquare_fuse16" isobbox_sdsquare_fuse16_T1_seed0

echo "############ backfill: phase-2 sdsquash_fuse16 T=1 (the baseline) ############"
./scripts/analyze_isobbox_arm.sh bsb_isobbox_sdsquash_T1_seed0_phase2 \
  ./experiments/guitar/encoder_isobbox_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T1_seed0_phase2/pred_f.pth \
  "$PIX_sdsquash_fuse16" isobbox_sdsquash_fuse16_T1_seed0

echo "############ comparison A: FUSE RESOLUTION isolated, at ZS (square framing both sides) ############"
./scripts/compare_isobbox_arms.sh \
  bsb_isobbox_ZS_sdsquare_fuse32 "ZS sdsquare_fuse32 (SD native 32x24)" \
  bsb_isobbox_ZS_sdsquare "ZS sdsquare_fuse16 (SD onto DINO 16x12)" \
  comparison_vs_ZS_sdsquare_fuse16 \
  isobbox_ZS_sdsquare_fuse32 isobbox_ZS_sdsquare_fuse16

echo "############ comparison B: FUSE RESOLUTION isolated, at phase-2 T=1 ############"
./scripts/compare_isobbox_arms.sh \
  bsb_isobbox_sdsquare_fuse32_T1_seed0_phase2 "phase-2 sdsquare_fuse32 T=1" \
  bsb_isobbox_sdsquare_olddiag2d_T1_seed0_phase2 "phase-2 sdsquare_fuse16 T=1" \
  comparison_vs_sdsquare_fuse16_T1 \
  isobbox_sdsquare_fuse32_T1_seed0 isobbox_sdsquare_fuse16_T1_seed0

echo "############ comparison C: does PHASE-2 still hurt at fuse32? (the 2026-09-07 headline) ############"
./scripts/compare_isobbox_arms.sh \
  bsb_isobbox_ZS_sdsquare_fuse32 "ZS sdsquare_fuse32 (phase-1 only)" \
  bsb_isobbox_sdsquare_fuse32_T1_seed0_phase2 "phase-2 sdsquare_fuse32 T=1" \
  comparison_vs_phase2_fuse32 \
  isobbox_ZS_sdsquare_fuse32 isobbox_sdsquare_fuse32_T1_seed0

echo "############ comparison D: FRAMING isolated at phase-2 T=1 (square vs squash, both fuse16) ############"
./scripts/compare_isobbox_arms.sh \
  bsb_isobbox_sdsquare_olddiag2d_T1_seed0_phase2 "phase-2 sdsquare_fuse16 T=1 (square)" \
  bsb_isobbox_sdsquash_T1_seed0_phase2 "phase-2 sdsquash_fuse16 T=1 (squash, baseline)" \
  comparison_vs_sdsquash_fuse16_T1 \
  isobbox_sdsquare_fuse16_T1_seed0 isobbox_sdsquash_fuse16_T1_seed0

echo "############ the whole 2x2 on one eval figure ############"
python analysis/compare_eval_runs.py \
  --runs $EVALDIR/eval_isobbox_ZS_sdsquare_fuse32_v2.pth \
         $EVALDIR/eval_isobbox_ZS_sdsquare_fuse16_v2.pth \
         $EVALDIR/eval_isobbox_ZS_sdsquash_fuse16_v2.pth \
         $EVALDIR/eval_isobbox_sdsquare_fuse32_T1_seed0_v2.pth \
         $EVALDIR/eval_isobbox_sdsquare_fuse16_T1_seed0_v2.pth \
         $EVALDIR/eval_isobbox_sdsquash_fuse16_T1_seed0_v2.pth \
  --labels "ZS sdsquare_fuse32" "ZS sdsquare_fuse16" "ZS sdsquash_fuse16" \
           "P2 sdsquare_fuse32 T=1" "P2 sdsquare_fuse16 T=1" "P2 sdsquash_fuse16 T=1" \
  --save_path $EVALDIR/eval_fuse_2x2_all_arms_v2.png
echo "===== FUSE32 ANALYSIS QUEUE DONE ====="
