#!/usr/bin/env bash
# The three fuse variants at T=1 on the isobbox mesh (Tal, 2026-09-03).
#   A  = sd_square, zero pad          -- ALREADY RUN earlier today, not repeated here.
#   A' = sd_square, replicate pad
#   B  = aspect_nopad + common_grid 32 (Tal's change from 64: 32x24 is SD's native grid, so SD is
#        not resampled at all before normalization).
# small_analyze only, no step progression. Its matched control is the isobbox T=1 SQUASH run, which
# is exactly the baseline comparison asked for.
set -uo pipefail
cd /workspace/bsb; L=/workspace/logs; fail=0
run(){ echo "===== $(date +%H:%M:%S) $1"; shift; bash -c "$*" && echo "  OK" || { echo "  FAILED"; fail=1; }; }

RA=encoder_isobbox_dino_sd_stride14_sdpadreplicate_relational_geodesic_mu3dscale1_T1_seed0_phase2
DA=bsb_isobbox_sd_dino_stride14_sdpadreplicate_T1_seed0_phase2
TA=isobbox_stride14_sdpadreplicate_T1_seed0
PFA=demo/guitar/features_analyze/dino_sd_stride14_sdpad_replicate/guitar_01_pix_stride14_sdpad_replicate.pth
run "1/4 A' phase-2"      "./scripts/run_isobbox_sdpadreplicate_T1_phase2_train.sh > $L/isobbox_sdpadreplicate_T1_train.log 2>&1"
run "2/4 A' small analyze" "./scripts/small_analyze_isobbox_readT1.sh $RA $DA $TA $PFA > $L/isobbox_sdpadreplicate_T1_small_analyze.log 2>&1"

RB=encoder_isobbox_dino_sd_stride14_Bnopadcg32_relational_geodesic_mu3dscale1_T1_seed0_phase2
DB=bsb_isobbox_sd_dino_stride14_Bnopadcg32_T1_seed0_phase2
TB=isobbox_stride14_Bnopadcg32_T1_seed0
PFB=demo/guitar/features_analyze/dino_sd_stride14_B_nopad_cg32/guitar_01_pix_stride14_B_nopad_cg32.pth
run "3/4 B@32 features + phase-2" "./scripts/run_photo_features_B_nopad_cg32.sh > $L/feat_B_nopad_cg32.log 2>&1 && ./scripts/run_isobbox_Bnopadcg32_T1_phase2_train.sh > $L/isobbox_Bnopadcg32_T1_train.log 2>&1"
run "4/4 B@32 small analyze" "./scripts/small_analyze_isobbox_readT1.sh $RB $DB $TB $PFB > $L/isobbox_Bnopadcg32_T1_small_analyze.log 2>&1"
echo "===== $(date +%H:%M:%S) FUSE VARIANTS QUEUE FINISHED (fail=$fail)"
