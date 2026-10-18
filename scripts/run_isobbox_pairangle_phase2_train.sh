#!/usr/bin/env bash
# PAIRWISE (2-point) feature-angle term, Tal's variant (2026-09-04), on the ISOBBOX guitar mesh.
#   cos(center, neighbour) image-side vs mesh-side, one per pair, SOFT student (gradient through
#   all 10 neighbours), added ALONGSIDE the geometric relational loss -- not instead of it.
#   No mu anywhere: a cosine is scale-invariant by construction.
# Usage: ./run_isobbox_pairangle_phase2_train.sh <weight> <attached|detachfeat|detachw> [all|in_segment]
#   detachw = --feat_detach_weights (Tal, 2026-09-04): freeze the softmax weights w, leave F live,
#             so the gradient reaches the encoder ONLY through the F[v_k] values.
#   3rd arg = which neighbours the PAIR-ANGLE term uses (default all). `in_segment` = the 8 CLOSE
#   ones only (Tal, 2026-09-04); the spatial loss still uses all 16 either way.
#   attached   = F on the gradient path (the plain variant)
#   detachfeat = --feat_detach_features, so w is the only route to the encoder -- the ablation that
#                tests whether the term is being satisfied by reshaping F rather than by fixing
#                the correspondence.
# Base recipe = the isobbox T=1 baseline (Tal's arm on this mesh), everything else identical.
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

W=$1
MODE=$2
NB=${3:-all}
SPATIAL=${4:-on}      # on | off -- `off` sets --rkd_spatial_dist_weight 0, i.e. the pair-angle term ALONE
DW=${5:-0}            # RKD-D (feature DISTANCE) weight; 0 = off, keeps every earlier run's name
# NAMING (2026-09-04, Tal): the tag names the gradient path that stays LIVE, not the one that is
# cut -- "detachfeat"/"detachw" were confusing because each leaves the OTHER path active.
#   gradFW = both paths live      gradW = only the softmax-weight path      gradF = only the F values
case "$MODE" in
  attached|gradFW)   DETACH=""; TAG="gradFW" ;;
  detachfeat|gradW)  DETACH="--feat_detach_features"; TAG="gradW" ;;
  detachw|gradF)     DETACH="--feat_detach_weights"; TAG="gradF" ;;
  *) echo "usage: $0 <weight> <gradFW|gradW|gradF> [all|in_segment]"; exit 1 ;;
esac
case "$NB" in
  all)        NBTAG="" ;;
  in_segment) NBTAG="_nbrClose" ;;
  *) echo "3rd arg must be all|in_segment"; exit 1 ;;
esac
# --rkd_spatial_dist_weight 0 removes the geometric relational term entirely, leaving the pair-angle term
# as the whole objective (Tal, 2026-09-04: "rak et ha-loss ha-ze, klomar lelo spatial"). NOTE the
# spatial term is what anchors the map to real geometry, so this run has nothing holding the
# correspondence in place -- that is the experiment, not an oversight.
case "$SPATIAL" in
  on)  RELW=""; SPTAG="" ;;
  off) RELW="--rkd_spatial_dist_weight 0"; SPTAG="_noSpatial" ;;
  *) echo "4th arg must be on|off"; exit 1 ;;
esac
# RKD-D on features (--rkd_feat_dist_weight): the ORIGINAL feature term, which unlike the angle needs
# mu on both sides (fixed, measured once from phase-1: mu_img 0.9889, mu_msh 0.66054 for the soft
# student). The detach flags above apply to BOTH feature terms at once -- they share `msh_all` --
# so gradF/gradW/gradFW means the same thing for the distance term as for the angle term.
DISTFLAG=""; DTAG=""
if [ "$DW" != "0" ]; then DISTFLAG="--rkd_feat_dist_weight $DW"; DTAG="_dist${DW}"; fi

# stride14 / nosdsquare / dino_sd dropped from the name: constant for every run, and the
# dino_sd-vs-sd_dino inconsistency was itself the cause of a folder-split bug on 2026-09-04.
OUT=./experiments/guitar/encoder_isobbox_pairangle${W}${NBTAG}${DTAG}_soft_${TAG}${SPTAG}_T1_seed0_phase2
echo "=== pairangle=$W dist=$DW mode=$TAG neighbours=$NB spatial=$SPATIAL -> $OUT ==="
time python phase2/train_relational_loss_l2norm.py \
  --obj_path ./meshes/guitar_isotropic_bboxnorm.obj --name guitar_isobbox \
  --init_encoder_dir ./experiments/guitar/encoder_isobbox_dino_sd_dfd_crossview_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 --temperature 1.0 --train_seed 0 \
  --feat_student_mode soft --rkd_feat_angle2_weight "$W" $DETACH \
  --rkd_feat_angle2_neighbors "$NB" $RELW $DISTFLAG \
  --geo_table_3d_path ./experiments/guitar/geodesic_tables/guitar_isobbox_geo_table_3d.pth \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir "$OUT"
echo "PAIRANGLE PHASE-2 DONE w=$W mode=$TAG nb=$NB spatial=$SPATIAL"
