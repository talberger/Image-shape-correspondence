#!/bin/bash
# Three-way comparison (Tal, 2026-09-15): an arm vs the MAIN baseline vs the PROTECTED smooth squash run,
# stacked one row per arm. Output: <arm run dir>/comparison_vs_main_and_squash/.
# Each arm is shown at ITS OWN read T: MAIN read T=1, squash read T=0.01 (its analyze_readT), the arm at READ_T.
# The squash run lives on the OLD 4036v mesh: its heatmap uses its own query vertices and it has no eval v2 labels,
# Since 2026-09-16 every mesh has v2 pairs, so the eval figure carries arm + MAIN for any arm;
# squash itself is still shown visual-only (it is a different recipe, not just a different mesh).
#
# Usage: ./compare_vs_main_and_squash.sh <arm_demo_run> <arm_label> <arm_eval_tag>
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

# MAIN always lives in the isobbox frame; an arm on another mesh sources its own override BELOW,
# after these are captured, so the two eval dirs do not collide.
MAIN_EVALDIR=$EVALDIR
MAIN_CHK_STEM=$CHK_STEM
# ARM_NECKRINGS4=1: the arm is on meshes/guitar_neckrings4.obj -> its own checker stem, its own
# eval v2 pairs (2026-09-16). Query vertices are the old-mesh ones, which stay valid there.
[ -n "$ARM_NECKRINGS4" ] && source scripts/lib_neckrings4_override.sh
# ARM_OLDMESH=1: the arm is on guitar.obj. Since 2026-09-16 that mesh HAS v2 eval pairs too
# (.../v2_edge_inset/oldmesh/), so these arms are no longer eval-less.
[ -n "$ARM_OLDMESH" ] && source scripts/lib_oldmesh_override.sh

A=$1; LA=$2; TA=$3
[ -z "$TA" ] && { echo "usage: $0 <arm_demo_run> <arm_label> <arm_eval_tag>"; exit 1; }
DA=$(run_dir "$A")/analyze
MAIN_RUN=bsb_isobbox_sdsquare_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T1_seed0_phase2
MAIN_TAG=isobbox_sdsquare_fuse16_truediag2d_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T1_seed0
DM=$(run_dir "$MAIN_RUN")/analyze
LM="MAIN baseline (mu_learn_perseg + cycle), read T=1"
SQ=demo/guitar/runs/oldmesh_sweeps/stride14/bsb_sd_dino_stride14_nosdsquare_T0.01_seed0_phase2
SQ_ENC=experiments/guitar/encoder_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T0.01_seed0_phase2
LS="SQUASH smooth (PROTECTED, old mesh 4036v), read T=0.01"
OUT=$(run_dir "$A")/comparison_vs_main_and_squash
mkdir -p "$OUT"

ISO_BP=back_projection/${MB}_backproj_${MAIN_CHK_STEM}_soft_snap_k1_side_by_side.png
ISO_K1=back_projection/${MB}_backproj_${MAIN_CHK_STEM}_k1_side_by_side.png
ARM_BP=back_projection/${MB}_backproj_${CHK_STEM}_soft_snap_k1_side_by_side.png
ARM_K1=back_projection/${MB}_backproj_${CHK_STEM}_k1_side_by_side.png
SQ_BP=back_projection/${MB}_backproj_guitar_mesh_checker16_color_soft_snap_k1_side_by_side.png
SQ_K1=back_projection/${MB}_backproj_guitar_mesh_checker16_color_k1_side_by_side.png
ROW=features_analyze/sim_heatmap/simheatmap_row.png
# An arm on a non-isobbox mesh uses ITS OWN checker stem (set by the override sourced above) and the
# old-mesh query vertices, which stay valid on guitar_neckrings4.obj because densify_neck.py appends.
A_BP=$ARM_BP; A_K1=$ARM_K1; A_Q=$QUERY_VERTICES
[ -n "$ARM_OLDMESH" ] && A_Q="2760 642 1545 1527 (old mesh)"

echo "=== [1/4] back-projection soft-snap ==="
python analysis/stack_images_nway.py \
  --panel "$DA/$A_BP" "$LA (soft-snap)" \
  --panel "$DM/$ISO_BP" "$LM (soft-snap)" \
  --panel "$SQ/analyze_readT/$SQ_BP" "$LS (soft-snap)" \
  --title "back-projection soft-snap k=1 -- arm vs MAIN vs squash smooth" \
  --output_path "$OUT/1_back_projection_soft.png"

echo "=== [2/4] back-projection top1k (cosine argmax, independent of read T) ==="
python analysis/stack_images_nway.py \
  --panel "$DA/$A_K1" "$LA (top1k)" \
  --panel "$DM/$ISO_K1" "$LM (top1k)" \
  --panel "$SQ/analyze/$SQ_K1" "SQUASH smooth (PROTECTED, old mesh 4036v) (top1k)" \
  --title "back-projection top1k -- arm vs MAIN vs squash smooth" \
  --output_path "$OUT/2_back_projection_top1k.png"

echo "=== [3/4] similarity heatmap, one row per arm ==="
SQ_ROWDIR=$SQ/analyze/features_analyze/sim_heatmap
if [ ! -f "$SQ_ROWDIR/simheatmap_row.png" ]; then
  # old mesh -> the old query vertices (mirror regions: horn, neck, belly-left, belly-right)
  python analysis/feature_similarity_heatmap.py --obj_path ./meshes/guitar.obj \
    --query_pixels $QUERY_PIXELS --query_vertices 2760 642 1545 1527 \
    --save_dir "$SQ_ROWDIR" --grid_rows 1 --grid_output_path "$SQ_ROWDIR/simheatmap_row.png" \
    --mask_path "$MASK" --features_path "$PIX_sdsquash_fuse16" --vertex_features_path "$SQ_ENC/pred_f.pth"
fi
python analysis/stack_images_nway.py \
  --panel "$DA/$ROW" "$LA (queries $A_Q)" \
  --panel "$DM/$ROW" "$LM (queries $QUERY_VERTICES)" \
  --panel "$SQ_ROWDIR/simheatmap_row.png" "SQUASH smooth, old mesh (its own queries 2760 642 1545 1527 -- other vertices)" \
  --title "feature similarity heatmap -- arm vs MAIN vs squash smooth" \
  --output_path "$OUT/3_features_sim_heatmap.png"

echo "=== [4/4] eval v2 (arm vs MAIN) ==="
# NOTE: an arm on guitar.obj / guitar_neckrings4.obj is scored against ITS OWN mesh's v2 pairs and
# geodesic table, MAIN against isobbox's. The picked points are identical and all three meshes share
# one raw frame, but the ground-truth VERTEX each point snaps to differs per mesh (snap distance
# mean: isobbox 0.0069, neckrings4 0.0144, old mesh 0.0178) -- so read a cross-mesh gap of well
# under an AUC point as noise, not signal.
# MAIN is on isobbox; an arm on another mesh needs the raw-coordinate rescale.
python analysis/compare_eval_runs.py --rescale_across_meshes \
  --runs "$EVALDIR/eval_${TA}_v2.pth" "$MAIN_EVALDIR/eval_${MAIN_TAG}_v2.pth" \
  --labels "$LA" "$LM" \
  --save_path "$OUT/4_eval_v2.png"
echo "===== COMPARISON DONE: $OUT ====="
