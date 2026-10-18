#!/bin/bash
# THE DENSITY COMPARISON (Tal, 2026-09-16): the neck-rings arm vs arm 3, which are identical in every
# respect except the 144 vertices added to the neck gap -- same recipe, same phase-1 checkpoint, same
# frame, same photo features, same cycle weight (deliberately not recalibrated), same seed.
# Everything else this project varies is held fixed, so a difference here IS the vertex density.
#
# Output: <neckrings4 run dir>/comparison_vs_oldmesh/
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh

N4=bsb_neckrings4_sdsquash_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0_phase2
OM=bsb_oldmesh_sdsquash_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0_phase2
DN=$(run_dir "$N4")/analyze
DO=$(run_dir "$OM")/analyze
OUT=$(run_dir "$N4")/comparison_vs_oldmesh
mkdir -p "$OUT"

LN="NECK RINGS (4180v, gap filled), read T=0.01"
LO="ARM 3 -- old mesh (4036v, empty gap), read T=0.01"
N4_CHK=guitar_neckrings4_mesh_checker16_color
OM_CHK=guitar_mesh_checker16_color
ROW=features_analyze/sim_heatmap/simheatmap_row.png

echo "=== [1/4] back-projection soft-snap ==="
python analysis/stack_images_nway.py \
  --panel "$DN/back_projection/${MB}_backproj_${N4_CHK}_soft_snap_k1_side_by_side.png" "$LN (soft-snap)" \
  --panel "$DO/back_projection/${MB}_backproj_${OM_CHK}_soft_snap_k1_side_by_side.png" "$LO (soft-snap)" \
  --title "DENSITY TEST -- back-projection soft-snap k=1; the ONLY difference is 144 neck vertices" \
  --output_path "$OUT/1_back_projection_soft.png"

echo "=== [2/4] back-projection top1k ==="
python analysis/stack_images_nway.py \
  --panel "$DN/back_projection/${MB}_backproj_${N4_CHK}_k1_side_by_side.png" "$LN (top1k)" \
  --panel "$DO/back_projection/${MB}_backproj_${OM_CHK}_k1_side_by_side.png" "$LO (top1k)" \
  --title "DENSITY TEST -- back-projection top1k" \
  --output_path "$OUT/2_back_projection_top1k.png"

echo "=== [3/4] similarity heatmap ==="
python analysis/stack_images_nway.py \
  --panel "$DN/$ROW" "$LN (queries 2760 642 1545 1527)" \
  --panel "$DO/$ROW" "$LO (queries 2760 642 1545 1527)" \
  --title "DENSITY TEST -- feature similarity heatmap (same query vertices: indices are preserved)" \
  --output_path "$OUT/3_features_sim_heatmap.png"

echo "=== [4/4] eval v2 ==="
# Each arm is scored on ITS OWN mesh's v2 pairs and geodesic table. The picked points are the same
# 100 and all meshes share one raw frame, but the ground-truth vertex differs for the 5 points that
# sat in the gap -- which is the very thing being tested, so this comparison is the point, not a flaw.
# the two arms are on different meshes (4180v vs 4036v), so rescale to raw .obj coords.
python analysis/compare_eval_runs.py --rescale_across_meshes \
  --runs "demo/guitar/eval_labels/v2_edge_inset/neckrings4/eval_neckrings4_sdsquash_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0_v2.pth" \
         "demo/guitar/eval_labels/v2_edge_inset/oldmesh/eval_oldmesh_sdsquash_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0_v2.pth" \
  --labels "$LN" "$LO" \
  --save_path "$OUT/4_eval_v2.png"
echo "===== DENSITY COMPARISON DONE: $OUT ====="
