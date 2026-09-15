#!/usr/bin/env bash
# Adds a comparison of one ISOBBOX arm against the ISOBBOX SD+DINO reference run (Tal, 2026-09-02).
# Usage: ./compare_isobbox_arm_vs_sd_dino.sh <arm_demo_dir_name> <arm_label> <arm_eval_tag>
#
# This is the MATCHED control that the small_analyze gate does not build: that one compares against
# the two ORIGINAL-mesh baselines, which differ from an isobbox arm in mesh AND recipe at once. Here
# everything is held fixed except the one thing under test -- same mesh, same frame, same phase-1
# renders, same T, same seed, same checker, same 100 eval points, same geodesic table. So the eval
# comparison needs NO cross-mesh rescaling, and the images are directly stackable.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb

ARM_DIR=demo/guitar/$1
ARM_LABEL=$2
ARM_EVAL_TAG=$3
REF=demo/guitar/runs/isobbox/bsb_isobbox_sdsquash_T0.01_seed0_phase2
REF_LABEL="ISOBBOX SD+DINO T=0.01 seed0 (reference)"
EL=demo/guitar/eval_labels
OUTC=$ARM_DIR/comparison_vs_isobbox_sd_dino
MB=guitar_01_click_270_370_sam_mask_2
CHK=guitar_isotropic_bboxnorm_mesh_checker16_color   # both arms share this mesh's checker
BP=back_projection/${MB}_backproj_${CHK}_soft_snap_k1_side_by_side.png
BPK1=back_projection/${MB}_backproj_${CHK}_k1_side_by_side.png
mkdir -p "$OUTC"

python analysis/stack_images_labeled.py \
  --top_path "$ARM_DIR/analyze_readT/$BP" --top_label "$ARM_LABEL, read T=0.01" \
  --bottom_path "$REF/analyze_readT/$BP" --bottom_label "$REF_LABEL, read T=0.01" \
  --title "back-projection (checker16, soft-snap k=1) -- matched read T=0.01, same mesh" \
  --output_path "$OUTC/7_back_projection_readT_matched.png"

python analysis/stack_images_labeled.py \
  --top_path "$ARM_DIR/analyze/$BP" --top_label "$ARM_LABEL, read T=1 (soft-snap)" \
  --bottom_path "$REF/analyze/$BP" --bottom_label "$REF_LABEL, read T=1 (soft-snap)" \
  --title "back-projection soft-snap k=1, read T=1 -- same mesh" \
  --output_path "$OUTC/8_back_projection_soft_readT1.png"

python analysis/stack_images_labeled.py \
  --top_path "$ARM_DIR/analyze/$BPK1" --top_label "$ARM_LABEL, read T=1 (top1k)" \
  --bottom_path "$REF/analyze/$BPK1" --bottom_label "$REF_LABEL, read T=1 (top1k)" \
  --title "back-projection top1k, read T=1 -- same mesh" \
  --output_path "$OUTC/9_back_projection_top1k_readT1.png"

# same mesh on both sides, so no --rescale_across_meshes here -- and if that ever becomes necessary
# it means something is wrong with the pair, not with the tool.
python analysis/compare_eval_runs.py \
  --runs $EL/eval_${ARM_EVAL_TAG}.pth $EL/eval_isobbox_stride14_nosdsquare_T0.01_seed0.pth \
  --labels "$ARM_LABEL" "$REF_LABEL" \
  --save_path "$OUTC/eval_vs_isobbox_sd_dino.png"

echo "wrote $OUTC"
