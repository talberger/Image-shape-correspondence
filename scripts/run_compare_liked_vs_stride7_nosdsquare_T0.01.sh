#!/bin/bash
# Head-to-head, ONLY these two runs (Tal's request, 2026-09-01):
#   liked : encoder_dino_sd_dfd_crossview_relational_geodesic_mu3dscale1_T0.01_phase2 (2026-08-27)
#   s7    : encoder_dino_sd_stride7_gridalign_nosdsquare_..._T0.01_seed0_phase2 (2026-08-31)
# Both: squashed SD framing, train T=0.01, read T=0.01 (the s7 side uses analyze_readT/).
# Outputs: the n=100 ground-truth eval figure + the back-projection pair Tal named.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
EL=demo/guitar/eval_labels
D=demo/guitar
LIKED=$D/bsb_sd_dino_dfd_crossview_relational_geodesic_mu3dscale1_T0.01_phase2
S7=$D/bsb_sd_dino_stride7_gridalign_nosdsquare_T0.01_seed0_phase2
OUT=$S7/comparison_liked_head_to_head
BP=back_projection/guitar_01_click_270_370_sam_mask_2_backproj_guitar_mesh_checker16_color_soft_snap_k1_side_by_side.png
mkdir -p "$OUT"

python analysis/compare_eval_runs.py \
  --runs $EL/eval_T0.01_phase2.pth $EL/eval_stride7_gridalign_nosdsquare_T0.01_seed0.pth \
  --labels "LIKED: stride14 T=0.01 (08-27)" "stride7+gridalign T=0.01 seed0" \
  --save_path "$OUT/eval_liked_vs_stride7_nosdsquare_T0.01.png"

python analysis/stack_images_labeled.py \
  --top_path "$S7/analyze_readT/$BP" \
  --top_label "stride7 + gridalign + squash, train T=0.01, read T=0.01 (seed0)" \
  --bottom_path "$LIKED/analyze/$BP" \
  --bottom_label "LIKED: stride14 + squash, train T=0.01, read T=0.01 (2026-08-27, unseeded)" \
  --title "back-projection (checker16, soft-snap k=1) -- matched framing and read temperature, stride differs" \
  --output_path "$OUT/backproj_liked_vs_stride7_readT_matched.png"
