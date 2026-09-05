#!/bin/bash
# SMALL ANALYZE -- the fast gate before full_analyze (Tal's spec, 2026-08-31).
# Usage:  ./small_analyze.sh <RUN_NAME_SUFFIX> <EVAL_TAG> [TRAIN_T]
#   e.g.  ./small_analyze.sh featrkd_argmax_dist0.5_ang0 featrkd_argmax_dist0.5_ang0
# RUN_NAME_SUFFIX is the part after encoder_dino_sd_stride14_nosdsquare_ / bsb_sd_dino_stride14_nosdsquare_
#
# What it contains (and deliberately not the rest of full_analyze):
#   at read T=1  -- final step render (the last panel of the step progression), back-projection
#                   top1k + soft-snap, PCA, similarity heatmap
#   at read T=0.01 -- back-projection only
#   then comparison vs BOTH baselines, and eval vs BOTH baselines (into the run's own folder)
# Dropped vs full_analyze: k=1/k=100 colorings, 9-pixel hard-snap markers, 10-panel summary,
# and the 12 intermediate step-progression checkpoints.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb

SUF=$1
EVAL_TAG=$2
[ -z "$SUF" ] && { echo "usage: $0 <run_name_suffix> <eval_tag>"; exit 1; }

RUN=experiments/guitar/encoder_dino_sd_stride14_nosdsquare_${SUF}_T0.01_seed0_phase2
DEMO=demo/guitar/runs/oldmesh_sweeps/stride14/bsb_sd_dino_stride14_nosdsquare_${SUF}_T0.01_seed0_phase2
OUT=$DEMO/analyze
RT=$DEMO/analyze_readT
EV=$DEMO/eval
EL=demo/guitar/eval_labels

MASK=demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
PIXFEAT=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth
POSCOLOR=demo/guitar/segment_masks_pixel270_370/pixel_coloring/guitar_01_click_270_370_sam_mask_2_position_color.png
CHECKER=demo/guitar/colored_mesh/guitar_mesh_checker16_color.pth
MB=guitar_01_click_270_370_sam_mask_2
PIXTAG=explicit9
PIXEL_INDICES="9501 22740 31719 46561 75904 92318 156233 184915 188068"
PRED_F=$RUN/pred_f.pth

mkdir -p "$OUT/color_mesh_from_knn_k10/render" "$OUT/color_mesh_from_soft_mapping/render" \
         "$OUT/back_projection" "$OUT/features_analyze/pca" "$OUT/features_analyze/sim_heatmap" \
         "$RT/color_mesh_from_soft_mapping/render" "$RT/back_projection" "$EV"

echo "=== [1/9] pixel_to_vertex_nn (top_k=10) ==="
python analysis/pixel_to_vertex_nn.py --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT" --top_k 10
KNN="$OUT/${MB}_pixel_to_vertex_nn.pth"

echo "=== [2/9] final-state mesh coloring (k=10, 9 tracked pixels) = last step-progression panel ==="
python analysis/color_mesh_from_knn.py --mask_path "$MASK" --save_dir "$OUT/color_mesh_from_knn_k10" \
  --knn_path "$KNN" --color_mode position --color_path "$POSCOLOR" --k 10 --pixel_indices $PIXEL_INDICES
python analysis/render_colored_mesh.py \
  --ply_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}.ply" \
  --save_dir "$OUT/color_mesh_from_knn_k10/render" --azim_deg 90 \
  --source_image_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_clicked_pixels.png" \
  --click_markers_path "$OUT/color_mesh_from_knn_k10/${MB}_mesh_color_position_k10_pix_${PIXTAG}_click_markers.pth"

echo "=== [3/9] soft mapping (read T=1) -- coverage ==="
python analysis/color_mesh_from_soft_mapping.py --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$OUT/color_mesh_from_soft_mapping" \
  --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10
SOFT="$OUT/color_mesh_from_soft_mapping/${MB}_soft_snap_correspondence.pth"

echo "=== [4/9] back-projection, top1k + soft-snap (read T=1) ==="
python analysis/color_image_from_knn.py --mask_path "$MASK" --knn_dir "$OUT" --knn_path "$KNN" \
  --vertex_colors_path "$CHECKER" --k 1 --save_dir "$OUT/back_projection"
python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$OUT/color_mesh_from_soft_mapping" --knn_path "$SOFT" \
  --vertex_colors_path "$CHECKER" --k 1 --save_dir "$OUT/back_projection" \
  --output_path "$OUT/back_projection/${MB}_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"
rm -f "$OUT/back_projection/${MB}_backproj_guitar_mesh_checker16_color_k1.png" \
      "$OUT/back_projection/${MB}_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"

echo "=== [5/9] feature analysis: PCA ==="
python analysis/pca_feature_compare.py --display both --save_dir "$OUT/features_analyze/pca" \
  --mask_path "$MASK" --features_path "$PIXFEAT" --vertex_features_path "$PRED_F"

echo "=== [6/9] feature analysis: similarity heatmap ==="
python analysis/feature_similarity_heatmap.py \
  --query_pixels 151 94 571 206 820 360 820 740 --query_vertices 1527 1545 2760 642 \
  --save_dir "$OUT/features_analyze/sim_heatmap" --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F"

echo "=== [7/9] read T=0.01: soft mapping + back-projection ONLY ==="
python analysis/color_mesh_from_soft_mapping.py --mask_path "$MASK" --features_path "$PIXFEAT" \
  --vertex_features_path "$PRED_F" --save_dir "$RT/color_mesh_from_soft_mapping" \
  --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10 --temperature 0.01
python analysis/color_image_from_knn.py --mask_path "$MASK" \
  --knn_dir "$RT/color_mesh_from_soft_mapping" \
  --knn_path "$RT/color_mesh_from_soft_mapping/${MB}_soft_snap_correspondence.pth" \
  --vertex_colors_path "$CHECKER" --k 1 --save_dir "$RT/back_projection" \
  --output_path "$RT/back_projection/${MB}_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"
rm -f "$RT/back_projection/${MB}_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"

COV=$(python - "$DEMO" <<'PY'
import sys, torch, glob, os
p = glob.glob(os.path.join(sys.argv[1], 'analyze', 'color_mesh_from_soft_mapping',
                           '*_soft_snap_correspondence.pth'))
if not p: print('n/a'); raise SystemExit
c = torch.load(p[0], map_location='cpu')
n = int(c['num_vertices']); k = int(torch.unique(c['top1_vertex_idx']).numel())
print(f'{k}/{n} = {100*k/n:.2f}%')
PY
)
echo "coverage (read T=1): $COV"

BP=back_projection/${MB}_backproj_guitar_mesh_checker16_color_soft_snap_k1_side_by_side.png
B001=demo/guitar/runs/oldmesh_sweeps/stride14/bsb_sd_dino_stride14_nosdsquare_T0.01_seed0_phase2
B1=demo/guitar/runs/oldmesh_sweeps/stride14/bsb_sd_dino_stride14_nosdsquare_T1_seed0_phase2

echo "=== [8/9] comparison vs BOTH baselines ==="
python analysis/make_run_vs_baseline_comparison.py \
  --run_label "$SUF" --run_phase2_dir "$DEMO" --run_phase2_coverage "$COV" --skip_missing \
  --phase2_out_dir "$DEMO/comparison_vs_baseline_T0.01"
python analysis/make_run_vs_baseline_comparison.py \
  --run_label "$SUF" --run_phase2_dir "$DEMO" --run_phase2_coverage "$COV" \
  --baseline_label "stride14 + squash, T=1, train_seed 0 -- PROTECTED COMPANION BASELINE" \
  --baseline_phase2_dir "$B1" --baseline_phase2_coverage "1038/4036 = 25.72%" --skip_missing \
  --phase2_out_dir "$DEMO/comparison_vs_baseline_T1"
python analysis/stack_images_labeled.py \
  --top_path "$DEMO/analyze_readT/$BP" --top_label "$SUF, read T=0.01" \
  --bottom_path "$B001/analyze_readT/$BP" \
  --bottom_label "BASELINE stride14 squash T=0.01 seed0, read T=0.01" \
  --title "back-projection (checker16, soft-snap k=1) -- matched read T" \
  --output_path "$DEMO/comparison_vs_baseline_T0.01/7_back_projection_readT_matched.png"

BPK1=back_projection/${MB}_backproj_guitar_mesh_checker16_color_k1_side_by_side.png
for pair in "T0.01:$B001" "T1:$B1"; do
  tag=${pair%%:*}; bdir=${pair#*:}
  python analysis/stack_images_labeled.py \
    --top_path "$DEMO/analyze/$BP" --top_label "$SUF, read T=1 (soft-snap)" \
    --bottom_path "$bdir/analyze/$BP" --bottom_label "BASELINE $tag, read T=1 (soft-snap)" \
    --title "back-projection soft-snap k=1, read T=1 -- vs baseline $tag" \
    --output_path "$DEMO/comparison_vs_baseline_$tag/8_back_projection_soft_readT1.png"
  python analysis/stack_images_labeled.py \
    --top_path "$DEMO/analyze/$BPK1" --top_label "$SUF, read T=1 (top1k)" \
    --bottom_path "$bdir/analyze/$BPK1" --bottom_label "BASELINE $tag, read T=1 (top1k)" \
    --title "back-projection top1k, read T=1 -- vs baseline $tag" \
    --output_path "$DEMO/comparison_vs_baseline_$tag/9_back_projection_top1k_readT1.png"
done

echo "=== [9/9] eval vs BOTH baselines (into the run's own folder) ==="
if [ ! -f "$EL/eval_${EVAL_TAG}.pth" ]; then
  echo "  (no eval_${EVAL_TAG}.pth yet -- running eval_geodesic_error.py)"
  python analysis/eval_geodesic_error.py \
    --run_label "$SUF" \
    --eval_pairs_path "$EL/pixel_vertex_eval_pairs.pth" --features_path "$PIXFEAT" \
    --vertex_features_path "$PRED_F" --k_vertices_per_pixel 10 \
    --save_path "$EL/eval_${EVAL_TAG}.pth" --plot_path "$EL/eval_${EVAL_TAG}.png"
fi
python analysis/compare_eval_runs.py \
  --runs $EL/eval_${EVAL_TAG}.pth \
         $EL/eval_stride14_nosdsquare_T0.01_seed0.pth \
         $EL/eval_stride14_nosdsquare_T1_seed0.pth \
  --labels "$SUF" "BASELINE T=0.01" "BASELINE companion T=1" \
  --save_path "$EV/eval_vs_both_baselines.png"

echo "===== SMALL ANALYZE DONE: $DEMO ====="
