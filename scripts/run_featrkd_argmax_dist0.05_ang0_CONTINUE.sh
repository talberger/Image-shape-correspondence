#!/bin/bash
# Continuation of run_featrkd_argmax_ANALYZE_BEST.sh for the beta=0.05 arm (2026-08-31, new pod).
# Stage [1/4] (step progression) already completed at the previous pod close -- its
# step_progression_grid.png and step300/ outputs are on disk -- so this picks up at [2/4].
# Stages 2-4 are verbatim copies of the corresponding blocks of run_featrkd_argmax_ANALYZE_BEST.sh,
# with BEST fixed to the arm that won the sweep on mean AUC.
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs
BEST=dist0.05_ang0

RUNBASE=encoder_dino_sd_stride14_nosdsquare_featrkd_argmax_${BEST}_T0.01_seed0_phase2
DEMOBASE=bsb_sd_dino_stride14_nosdsquare_featrkd_argmax_${BEST}_T0.01_seed0_phase2

echo "===== [2/4] full analyze (read T=1) ====="
./run_featrkd_argmax_${BEST}_full_analyze.sh > $L/featrkd_argmax_${BEST}_full_analyze.log 2>&1

echo "===== [3/4] soft mapping + back-projection at read T=0.01 ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  MASK=demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
  POSCOLOR=demo/guitar/segment_masks_pixel270_370/pixel_coloring/guitar_01_click_270_370_sam_mask_2_position_color.png
  CHECKER=demo/guitar/colored_mesh/guitar_mesh_checker16_color.pth
  MASKBASE=guitar_01_click_270_370_sam_mask_2
  PIX=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth
  RT=demo/guitar/$DEMOBASE/analyze_readT
  mkdir -p "$RT/color_mesh_from_soft_mapping/render" "$RT/back_projection"
  python analysis/color_mesh_from_soft_mapping.py --mask_path "$MASK" --features_path "$PIX" \
    --vertex_features_path experiments/guitar/$RUNBASE/pred_f.pth \
    --save_dir "$RT/color_mesh_from_soft_mapping" --color_mode position --color_path "$POSCOLOR" \
    --k_vertices_per_pixel 10 --temperature 0.01
  python analysis/render_colored_mesh.py \
    --ply_path "$RT/color_mesh_from_soft_mapping/${MASKBASE}_mesh_color_soft_snap_position_k10.ply" \
    --save_dir "$RT/color_mesh_from_soft_mapping/render" --azim_deg 90 --source_image_path "$POSCOLOR"
  python analysis/color_image_from_knn.py --mask_path "$MASK" \
    --knn_dir "$RT/color_mesh_from_soft_mapping" \
    --knn_path "$RT/color_mesh_from_soft_mapping/${MASKBASE}_soft_snap_correspondence.pth" \
    --vertex_colors_path "$CHECKER" --k 1 --save_dir "$RT/back_projection" \
    --output_path "$RT/back_projection/${MASKBASE}_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"
  rm -f "$RT/back_projection/${MASKBASE}_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"
) > $L/featrkd_argmax_${BEST}_readT.log 2>&1
grep -h "vertex coverage" $L/featrkd_argmax_${BEST}_readT.log | tail -1

echo "===== [4/4] comparison vs the baseline ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  BASE=demo/guitar/runs/oldmesh_sweeps/stride14/bsb_sd_dino_stride14_nosdsquare_T0.01_seed0_phase2
  RUN=demo/guitar/$DEMOBASE
  BP=back_projection/guitar_01_click_270_370_sam_mask_2_backproj_guitar_mesh_checker16_color_soft_snap_k1_side_by_side.png
  cov () { python - "$1" <<'PY'
import sys, torch, glob, os
p = glob.glob(os.path.join(sys.argv[1], 'analyze', 'color_mesh_from_soft_mapping',
                           '*_soft_snap_correspondence.pth'))
if not p:
    print('n/a'); raise SystemExit
c = torch.load(p[0], map_location='cpu')
n = int(c['num_vertices']); k = int(torch.unique(c['top1_vertex_idx']).numel())
print(f'{k}/{n} = {100*k/n:.2f}%')
PY
  }
  RCOV=$(cov "$RUN")
  python analysis/make_run_vs_baseline_comparison.py \
    --run_label "featrkd ARGMAX $BEST (stride14 squash T=0.01 seed0)" \
    --run_phase2_dir "$RUN" --run_phase2_coverage "$RCOV" \
    --phase2_out_dir "$RUN/comparison_vs_baseline"
  python analysis/stack_images_labeled.py \
    --top_path "$RUN/analyze_readT/$BP" \
    --top_label "featrkd ARGMAX $BEST, read T=0.01" \
    --bottom_path "$BASE/analyze_readT/$BP" \
    --bottom_label "BASELINE stride14 squash T=0.01 seed0, read T=0.01" \
    --title "back-projection (checker16, soft-snap k=1) -- feature-RKD argmax vs baseline, matched read T" \
    --output_path "$RUN/comparison_vs_baseline/7_back_projection_readT_matched.png" ) \
  > $L/featrkd_argmax_${BEST}_comparison.log 2>&1
tail -3 $L/featrkd_argmax_${BEST}_comparison.log
echo "===== ANALYSIS DONE: demo/guitar/$DEMOBASE ====="
