#!/bin/bash
# Full analysis for the best arm of the argmax feature-RKD beta sweep (2026-09-01, Tal's request).
# Waits on WAIT_PID (the sweep driver), picks the arm with the highest MEAN AUC across the three
# eval metrics, then runs the same pipeline every promoted run gets: step progression ->
# full_analyze at read T=1 -> soft mapping + back-projection at read T=0.01 (the baseline's own
# read temperature, so the back-projections are comparable) -> comparison vs the baseline.
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs
if [ -n "${WAIT_PID:-}" ]; then
  echo "waiting for pid $WAIT_PID (beta sweep) ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi

TAG=$( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh" >/dev/null 2>&1; cd $DISC_ROOT; python - <<'PY'
import torch, glob, os, re
best, best_tag = -1, None
for p in sorted(glob.glob('demo/guitar/eval_labels/eval_featrkd_argmax_dist*_ang0.pth')):
    a = torch.load(p, map_location='cpu')['auc']
    m = sum(a.values()) / len(a)
    tag = re.sub(r'^eval_featrkd_argmax_|\.pth$', '', os.path.basename(p))
    print(f'# {tag}: mean AUC {100*m:.2f}  ' +
          '  '.join(f'{k} {100*v:.1f}' for k, v in a.items()), flush=True)
    if m > best:
        best, best_tag = m, tag
print(best_tag)
PY
)
BEST=$(echo "$TAG" | tail -1)
echo "$TAG" | grep '^#' || true
echo "===== best arm: $BEST ====="

RUNBASE=encoder_dino_sd_stride14_nosdsquare_featrkd_argmax_${BEST}_T0.01_seed0_phase2
DEMOBASE=bsb_sd_dino_stride14_nosdsquare_featrkd_argmax_${BEST}_T0.01_seed0_phase2

# Generate this arm's two scripts from the baseline's, by path substitution only.
for kind in analysis full_analyze; do
  sed -e "s|encoder_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T0.01_seed0_phase2|$RUNBASE|g" \
      -e "s|bsb_sd_dino_stride14_nosdsquare_T0.01_seed0_phase2|$DEMOBASE|g" \
      run_stride14_nosdsquare_T0.01_seed0_${kind}.sh > run_featrkd_argmax_${BEST}_${kind}.sh
  chmod +x run_featrkd_argmax_${BEST}_${kind}.sh
done

echo "===== [1/4] step progression ====="
./run_featrkd_argmax_${BEST}_analysis.sh > $L/featrkd_argmax_${BEST}_analysis.log 2>&1

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
