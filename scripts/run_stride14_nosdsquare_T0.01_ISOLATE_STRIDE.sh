#!/bin/bash
# ITEM 3 of the 2026-09-01 queue: stride 14 + squash + read T=0.01 -- the run that matches the one
# Tal likes (dfd_crossview T=0.01, 2026-08-27) on ALL THREE axes that separated it from the
# 2026-08-31 sweep (SD framing, read temperature, stride), so the only thing left between this arm
# and the stride-7 + gridalign arms is the DINO patch stride.
# Stages: photo features -> phase-2 train (seed 0) -> ground-truth eval -> step progression ->
# full_analyze (read T=1) -> soft mapping + back-projection at read T=0.01 -> comparisons.
# Waits on a PID, never a pgrep -f pattern (2026-08-31 section 9).
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs
D=demo/guitar
EL=demo/guitar/eval_labels
RUN=experiments/guitar/encoder_dino_sd_stride14_nosdsquare_relational_geodesic_mu3dscale1_T0.01_seed0_phase2
OUTROOT=$D/bsb_sd_dino_stride14_nosdsquare_T0.01_seed0_phase2
PIX=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth
MASK=demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
POSCOLOR=demo/guitar/segment_masks_pixel270_370/pixel_coloring/guitar_01_click_270_370_sam_mask_2_position_color.png
CHECKER=demo/guitar/colored_mesh/guitar_mesh_checker16_color.pth
MASKBASE=guitar_01_click_270_370_sam_mask_2
LIKED=$D/bsb_sd_dino_dfd_crossview_relational_geodesic_mu3dscale1_T0.01_phase2
S7=$D/bsb_sd_dino_stride7_gridalign_nosdsquare_T0.01_seed0_phase2

if [ -n "${WAIT_PID:-}" ]; then
  echo "waiting for pid $WAIT_PID (T=0.001 arm) ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi

echo "===== [1/7] photo features: stride 14, squash framing ====="
./run_sd_dino_stride14_nosdsquare_photo_features.sh > $L/stride14_nosdsquare_photo_features.log 2>&1

echo "===== [2/7] phase-2 train, T=0.01, seed 0 ====="
./run_stride14_nosdsquare_T0.01_seed0_phase2_train.sh > $L/stride14_nosdsquare_T0.01_seed0_train.log 2>&1

echo "===== [3/7] ground-truth eval (n=100, eval temperature fixed at 1.0 as always) ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  python analysis/eval_geodesic_error.py \
    --run_label "stride14 no-sd_square T=0.01 seed0" \
    --eval_pairs_path $EL/pixel_vertex_eval_pairs.pth \
    --features_path $PIX \
    --vertex_features_path $RUN/pred_f.pth \
    --k_vertices_per_pixel 10 \
    --save_path $EL/eval_stride14_nosdsquare_T0.01_seed0.pth \
    --plot_path $EL/eval_stride14_nosdsquare_T0.01_seed0.png
  python analysis/compare_eval_runs.py \
    --runs $EL/eval_stride14_nosdsquare_T0.01_seed0.pth \
           $EL/eval_stride7_gridalign_nosdsquare_T0.01_seed0.pth \
           $EL/eval_stride7_gridalign_nosdsquare_T1_seed0.pth \
    --labels "stride14 T=0.01" "stride7+gridalign T=0.01" "stride7+gridalign T=1" \
    --save_path $EL/eval_compare_stride_ab_nosdsquare_T0.01.png ) > $L/stride14_nosdsquare_T0.01_seed0_eval.log 2>&1

echo "===== [4/7] step progression ====="
./run_stride14_nosdsquare_T0.01_seed0_analysis.sh > $L/stride14_nosdsquare_T0.01_seed0_analysis.log 2>&1

echo "===== [5/7] full analyze, read T=1 ====="
./run_stride14_nosdsquare_T0.01_seed0_full_analyze.sh > $L/stride14_nosdsquare_T0.01_seed0_full_analyze.log 2>&1
grep -h "vertex coverage" $L/stride14_nosdsquare_T0.01_seed0_full_analyze.log | tail -1

echo "===== [6/7] soft mapping + back-projection at read T=0.01 ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
  RT=$OUTROOT/analyze_readT
  mkdir -p "$RT/color_mesh_from_soft_mapping/render" "$RT/back_projection"
  python analysis/color_mesh_from_soft_mapping.py --mask_path "$MASK" --features_path "$PIX" \
    --vertex_features_path "$RUN/pred_f.pth" --save_dir "$RT/color_mesh_from_soft_mapping" \
    --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10 --temperature 0.01
  python analysis/render_colored_mesh.py \
    --ply_path "$RT/color_mesh_from_soft_mapping/${MASKBASE}_mesh_color_soft_snap_position_k10.ply" \
    --save_dir "$RT/color_mesh_from_soft_mapping/render" --azim_deg 90 \
    --source_image_path "$POSCOLOR"
  python analysis/color_image_from_knn.py --mask_path "$MASK" \
    --knn_dir "$RT/color_mesh_from_soft_mapping" \
    --knn_path "$RT/color_mesh_from_soft_mapping/${MASKBASE}_soft_snap_correspondence.pth" \
    --vertex_colors_path "$CHECKER" --k 1 --save_dir "$RT/back_projection" \
    --output_path "$RT/back_projection/${MASKBASE}_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"
  rm -f "$RT/back_projection/${MASKBASE}_backproj_guitar_mesh_checker16_color_soft_snap_k1.png"
) > $L/stride14_nosdsquare_T0.01_seed0_readT.log 2>&1
grep -h "vertex coverage" $L/stride14_nosdsquare_T0.01_seed0_readT.log | tail -1

echo "===== [7/7] comparisons ====="
( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
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
  RCOV=$(cov "$OUTROOT"); S7COV=$(cov "$S7")
  echo "stride14 T=0.01 coverage: $RCOV ; stride7+gridalign T=0.01 coverage: $S7COV"

  # (A) vs the run Tal likes -- same framing, same stride, same read T after stage 6
  python analysis/make_run_vs_baseline_comparison.py \
    --run_label "stride14 no sd_square T=0.01 (seed0, this pod)" \
    --run_phase2_dir "$OUTROOT" --run_phase2_coverage "$RCOV" \
    --baseline_label "LIKED: dfd_crossview stride14 T=0.01 (2026-08-27, unseeded)" \
    --baseline_phase2_dir "$LIKED" --baseline_phase2_coverage "1162/4036 = 28.79%" \
    --phase2_out_dir "$OUTROOT/comparison_vs_liked_T0.01"

  # (B) vs the stride-7 + gridalign arm at the same T -- the stride A/B itself
  python analysis/make_run_vs_baseline_comparison.py \
    --run_label "stride14 no sd_square T=0.01 (seed0)" \
    --run_phase2_dir "$OUTROOT" --run_phase2_coverage "$RCOV" \
    --baseline_label "stride7 + gridalign no sd_square T=0.01 (seed0)" \
    --baseline_phase2_dir "$S7" --baseline_phase2_coverage "$S7COV" \
    --phase2_out_dir "$OUTROOT/comparison_vs_stride7"

  # (C) the back-projections at MATCHED read temperature -- the image Tal actually judges by,
  # and the one make_run_vs_baseline_comparison.py does not cover on the phase-2 side.
  BP=back_projection/${MASKBASE}_backproj_guitar_mesh_checker16_color_soft_snap_k1_side_by_side.png
  python analysis/stack_images_labeled.py \
    --top_path "$OUTROOT/analyze_readT/$BP" \
    --top_label "stride14 + squash, train T=0.01, READ T=0.01 (seed0, this pod)" \
    --bottom_path "$LIKED/analyze/$BP" \
    --bottom_label "LIKED: dfd_crossview stride14 T=0.01, read T=0.01 (2026-08-27)" \
    --title "back-projection (checker16, soft-snap k=1) -- matched framing, stride AND read temperature" \
    --output_path "$OUTROOT/comparison_vs_liked_T0.01/7_back_projection_readT_matched.png"
  python analysis/stack_images_labeled.py \
    --top_path "$OUTROOT/analyze_readT/$BP" \
    --top_label "stride14 + squash, read T=0.01 (seed0)" \
    --bottom_path "$S7/analyze_readT/$BP" \
    --bottom_label "stride7 + gridalign + squash, read T=0.01 (seed0)" \
    --title "back-projection (checker16, soft-snap k=1) -- STRIDE A/B at matched read temperature" \
    --output_path "$OUTROOT/comparison_vs_stride7/7_back_projection_readT_matched.png"
) > $L/stride14_nosdsquare_T0.01_seed0_comparisons.log 2>&1
tail -3 $L/stride14_nosdsquare_T0.01_seed0_comparisons.log

echo "===== STRIDE-14 ISOLATION ARM DONE ====="
