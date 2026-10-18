#!/bin/bash
# Full analysis for all four no-sd_square arms + the two comparison sets Tal asked for
# (2026-08-31). Waits for the training sweep to finish first.
#
# READ TEMPERATURE, and why there are two passes. color_mesh_from_soft_mapping.py takes its own
# --temperature, independent of the training one, and per the 2026-08-30 finding it is the READ
# value that moves the weight profile and therefore the renders. The run Tal likes
# (dfd_crossview T=0.01, 2026-08-27) was READ at 0.01; every stride-7 + gridalign run so far was
# read at the default 1. So:
#   pass 1 -- standard full_analyze, read T=1: keeps this sweep internally comparable AND
#             comparable to last pod's sd_square arms.
#   pass 2 -- soft mapping + back-projection re-run at read T = the arm's own training T, into
#             analyze_readT/: this is what makes the T=0.01 arm a fair comparison against the
#             run Tal likes. Skipped for T=1 (identical to pass 1 by construction).
set -e
cd $DISC_ROOT/scripts
L=$DISC_WS/logs
D=$DISC_ROOT/demo/guitar
MASK=demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
POSCOLOR=demo/guitar/segment_masks_pixel270_370/pixel_coloring/guitar_01_click_270_370_sam_mask_2_position_color.png
CHECKER=demo/guitar/colored_mesh/guitar_mesh_checker16_color.pth
MASKBASE=guitar_01_click_270_370_sam_mask_2
PIX=demo/guitar/features_analyze/dino_sd_stride7_gridalign_nosdsquare/guitar_01_pix_stride7_gridalign_nosdsquare.pth

# --- wait for the training sweep -------------------------------------------------------------
while pgrep -f "run_stride7_gridalign_nosdsquare_T_sweep" >/dev/null; do sleep 20; done
echo "===== training sweep finished, starting analysis ====="

for T in 1 0.1 0.01 0.001; do
  OUTROOT=demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_gridalign_nosdsquare_T${T}_seed0_phase2
  RUN=experiments/guitar/encoder_dino_sd_stride7_gridalign_nosdsquare_relational_geodesic_mu3dscale1_T${T}_seed0_phase2

  rm -f $L/nosdsquare_T${T}_seed0_analysis.log $L/nosdsquare_T${T}_seed0_full_analyze.log \
        $L/nosdsquare_T${T}_seed0_readT.log
  echo "===== [step progression] T=$T ====="
  ./run_stride7_gridalign_nosdsquare_T${T}_seed0_analysis.sh > $L/nosdsquare_T${T}_seed0_analysis.log 2>&1

  echo "===== [full analyze, read T=1] T=$T ====="
  ./run_stride7_gridalign_nosdsquare_T${T}_seed0_full_analyze.sh > $L/nosdsquare_T${T}_seed0_full_analyze.log 2>&1
  grep -h "vertex coverage" $L/nosdsquare_T${T}_seed0_full_analyze.log | tail -1

  if [ "$T" != "1" ]; then
    echo "===== [read at train T=$T] soft mapping + back-projection ====="
    ( source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
      RT=$OUTROOT/analyze_readT
      mkdir -p "$RT/color_mesh_from_soft_mapping/render" "$RT/back_projection"
      python analysis/color_mesh_from_soft_mapping.py --mask_path "$MASK" --features_path "$PIX" \
        --vertex_features_path "$RUN/pred_f.pth" --save_dir "$RT/color_mesh_from_soft_mapping" \
        --color_mode position --color_path "$POSCOLOR" --k_vertices_per_pixel 10 --temperature $T
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
    ) > $L/nosdsquare_T${T}_seed0_readT.log 2>&1
    grep -h "vertex coverage" $L/nosdsquare_T${T}_seed0_readT.log | tail -1
  fi
done

echo "===== ALL FOUR ANALYSED ====="
