#!/bin/bash
# Finish the T=0.001 arm of the no-sd_square stride-7 + gridalign sweep (2026-09-01).
# Analysis for this arm was deliberately deferred at the 2026-08-31 pod close (PROJECT_NOTES
# section 7); training and eval were already done. Three stages, same as the other three arms:
#   1. step-progression analysis          -> step_progression_grid.png
#   2. full_analyze at read T=1           -> analyze/ + analysis_summary.png
#   3. soft mapping + back-projection at read T = train T = 0.001 -> analyze_readT/
# No pgrep wait loops here on purpose (2026-08-31 section 9: a -f pattern broad enough to match
# the job also matches any shell that merely mentions it).
set -e
cd /workspace/bsb/scripts
L=/workspace/logs
T=0.001
OUTROOT=demo/guitar/runs/oldmesh_sweeps/stride7/bsb_sd_dino_stride7_gridalign_nosdsquare_T${T}_seed0_phase2
RUN=experiments/guitar/encoder_dino_sd_stride7_gridalign_nosdsquare_relational_geodesic_mu3dscale1_T${T}_seed0_phase2
MASK=demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
POSCOLOR=demo/guitar/segment_masks_pixel270_370/pixel_coloring/guitar_01_click_270_370_sam_mask_2_position_color.png
CHECKER=demo/guitar/colored_mesh/guitar_mesh_checker16_color.pth
MASKBASE=guitar_01_click_270_370_sam_mask_2
PIX=demo/guitar/features_analyze/dino_sd_stride7_gridalign_nosdsquare/guitar_01_pix_stride7_gridalign_nosdsquare.pth

echo "===== [step progression] T=$T ====="
./run_stride7_gridalign_nosdsquare_T${T}_seed0_analysis.sh > $L/nosdsquare_T${T}_seed0_analysis.log 2>&1

echo "===== [full analyze, read T=1] T=$T ====="
./run_stride7_gridalign_nosdsquare_T${T}_seed0_full_analyze.sh > $L/nosdsquare_T${T}_seed0_full_analyze.log 2>&1
grep -h "vertex coverage" $L/nosdsquare_T${T}_seed0_full_analyze.log | tail -1

echo "===== [read at train T=$T] soft mapping + back-projection ====="
( source /workspace/activate_bsb.sh; cd /workspace/bsb
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

echo "===== T=0.001 ARM DONE ====="
