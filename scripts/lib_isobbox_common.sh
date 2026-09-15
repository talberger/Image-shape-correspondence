# Shared constants for every isobbox arm. Sourced, not executed.
# 2026-09-08: written when the A/A'/B arm names were replaced by <framing>_<fusegrid>.
OBJ=./meshes/guitar_isotropic_bboxnorm.obj
NAME=guitar_isobbox
MASK=demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth
FINE=demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth
POSCOLOR=demo/guitar/segment_masks_pixel270_370/pixel_coloring/guitar_01_click_270_370_sam_mask_2_position_color.png
CHECKER=demo/guitar/colored_mesh_isobbox/guitar_isotropic_bboxnorm_mesh_checker16_color.pth
GEO3D=./experiments/guitar/geodesic_tables/guitar_isobbox_geo_table_3d.pth
GEO2D=./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth
PHASE1=./experiments/guitar/encoder_isobbox_dino_sd_dfd_crossview_phase1
RUNS=demo/guitar/runs/isobbox
# 2026-09-15, Tal: every arm of the PROPORTION direction (the per-segment mu_3d_scale family and everything
# built on the new main baseline) lives in its own root. Its names all carry a `_mu_` token
# (mu_learn_perseg, mu_closed_perseg_nbrmix, ...); ZS and the older controls stay in $RUNS.
PROP_RUNS=demo/guitar/runs/isobbox_prop
run_dir() {  # <demo_run_name> -> its directory under whichever root it belongs to
  case "$1" in *_mu_*) echo "$PROP_RUNS/$1";; *) echo "$RUNS/$1";; esac
}
MB=guitar_01_click_270_370_sam_mask_2
CHK_STEM=guitar_isotropic_bboxnorm_mesh_checker16_color
PIXTAG=explicit9
PIXEL_INDICES="9501 22740 31719 46561 75904 92318 156233 184915 188068"
# ORDER MATTERS -- feature_similarity_heatmap.py lays panels out in ARGUMENT order, so a different
# order puts a different vertex in the same panel position and every comparison is meaningless.
QUERY_VERTICES="535 549 1017 2885"
QUERY_PIXELS="151 94 571 206 820 360 820 740"
# eval v2 (Tal, 2026-09-07): v2's .pp was picked on guitar_isotropic.obj, so it is ISOBBOX-ONLY --
# there are no v2 labels for the old 4036v mesh and old-mesh baselines can no longer be compared.
EVALPAIRS=demo/guitar/eval_labels/v2_edge_inset/isobbox/pixel_vertex_eval_pairs.pth
EVALDIR=demo/guitar/eval_labels/v2_edge_inset/isobbox
# The four fuse arms, by photo-feature file. <sd_frame>_<fuse grid>.
PIX_sdsquash_fuse16=demo/guitar/features_analyze/dino_sd_stride14_nosdsquare/guitar_01_pix_stride14_nosdsquare.pth
PIX_sdsquare_fuse16=demo/guitar/features_analyze/dino_sd_aligned/guitar_01_pix_dino_sd_features_aligned.pth
PIX_sdnopad_fuse32=demo/guitar/features_analyze/dino_sd_stride14_B_nopad_cg32/guitar_01_pix_stride14_B_nopad_cg32.pth
PIX_sdsquare_fuse32=demo/guitar/features_analyze/dino_sd_sdsquare_cg32/guitar_01_pix_sdsquare_cg32.pth
