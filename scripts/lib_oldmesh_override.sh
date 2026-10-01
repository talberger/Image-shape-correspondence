# Sourced AFTER lib_isobbox_common.sh when OLDMESH=1 (2026-09-15): switch the mesh-side constants to the
# OLD 4036v mesh (guitar.obj) -- the mesh of the PROTECTED smooth squash run. Photo side (mask, 2D geo table,
# photo features) is unchanged. See the EVALPAIRS note below for eval.
OBJ=./meshes/guitar.obj
NAME=guitar
CHECKER=demo/guitar/colored_mesh/guitar_mesh_checker16_color.pth
CHK_STEM=guitar_mesh_checker16_color
GEO3D=./experiments/guitar/geodesic_tables/guitar_geo_table_3d.pth
PHASE1=./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1
# old-mesh query vertices, same regions as the isobbox set (horn, neck, belly-left, belly-right)
QUERY_VERTICES="2760 642 1545 1527"
# 2026-09-16: the v2 picked points were re-snapped onto guitar.obj too (they are in the same raw
# frame as guitar_isotropic.obj, which is what they were picked on), so old-mesh arms CAN be
# evaluated now -- the "no eval v2 labels for this mesh" note above is obsolete. Snap quality is
# worse than isobbox's (mean 0.0178 vs 0.0069, max 0.153) and the 5 worst points all sit in the
# neck gap, which is exactly what meshes/guitar_neckrings4.obj fixes.
EVALPAIRS=demo/guitar/eval_labels/v2_edge_inset/oldmesh/pixel_vertex_eval_pairs.pth
EVALDIR=demo/guitar/eval_labels/v2_edge_inset/oldmesh
MESH_LABEL="old mesh 4036v"
