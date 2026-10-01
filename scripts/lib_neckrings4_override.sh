# Sourced AFTER lib_isobbox_common.sh when NECKRINGS4=1 (2026-09-16): switch the mesh-side constants
# to meshes/guitar_neckrings4.obj -- guitar.obj with 4 rings x 36 vertices added to the neck gap
# (analysis/densify_neck.py), and NOTHING else changed.
#
# Why this mesh exists: guitar.obj's neck had a 0.321-long stretch (y -0.027 .. 0.294, ~15% of the
# guitar) spanned by a single quad strip of 72 long triangles and containing ZERO vertices. That is
# the leading suspect for the PROTECTED squash run's one long uniform cell on the neck.
#
# The environment below is EXACT, not approximate -- every shared quantity is bit-identical to the
# old mesh's, so the added vertices are the only difference in the whole experiment:
#   frame      `.normalizer.json` {"center":"none"} -- guitar.obj's own normalizer is the identity,
#              so leaving this file untransformed puts it in guitar.obj's EXACT frame. Without the
#              pin, loadmesh re-centres on the new centroid (+0.0046, +0.46% radius), which was
#              measured to flip the argmax vertex of 12% of the mask's pixels.
#   phase-1    the SAME checkpoint as the old mesh, re-evaluated at 4180 vertices. Phase-1 is a
#              coordinate MLP distilled from per-PIXEL (xyz, feature) pairs, so vertex count never
#              entered its training data -- see train_encoder_dfd_crossview.py. Reuse on the shared
#              4036 vertices: mean cos 1.000000, max|diff| 1.6e-06.
#   checker    identical grid (53 cells, [5 17 2], cell size 0.1127) and bit-identical colours on
#              all 4036 shared vertices.
#   geo table  t_coef 20, same as guitar.obj's (t_coef 5 is more accurate but would be a SECOND
#              change; the t5 table is parked in the session scratchpad).
#   photo side untouched: mask, fine segments, 2D geodesic table, photo features.
#
# Eval: v2 labels re-snapped onto this mesh, AND onto plain guitar.obj (.../v2_edge_inset/oldmesh/)
# so the two are directly comparable for the first time. 95/100 labels keep the same ground-truth
# vertex; the 5 that move are exactly the 5 that sat in the gap, where the old mesh's worst snap
# distance was 0.153 (vs 0.024 here).
OBJ=./meshes/guitar_neckrings4.obj
NAME=guitar_neckrings4
CHECKER=demo/guitar/colored_mesh_neckrings4/guitar_neckrings4_mesh_checker16_color.pth
CHK_STEM=guitar_neckrings4_mesh_checker16_color
GEO3D=./experiments/guitar/geodesic_tables/guitar_neckrings4_geo_table_3d.pth
PHASE1=./experiments/guitar/encoder_neckrings4_dino_sd_dfd_crossview_phase1
EVALPAIRS=demo/guitar/eval_labels/v2_edge_inset/neckrings4/pixel_vertex_eval_pairs.pth
EVALDIR=demo/guitar/eval_labels/v2_edge_inset/neckrings4
# The old mesh's query vertices stay valid: densify_neck.py APPENDS, so indices 0-4035 are untouched.
QUERY_VERTICES="2760 642 1545 1527"
MESH_LABEL="neckrings4 4180v"
