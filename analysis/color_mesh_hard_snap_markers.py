"""
mesh markers, alternative to color_mesh_from_knn.py's *_click_markers.pth: instead of a
pixel's top-K feature-nearest vertices, mark only the model's own final choice -- the
hard-snapped vertex color_mesh_from_soft_mapping.py already computed for it (softmax-weighted
average of the pixel's top-K feature-nearest vertices' positions -> nearest actual vertex,
DPC Eq. 1-3). One sphere marker per selected pixel, colored the same as that pixel's own
marker color in the source photo (same load_pixel_colors() convention color_mesh_from_knn.py
uses for its own click markers) -- no vertex painting at all, since
color_mesh_from_soft_mapping.py's own *_mesh_color_soft_snap_*.ply already shows the painted
version (as an unweighted average over ALL masked pixels landing on a vertex, not a per-pixel
trace of a handful of chosen ones), and a single painted vertex is invisible at render
resolution anyway (see render_colored_mesh.py's own docstring on why click markers exist at
all) -- painting one here would add nothing visible.

Reads an existing *_soft_snap_correspondence.pth (color_mesh_from_soft_mapping.py's own output --
already computed for every fully-documented run) rather than recomputing the soft mapping: that
file already stores, for every masked pixel, its hard-snapped vertex, in the same
(pixel_row, pixel_col) mask-iteration order pixel_to_vertex_nn.py/color_mesh_from_knn.py use, so
the same --num_pixels/--seed selection lands on the identical pixels as every other seed-152
recipe in this project (verified byte-identical against an existing run's *_click_markers.pth
top-1 column before this script was written). --save_dir defaults to sitting right next to
that file -- i.e. inside the run's own analyze/color_mesh_from_soft_mapping/ folder, alongside
the full-mesh soft-snap coloring this method is a companion view of, rather than a separate
top-level folder.

Does NOT generate its own marked source photo: color_mesh_from_knn.py's per-checkpoint pipeline
(--num_pixels 7 --seed 152) already produces a *_clicked_pixels.png for the exact same 7
pixels/colors (same selection, same --color_path) as part of every run's step_progression_grid
-- pass that existing file to render_colored_mesh.py's --source_image_path directly rather than
regenerating a duplicate; the side-by-side render already embeds it, so no standalone copy is
kept here either.

The sphere markers are baked directly into a real, standalone .ply via modules.utils.add_spheres
(the same helper the original demo pipeline already uses this exact way --
image_shape_correspondence.py:326 -- the merge logic itself is not reimplemented), rather than
only existing as a render-time overlay: the saved .ply is viewable on its own (e.g. in MeshLab)
with the markers already part of the geometry, no --click_markers_path/render_colored_mesh.py
step required to see them. add_spheres's own save_mesh=True path writes via open3d's raw writer,
whose face-index field name (open3d writes 'vertex_indices') this project's own
render_colored_mesh.py doesn't read (it expects write_ply's 'vertex_index', the convention every
other .ply here uses) -- so save_mesh=False is used instead and the merged geometry is written
through write_ply, keeping this file readable the same way as everything else. Built in
loadmesh()'s normalized (bounding-sphere) space, matching the space soft_map_to_vertices/
render_colored_mesh.py's own click markers already use -- vertex order is unaffected by
normalization, so hard_vertex_idx indexes it exactly the same as the raw mesh.

Run (mu3dscale1.1, step300, matching the project's standard 7-pixel seed-152 selection):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/color_mesh_hard_snap_markers.py \
  --mask_path ./demo/guitar/bsb_origin/guitar_01_click_130_140_sam_mask_2.pth \
  --soft_snap_correspondence_path ./demo/guitar/bsb_relational_bbox_norm_mu3dscale1.1/analyze/color_mesh_from_soft_mapping/guitar_01_click_130_140_sam_mask_2_soft_snap_correspondence.pth \
  --color_path ./demo/guitar/bsb_retrained/pixel_coloring/guitar_01_click_130_140_sam_mask_2_position_color.png

Then render directly -- no --click_markers_path needed, the spheres are already in the geometry:
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/render_colored_mesh.py \
  --ply_path ./demo/guitar/bsb_relational_bbox_norm_mu3dscale1.1/analyze/color_mesh_from_soft_mapping/guitar_01_click_130_140_sam_mask_2_hard_snap_spheres_pix_random7_seed152.ply \
  --save_dir ./demo/guitar/bsb_relational_bbox_norm_mu3dscale1.1/analyze/color_mesh_from_soft_mapping/render \
  --azim_deg 90 \
  --source_image_path ./demo/guitar/bsb_relational_bbox_norm_mu3dscale1.1/step300/color_mesh_from_knn_k10/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_clicked_pixels.png
"""

import argparse
import os
import sys

import numpy as np
import torch
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.color_mesh_from_knn import check_mask_matches_knn, load_pixel_colors, select_pixels
from modules.utils import add_spheres, loadmesh, write_ply


def main(args):
    mask_basename = os.path.splitext(os.path.basename(args.mask_path))[0]
    save_dir = args.save_dir or os.path.dirname(os.path.abspath(args.soft_snap_correspondence_path))
    os.makedirs(save_dir, exist_ok=True)

    soft = torch.load(args.soft_snap_correspondence_path, map_location='cpu')
    for key in ('pixel_row', 'pixel_col', 'top1_vertex_idx', 'mask_shape'):
        if key not in soft:
            raise ValueError(f'{args.soft_snap_correspondence_path} missing expected key {key!r} '
                              f'-- not a color_mesh_from_soft_mapping.py correspondence file?')
    check_mask_matches_knn(args.mask_path, soft, args.soft_snap_correspondence_path)

    n_masked = soft['pixel_row'].shape[0]
    selected_idx, selection_desc = select_pixels(args, n_masked)
    if selected_idx is None:
        raise ValueError('--num_pixels or --pixel_indices is required -- marking every masked '
                          'pixel defeats the point of this script (use '
                          'color_mesh_from_soft_mapping.py\'s own full-mesh .ply for that)')
    selected_idx_t = torch.from_numpy(selected_idx)

    pixel_row = soft['pixel_row'][selected_idx_t]
    pixel_col = soft['pixel_col'][selected_idx_t]
    hard_vertex_idx = soft['top1_vertex_idx'][selected_idx_t]  # [N]

    pixel_colors = load_pixel_colors(args.color_path, soft['mask_shape'], pixel_row, pixel_col)

    print(f'selected {len(selected_idx)}/{n_masked} masked pixels via {selection_desc}:')
    for local_i, orig_i in enumerate(selected_idx):
        r, c = int(pixel_row[local_i]), int(pixel_col[local_i])
        v = int(hard_vertex_idx[local_i])
        print(f'  pixel #{orig_i} (row={r}, col={c}) -> hard-snap vertex {v}')

    # render_colored_mesh.py's --click_markers_path shape (color_mesh_from_knn.py's own
    # *_click_markers.pth convention), K=1: exactly one marker per pixel, at its hard-snap
    # vertex, in its own marker color. 'similarity' only ever drives per-pixel *relative*
    # radius spread across a pixel's K markers (add_click_marker_spheres); with K=1 there is
    # nothing to spread against, so it already renders at full --marker_sphere_radius
    # regardless of this value -- kept only because the field is required by that code path.
    markers_path = args.markers_output_path or os.path.join(
        save_dir, f'{mask_basename}_hard_snap_markers_pix_{selection_desc}.pth')
    torch.save({
        'vertex_idx': hard_vertex_idx.long().unsqueeze(1),                    # [N, 1]
        'similarity': torch.ones(len(selected_idx), 1, dtype=torch.float32),  # [N, 1], placeholder
        'color': pixel_colors,                                                # [N, 3]
    }, markers_path)
    print(f'hard-snap markers (1 per selected pixel) saved to {markers_path}')

    # Real, standalone .ply: mesh left at a flat --uncovered_color everywhere (no vertex
    # painting -- see module docstring), with one sphere per selected pixel merged directly into
    # the geometry via modules.utils.add_spheres (same helper/pattern already used for this exact
    # purpose in image_shape_correspondence.py) -- viewable on its own, no render-time
    # --click_markers_path needed. Normalized (bounding-sphere) space, matching the space
    # render_colored_mesh.py's own click markers already use -- vertex order is unaffected by
    # normalization, so hard_vertex_idx indexes it exactly the same as the raw mesh would.
    norm_mesh = loadmesh(args.obj_path, name='guitar')
    verts = norm_mesh.vertices.detach().cpu().numpy()
    faces = norm_mesh.faces.detach().cpu().numpy()

    base_o3d = o3d.geometry.TriangleMesh()
    base_o3d.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    base_o3d.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    uncovered_frac = np.array(args.uncovered_color, dtype=np.float64) / 255.0
    base_o3d.vertex_colors = o3d.utility.Vector3dVector(np.tile(uncovered_frac, (verts.shape[0], 1)))

    sphere_points = verts[hard_vertex_idx.numpy()].astype(np.float32)          # [N, 3]
    sphere_colors = (pixel_colors.numpy() / 255.0).astype(np.float32)          # [N, 3], in [0, 1]

    # save_mesh=False: add_spheres's own save path writes via open3d's raw writer (binary,
    # face field named 'vertex_indices'), which render_colored_mesh.py's load_ply_geometry can't
    # read (it expects this project's own write_ply convention -- ASCII, 'vertex_index' singular
    # -- the same one every other .ply in this pipeline, including mesh_coloring.py's checker
    # output, already uses successfully through that exact code path). Take the merged geometry
    # back instead and write it through write_ply so this file is readable the same way as
    # everything else.
    merged = add_spheres(base_o3d, sphere_points, sphere_colors, save_mesh=False,
                          sphere_radius=args.sphere_radius, sphere_resolution=args.sphere_resolution)
    merged_verts = np.asarray(merged.vertices, dtype=np.float32)
    merged_faces = np.asarray(merged.triangles, dtype=np.int64)
    merged_colors = np.clip(np.round(np.asarray(merged.vertex_colors) * 255.0), 0, 255).astype(np.int32)

    spheres_path = args.spheres_output_path or os.path.join(
        save_dir, f'{mask_basename}_hard_snap_spheres_pix_{selection_desc}.ply')
    write_ply(spheres_path, merged_verts, merged_faces, merged_colors)
    print(f'mesh with hard-snap spheres baked into the geometry saved to {spheres_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=str, required=True,
                         help='used to confirm --soft_snap_correspondence_path was computed from '
                              'this same mask')
    parser.add_argument('--soft_snap_correspondence_path', type=str, required=True,
                         help='*_soft_snap_correspondence.pth from color_mesh_from_soft_mapping.py '
                              '-- read directly, never recomputed')
    parser.add_argument('--color_path', type=str, required=True,
                         help="pixel_coloring.py output (pca_color.png or position_color.png) to pull "
                              "each selected pixel's own marker color from")
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--save_dir', type=str, default=None,
                         help='defaults to the directory --soft_snap_correspondence_path is already '
                              'in (the run\'s analyze/color_mesh_from_soft_mapping/ folder), so this '
                              'view sits next to the full-mesh soft-snap coloring it is a companion of')
    parser.add_argument('--num_pixels', type=int, default=7,
                         help='how many masked pixels to mark (project convention: 7). Ignored if '
                              '--pixel_indices is given.')
    parser.add_argument('--pixel_selection', type=str, default='random', choices=['random'],
                         help='how to pick --num_pixels pixels (only "random" for now)')
    parser.add_argument('--pixel_indices', type=int, nargs='+', default=None,
                         help='explicit 0-based indices into the masked-pixel array to mark, instead '
                              'of --num_pixels/--pixel_selection')
    parser.add_argument('--seed', type=int, default=152,
                         help='random seed for --pixel_selection random (project convention: 152)')
    parser.add_argument('--markers_output_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_hard_snap_markers_pix_<selection>.pth')
    parser.add_argument('--spheres_output_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_hard_snap_spheres_pix_<selection>.ply')
    parser.add_argument('--uncovered_color', nargs=3, type=int, default=[200, 200, 200],
                         help='RGB (0-255) for every mesh vertex NOT covered by a sphere marker')
    parser.add_argument('--sphere_radius', type=float, default=0.04,
                         help='marker sphere radius, in the mesh\'s unit-bounding-sphere coordinate '
                              'frame -- matches render_colored_mesh.py\'s own --marker_sphere_radius '
                              'default, since these spheres are meant to look the same size')
    parser.add_argument('--sphere_resolution', type=int, default=12,
                         help='marker sphere mesh resolution (facets) -- matches '
                              'render_colored_mesh.py\'s own --marker_sphere_resolution default')

    args = parser.parse_args()
    main(args)
