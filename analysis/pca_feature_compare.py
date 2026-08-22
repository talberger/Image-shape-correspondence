"""
PCA feature comparison view: color the masked image, the mesh, or both, by
PCA(3) of their DINO features -- a thin wrapper around pixel_coloring.py's
--mode pca and mesh_coloring.py's --mode pca.

The point of --display both is a *comparable* view: with --basis shared
(default), one PCA is fit on the union of masked-pixel features and
per-vertex features, so equal colors in the two outputs actually mean
similar features across domains, not just within each domain separately
(which is all --basis independent, or running pixel_coloring.py/
mesh_coloring.py separately, ever gave).

Whenever a mesh is colored (--display mesh or both), this also renders it
(reusing render_colored_mesh.py's render_one -- same mesh load + Renderer
construction, same click-marker-sphere machinery, so this is a thin wrapper,
not a reimplementation of the slow kaolin setup, ~120s JIT warm-up).
Default: a single front view (elev=0, azim=90 -- this project's confirmed
front angle for meshes/guitar.obj). Pass --elev_deg/--azim_deg for more.

Run (both, shared basis):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/pca_feature_compare.py \
  --display both --save_dir ./demo/guitar/features_analyze/pca/ \
  --mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --vertex_features_path ./experiments/guitar/encoder_dino_relational_geodesic_mu3dscale1_frontcenteredinit_newmask/pred_f.pth
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pixel_coloring import (load_features, load_mask, build_output_image,
                                      pca_colors, pca_colors_shared)
from analysis.mesh_coloring import load_vertex_features
from analysis.render_colored_mesh import render_one
from modules.mesh import Mesh
from modules.utils import get_img_name, get_features_in_mask, write_ply, loadmesh, path_provenance_tag
from modules.render import Renderer


def main(args):
    show_image = args.display in ('image', 'both')
    show_mesh = args.display in ('mesh', 'both')

    if show_image and args.mask_path is None:
        raise ValueError('--mask_path is required when --display is image or both')
    if show_mesh and args.vertex_features_path is None:
        raise ValueError('--vertex_features_path is required when --display is mesh or both '
                          '(no default on purpose -- BSB-demo/phase-1/phase-2 pred_f.pth files '
                          'are not interchangeable)')

    features_in_mask = mask = mask_idx_row = mask_idx_col = pixel_tag = None
    if show_image:
        mask = load_mask(args.mask_path)
        img_name = get_img_name(args.img_path)
        features_path = args.features_path or os.path.join(args.save_dir, f'{img_name}_pix_dino_features.pth')
        pixel_features = load_features(features_path)
        pixel_tag = path_provenance_tag(features_path)
        if pixel_features.shape[:2] != mask.shape:
            raise ValueError(f'features spatial shape {tuple(pixel_features.shape[:2])} '
                              f'!= mask shape {tuple(mask.shape)}')
        features_in_mask, mask_idx_row, mask_idx_col = get_features_in_mask(pixel_features, mask)
        if features_in_mask.shape[0] < args.n_components:
            raise ValueError(f'mask has only {features_in_mask.shape[0]} True pixels, '
                              f'fewer than n_components={args.n_components}')
        print(f'pixel features: {features_path} (tag: {pixel_tag})')

    verts = faces = vertex_features = vertex_tag = None
    if show_mesh:
        mesh = Mesh(args.obj_path)
        verts = mesh.vertices.detach().cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy()
        vertex_features = load_vertex_features(args.vertex_features_path, verts.shape[0])
        vertex_tag = path_provenance_tag(args.vertex_features_path)
        if verts.shape[0] < args.n_components:
            raise ValueError(f'mesh has only {verts.shape[0]} vertices, '
                              f'fewer than n_components={args.n_components}')
        print(f'vertex features: {args.vertex_features_path} (tag: {vertex_tag})')

    pixel_colors = vertex_colors = None
    if args.display == 'both' and args.basis == 'shared':
        pixel_colors, vertex_colors = pca_colors_shared(
            features_in_mask, vertex_features, args.n_components, args.random_state)
    else:
        if show_image:
            pixel_colors = pca_colors(features_in_mask, args.n_components, args.random_state)
        if show_mesh:
            vertex_colors = pca_colors(vertex_features, args.n_components, args.random_state)

    basis_tag = args.basis if args.display == 'both' else 'independent'
    image_output_path = mesh_output_path = None

    if show_image:
        mask_basename = os.path.splitext(os.path.basename(args.mask_path))[0]
        image_output_path = args.image_output_path or os.path.join(
            args.save_dir, f'{mask_basename}_pca_{basis_tag}_color.png')
        img = build_output_image(mask.shape, mask_idx_row, mask_idx_col, pixel_colors,
                                  args.background_color, args.transparent_background)
        os.makedirs(os.path.dirname(os.path.abspath(image_output_path)), exist_ok=True)
        Image.fromarray(img).save(image_output_path)
        if args.display != 'both':
            print(f'image saved to {image_output_path}')

    if show_mesh:
        obj_name = os.path.splitext(os.path.basename(args.obj_path))[0]
        mesh_output_path = args.mesh_output_path or os.path.join(
            args.save_dir, f'{obj_name}_mesh_pca_{basis_tag}_color.ply')
        colors_output_path = os.path.splitext(mesh_output_path)[0] + '.pth'
        os.makedirs(os.path.dirname(os.path.abspath(mesh_output_path)), exist_ok=True)
        write_ply(mesh_output_path, verts, faces, vertex_colors.astype(np.int32))
        torch.save({
            'vertex_colors': torch.from_numpy(vertex_colors),
            'mode': 'pca',
            'basis': basis_tag,
            'obj_path': args.obj_path,
            'vertex_features_path': args.vertex_features_path,
            'features_path': features_path if show_image else None,
        }, colors_output_path)
        print(f'mesh saved to {mesh_output_path} (colors: {colors_output_path})')

        render_mesh = loadmesh(args.obj_path, name='guitar')
        renderer = Renderer(dim=(args.render_res, args.render_res))
        if args.display == 'both':
            # Side-by-side already contains both the plain image and the rendered
            # mesh -- no need to also keep either standalone copy.
            render_one(render_mesh, renderer, mesh_output_path, args.save_dir,
                       args.elev_deg, args.azim_deg, save_rendered_views_grid=False,
                       output_path=None, source_image_path=image_output_path,
                       side_by_side_path=None, click_markers_path=None,
                       marker_sphere_radius=0.04, marker_sphere_resolution=12, marker_min_radius_frac=0.35)
            os.remove(image_output_path)
        else:
            render_one(render_mesh, renderer, mesh_output_path, args.save_dir,
                       args.elev_deg, args.azim_deg, save_rendered_views_grid=True,
                       output_path=None, source_image_path=None,
                       side_by_side_path=None, click_markers_path=None,
                       marker_sphere_radius=0.04, marker_sphere_resolution=12, marker_min_radius_frac=0.35)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', type=str, choices=['image', 'mesh', 'both'], default='both')
    parser.add_argument('--basis', type=str, choices=['shared', 'independent'], default='shared',
                         help='--display both only. shared: fit one PCA on the union of masked-'
                              'pixel and vertex features, so equal colors mean similar features '
                              'across domains. independent: fit PCA separately per domain (like '
                              'running pixel_coloring.py/mesh_coloring.py separately) -- colors '
                              'are then not comparable between the image and the mesh.')
    parser.add_argument('--img_path', type=str, default='./images/guitar_01.png',
                         help='used only to derive the default pixel features filename')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--features_path', type=str, default=None,
                         help='defaults to <save_dir>/<img_name>_pix_dino_features.pth')
    parser.add_argument('--mask_path', type=str, default=None,
                         help='required when --display is image or both')
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--vertex_features_path', type=str, default=None,
                         help='required when --display is mesh or both. No default on purpose -- '
                              'BSB-demo/phase-1/phase-2 pred_f.pth files are not interchangeable.')
    parser.add_argument('--image_output_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_pca_<basis>_color.png')
    parser.add_argument('--mesh_output_path', type=str, default=None,
                         help='defaults to <save_dir>/<obj_name>_mesh_pca_<basis>_color.ply')
    parser.add_argument('--n_components', type=int, default=3)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--render_res', type=int, default=224, help='--display mesh or both only')
    parser.add_argument('--elev_deg', nargs='+', type=float, default=[0],
                         help='--display mesh or both only. Elevation angles (degrees) to render.')
    parser.add_argument('--azim_deg', nargs='+', type=float, default=[90],
                         help='--display mesh or both only. Azimuth angles (degrees) to render. '
                              'Default single view at 90, this project\'s confirmed front angle for '
                              'meshes/guitar.obj (see render_colored_mesh.py). Pass multiple values '
                              'for more angles.')
    parser.add_argument('--background_color', nargs=3, type=int, default=[255, 255, 255])
    parser.add_argument('--transparent_background', type=int, default=None,
                         help='if set (0-255), output PNG is RGBA with this alpha for background pixels')

    args = parser.parse_args()
    main(args)
