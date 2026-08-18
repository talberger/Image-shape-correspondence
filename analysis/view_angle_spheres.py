"""One sphere marker per phase-1 training-view camera angle (encoder.py's generate_random_views
output, --view_sampling random or fibonacci), baked directly into the guitar mesh's geometry as
a real, standalone .ply -- viewable on its own (e.g. in MeshLab), no render-time markers path
needed. Same modules.utils.add_spheres merge pattern color_mesh_hard_snap_markers.py already
uses for a different purpose (marking hard-snapped correspondence vertices); reused here to mark
camera positions instead.

Built 2026-08-15, Tal's request, to visually settle a disagreement about how "spread out" the
original random view sampling actually is vs. the new --view_sampling fibonacci scheme -- a
matplotlib scatter answered it well enough for a quick look, but a real .ply lets the actual
mesh geometry and the camera positions be inspected together, rotated freely, at full
resolution, the standard way every other geometric result in this project is reviewed.

Camera positions: elev/azim (radians, render_views' own convention) -> unit-sphere xyz exactly
as modules/render.py's Renderer computes it internally (get_camera_from_view2), pushed out to
--marker_radius_scale times the mesh's own bounding radius so the markers sit clearly outside
the mesh surface, not embedded in it.

**2026-08-16 fix**: the xyz formula here originally swapped Y<->Z relative to
get_camera_from_view2, despite this docstring already (wrongly) claiming they matched --
elev=0/azim=90 (the empirically verified front) landed at the top pole (0,1,0) instead of the
horizon (0,0,1). Found while investigating why the fibonacci view-sampling ZR eval underperforms
random despite better front coverage (see PROJECT_NOTES.md); confirmed via a direct numeric
comparison against get_camera_from_view2, not just re-reading the docstring. Fixed below.
Anything rendered with the old formula (any .ply this script produced before 2026-08-16) has its
markers in the wrong place and should be regenerated.

**2026-08-16**: coloring switched from raw (x,y,z) position to front/back/left/right quadrant
(matching analysis/view_angle_quadrant_plot.py's scheme and PROJECT_NOTES.md's quadrant
convention: front=azim in [45,135), left=[-45,45), back=[-135,-45), right=[135,225)) -- easier
to read at a glance than a continuous xyz color gradient, and directly comparable to that
script's 2D quadrant-count plot.

Mesh itself left at a flat --uncovered_color (default light gray, same as
color_mesh_hard_snap_markers.py's own default) -- no vertex painting, this script is purely
about where the cameras were, not about any correspondence result.
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils import add_spheres, loadmesh, write_ply

# matches analysis/view_angle_quadrant_plot.py's QUADRANT_COLORS (hex -> 0-255 RGB)
QUADRANT_COLORS = {
    'front': (225, 87, 89),
    'right': (78, 121, 167),
    'back': (89, 161, 79),
    'left': (241, 206, 99),
}


def quadrant_labels(azim_rad):
    """front=[45,135), left=[-45,45), back=[-135,-45), right=[135,225) -- see
    analysis/view_angle_quadrant_plot.py for the derivation/verification of this convention."""
    azim_deg = (np.degrees(azim_rad) + 180) % 360 - 180  # wrap to [-180, 180)
    labels = np.full(azim_deg.shape, 'right', dtype=object)
    labels[(azim_deg >= 45) & (azim_deg < 135)] = 'front'
    labels[(azim_deg >= -45) & (azim_deg < 45)] = 'left'
    labels[(azim_deg >= -135) & (azim_deg < -45)] = 'back'
    return labels


def main(args):
    data = torch.load(args.angles_path, map_location='cpu')
    elev = torch.cat(data['elev']).numpy()
    azim = torch.cat(data['azim']).numpy()
    n = len(elev)
    print(f'{n} camera angles loaded from {args.angles_path}')

    mesh = loadmesh(dir=args.obj_path, name=args.name)
    verts = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    mesh_radius = float(np.linalg.norm(verts, axis=1).max())

    base_o3d = o3d.geometry.TriangleMesh()
    base_o3d.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    base_o3d.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    uncovered_frac = np.array(args.uncovered_color, dtype=np.float64) / 255.0
    base_o3d.vertex_colors = o3d.utility.Vector3dVector(np.tile(uncovered_frac, (verts.shape[0], 1)))

    r = args.marker_radius_scale * mesh_radius
    x = r * np.cos(elev) * np.cos(azim)
    y = r * np.sin(elev)
    z = r * np.cos(elev) * np.sin(azim)
    sphere_points = np.stack([x, y, z], axis=1).astype(np.float32)

    labels = quadrant_labels(azim)
    sphere_colors = np.array([QUADRANT_COLORS[lbl] for lbl in labels], dtype=np.float32) / 255.0
    counts = {q: int((labels == q).sum()) for q in QUADRANT_COLORS}
    print(f'quadrant counts: {counts}')

    print(f'merging {n} sphere markers (radius={args.sphere_radius}, '
          f'resolution={args.sphere_resolution}) at {r:.3f} (marker_radius_scale='
          f'{args.marker_radius_scale} x mesh_radius={mesh_radius:.3f})...')
    merged = add_spheres(base_o3d, sphere_points, sphere_colors, save_mesh=False,
                         sphere_radius=args.sphere_radius, sphere_resolution=args.sphere_resolution)
    merged_verts = np.asarray(merged.vertices, dtype=np.float32)
    merged_faces = np.asarray(merged.triangles, dtype=np.int64)
    merged_colors = np.clip(np.round(np.asarray(merged.vertex_colors) * 255.0), 0, 255).astype(np.int32)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.output_name)
    write_ply(save_path, merged_verts, merged_faces, merged_colors)
    print(f'mesh + {n} view-angle spheres saved to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--angles_path', type=str, required=True,
                        help="a *random_viewing_angles.pt from encoder.py's "
                             'generate_random_views (either --view_sampling scheme)')
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, default='view_angle_spheres.ply')
    parser.add_argument('--uncovered_color', nargs=3, type=int, default=[200, 200, 200],
                        help='flat mesh color -- same default as '
                             'color_mesh_hard_snap_markers.py')
    parser.add_argument('--marker_radius_scale', type=float, default=1.4,
                        help='camera markers sit at this multiple of the mesh bounding radius')
    parser.add_argument('--sphere_radius', type=float, default=0.015,
                        help='small by design -- 1000 markers at the --sphere_radius=0.04 '
                             'default elsewhere in this project would overlap heavily')
    parser.add_argument('--sphere_resolution', type=int, default=4,
                        help='low-poly by design -- keeps a 1000-marker .ply a manageable size '
                             '(vs. the 12 used for a handful of markers elsewhere)')
    args = parser.parse_args()
    main(args)
