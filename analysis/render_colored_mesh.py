"""
visual QA: render a colored mesh (produced by color_mesh_from_knn.py) from
several camera angles, in the same spirit as image_shape_correspondence.py's
'rendered_views_*.png' outputs. Optionally also builds a side-by-side PNG:
a source image on the left, followed by all the rendered views (in order) on
the right, for a quick visual sanity check of the pixel->vertex
correspondence.

The source image should be:
  - the plain <mode>_color.png (pca_color.png or position_color.png,
    from pixel_coloring.py), when the mesh was colored from all masked
    pixels (color_mesh_from_knn.py with no --num_pixels/--pixel_indices), or
  - the *_clicked_pixels.png (also from color_mesh_from_knn.py), when the
    mesh was colored from a handful of hand-picked pixels, so you can see
    exactly which real-photo pixels produced the colors in the render.

Run (view grid only):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/render_colored_mesh.py \
  --ply_path ./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2_mesh_color_pca_k1.ply

Run (view grid + side-by-side against the pixel-coloring source):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/render_colored_mesh.py \
  --ply_path ./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2_mesh_color_pca_k1.ply \
  --source_image_path ./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2_pca_color.png

Run (view grid + side-by-side against the clicked-pixels photo, with sphere
markers on the rendered mesh so the tiny single-vertex dots are actually
visible, for a --num_pixels run):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/render_colored_mesh.py \
  --ply_path ./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2_mesh_color_pca_k1_pix_random5_seed0.ply \
  --source_image_path ./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2_mesh_color_pca_k1_pix_random5_seed0_clicked_pixels.png \
  --click_markers_path ./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2_mesh_color_pca_k1_pix_random5_seed0_click_markers.pth

BATCH MODE (--ply_paths/--save_dirs/..., 2026-08-12): a run's own step_progression_grid
analysis calls this script once per checkpoint (typically 12), and every call repeats
loadmesh(args.obj_path) (base geometry -- identical across checkpoints) and
Renderer(dim=...) construction (camera/rasterizer setup -- also identical: Renderer takes
the mesh as an argument to render_views(), not at construction, so it holds no
mesh-specific state and is safe to reuse across the differently-shaped meshes each
checkpoint's click-marker spheres produce). Measured at ~50s/checkpoint (this + its
color_mesh_from_knn.py call), the dominant cost of the whole per-checkpoint analysis.
Batch mode builds `mesh`/`renderer` ONCE, then loops only over
load_ply_vertex_colors + coloring + rendering + saving per checkpoint. Single-.ply mode
below is byte-for-byte unchanged; batch mode is additive, verified to produce pixel-
identical PNGs to running this script once per checkpoint.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils import device, loadmesh, setcolor_mesh_batched
from modules.render import Renderer


class RenderMesh:
    """Minimal duck-typed stand-in for modules.mesh.Mesh: Renderer.render_views
    and setcolor_mesh_batched only ever touch .vertices/.faces/.face_attributes,
    so this is enough to render arbitrary (verts, faces) without going through
    an .obj file (used when sphere markers are merged in at runtime)."""

    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.face_attributes = None


def add_spheres_varying_radius(mesh_o3d, points, colors, radii, resolution):
    """Like this repo's own modules.utils.add_spheres, but each point gets its
    own radius instead of one shared radius for every marker (add_spheres
    builds a single sphere template and only translates copies of it)."""
    for point, color, radius in zip(points, colors, radii):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=resolution)
        transformation = np.identity(4)
        transformation[:3, 3] = point
        sphere.transform(transformation)
        sphere.paint_uniform_color(color)
        mesh_o3d += sphere
    return mesh_o3d


def add_click_marker_spheres(mesh, vertex_colors, click_markers_path, sphere_radius, sphere_resolution,
                              min_radius_frac):
    """Merges a colored sphere at every one of each clicked pixel's stored
    nearest vertices (from color_mesh_from_knn.py's *_click_markers.pth) into
    `mesh`. A pixel's top-1 (most similar) neighbor is drawn at the full
    `sphere_radius`; weaker neighbors shrink proportionally to how much less
    similar they are than that pixel's own top-1 (floored at
    `min_radius_frac` so a marker never disappears entirely) -- so the sphere
    sizes visualize each neighbor's actual influence on the painted color.
    Returns a new RenderMesh plus the combined per-vertex color array
    (float 0-1) to feed setcolor_mesh_batched."""
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices.detach().cpu().numpy().astype(np.float64))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces.detach().cpu().numpy().astype(np.int32))
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.astype(np.float64))

    click = torch.load(click_markers_path, map_location='cpu')
    vertex_idx = click['vertex_idx']  # [N, K]
    similarity = click['similarity']  # [N, K]
    n, k = vertex_idx.shape

    # Per-pixel min-max normalization (not a raw ratio-to-top1): a pixel's K nearest
    # neighbors are typically all within a percent or two of each other in raw cosine
    # similarity, so scaling radius directly by similarity/top1_similarity makes every
    # sphere nearly the same size. Stretching each pixel's own [min, max] similarity
    # span to the full [min_radius_frac, 1.0] radius range keeps the "smaller sphere =
    # less influence" relationship but makes it actually visible.
    sim_min = similarity.min(dim=1, keepdim=True).values
    sim_max = similarity.max(dim=1, keepdim=True).values
    raw_span = sim_max - sim_min
    span = raw_span.clamp(min=1e-8)
    radius_frac = min_radius_frac + (1.0 - min_radius_frac) * (similarity - sim_min) / span
    # A pixel with no real spread among its shown neighbors (e.g. k=1, or exact ties)
    # has nothing to shrink relative to -- render those at full size instead of falling
    # through to min_radius_frac (which the 0/eps division above would give).
    radius_frac = torch.where(raw_span < 1e-8, torch.ones_like(radius_frac), radius_frac)
    radii = (radius_frac * sphere_radius).reshape(-1).numpy()

    points = mesh.vertices[vertex_idx.reshape(-1)].detach().cpu().numpy().astype(np.float64)
    pixel_colors = (click['color'] / 255.0)  # [N, 3]
    colors = pixel_colors.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3).numpy().astype(np.float64)

    combined_o3d = add_spheres_varying_radius(mesh_o3d, points, colors, radii, resolution=sphere_resolution)

    combined_verts = torch.from_numpy(np.asarray(combined_o3d.vertices, dtype=np.float32)).to(device)
    combined_faces = torch.from_numpy(np.asarray(combined_o3d.triangles, dtype=np.int64)).to(device)
    combined_colors = np.asarray(combined_o3d.vertex_colors, dtype=np.float32)

    print(f'click markers: {n} pixels x {k} nearest neighbors each, sphere radius '
          f'{radii.min():.4f}-{radii.max():.4f} (top-1 always {sphere_radius:.4f})')

    return RenderMesh(combined_verts, combined_faces), combined_colors


def load_ply_vertex_colors(ply_path):
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    colors = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.float32) / 255.0
    return colors


def load_ply_geometry(ply_path):
    """Vertices + faces straight out of the .ply, for files whose geometry is not
    the .obj's own. mesh_coloring.py --mode checker writes such a file: it splits
    every face into its own 3 vertices so each triangle can carry one flat color
    (a shared vertex would force the renderer to blend the two cells' colors).
    Rendering the .obj instead would silently throw that away."""
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    verts = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    faces = np.stack(ply['face'].data['vertex_index']).astype(np.int64)
    return verts, faces


def render_one_view(renderer, mesh, elev_deg, azim_deg):
    elev = torch.tensor([elev_deg], dtype=torch.float32).to(device) / 180 * np.pi
    azim = torch.tensor([azim_deg], dtype=torch.float32).to(device) / 360 * 2 * np.pi
    images, _, _ = renderer.render_views(
        mesh, num_views=1, show=False, random_views=False,
        center_elev=elev, center_azim=azim,
        lighting=True, background=torch.ones(3).to(device),
        return_views=True)
    return images[0].permute(1, 2, 0).detach().cpu().numpy()


def render_one(mesh, renderer, ply_path, save_dir, elev_deg, azim_deg, save_rendered_views_grid,
              output_path, source_image_path, side_by_side_path, click_markers_path,
              marker_sphere_radius, marker_sphere_resolution, marker_min_radius_frac):
    """One .ply -> rendered image(s) [+ side-by-side], given an already-constructed
    `mesh` (base geometry, checkpoint-independent) and `renderer` (camera/rasterizer,
    also checkpoint-independent -- see the module docstring's BATCH MODE note for why
    both are safe to reuse across calls). Factored out so single-.ply mode and batch
    mode share identical logic -- single mode is just a batch of size 1, not a
    separately maintained copy."""
    ply_basename = os.path.splitext(os.path.basename(ply_path))[0]

    vertex_colors = load_ply_vertex_colors(ply_path)
    render_mesh = mesh
    if vertex_colors.shape[0] != mesh.vertices.shape[0]:
        # A .ply with its own geometry (mesh_coloring.py's flat checkerboard splits
        # every face into 3 vertices). Render what the file actually contains; the
        # --obj_path mesh is only the default source of geometry, not a constraint.
        # setcolor_mesh_batched normalizes whatever it is handed, so framing is
        # unchanged.
        ply_verts, ply_faces = load_ply_geometry(ply_path)
        if ply_verts.shape[0] != vertex_colors.shape[0]:
            raise ValueError(f'{ply_path} is internally inconsistent: '
                              f'{ply_verts.shape[0]} vertices, {vertex_colors.shape[0]} colors')
        print(f'{ply_path} has {vertex_colors.shape[0]} vertices vs the .obj\'s '
              f'{mesh.vertices.shape[0]}; rendering the .ply\'s own geometry '
              f'({ply_faces.shape[0]} faces)')
        render_mesh = RenderMesh(torch.from_numpy(ply_verts).to(device),
                                 torch.from_numpy(ply_faces).to(device))

    if click_markers_path is not None:
        render_mesh, vertex_colors = add_click_marker_spheres(
            render_mesh, vertex_colors, click_markers_path,
            sphere_radius=marker_sphere_radius, sphere_resolution=marker_sphere_resolution,
            min_radius_frac=marker_min_radius_frac)

    vertex_colors_t = torch.from_numpy(vertex_colors).float().to(device)
    setcolor_mesh_batched(render_mesh, vertex_colors_t.unsqueeze(0))

    rendered = []  # [(elev, azim, img), ...] in row-major (elev outer, azim inner) order

    if save_rendered_views_grid:
        fig, axs = plt.subplots(len(elev_deg), len(azim_deg),
                                 figsize=(len(azim_deg) * 4.5, len(elev_deg) * 4.5),
                                 squeeze=False, constrained_layout=True)
        for i, e in enumerate(elev_deg):
            for j, a in enumerate(azim_deg):
                img = render_one_view(renderer, render_mesh, e, a)
                rendered.append((e, a, img))
                axs[i, j].imshow(img)
                axs[i, j].set_title(f'elev={e:g}, azim={a:g}', fontsize=11, pad=8)
                axs[i, j].axis('off')

        views_output_path = output_path or os.path.join(save_dir, f'{ply_basename}_rendered_views.png')
        fig.savefig(views_output_path)
        plt.close(fig)
        print(f'rendered views saved to {views_output_path}')
    else:
        for e in elev_deg:
            for a in azim_deg:
                rendered.append((e, a, render_one_view(renderer, render_mesh, e, a)))

    if source_image_path is not None:
        source_pil = Image.open(source_image_path).convert('RGB')
        target_h = source_pil.height

        panels = [source_pil]
        for _, _, img in rendered:
            panel = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            scale = target_h / panel.height
            panel = panel.resize((max(1, int(panel.width * scale)), target_h))
            panels.append(panel)

        total_w = sum(p.width for p in panels)
        combined = Image.new('RGB', (total_w, target_h), (255, 255, 255))
        x = 0
        for p in panels:
            combined.paste(p, (x, 0))
            x += p.width

        resolved_side_by_side_path = side_by_side_path or os.path.join(
            save_dir, f'{ply_basename}_side_by_side.png')
        combined.save(resolved_side_by_side_path)
        print(f'side-by-side (source image | {len(rendered)} rendered views) saved to '
              f'{resolved_side_by_side_path}')


def main(args):
    mesh = loadmesh(args.obj_path, name='guitar')
    renderer = Renderer(dim=(args.render_res, args.render_res))

    if args.ply_paths:
        n = len(args.ply_paths)

        def item(lst, i):
            return lst[i] if lst else None

        for i in range(n):
            if n > 1:
                print(f'=== batch {i + 1}/{n}: {args.ply_paths[i]} ===')
            render_one(mesh, renderer, args.ply_paths[i], args.save_dirs[i],
                      args.elev_deg, args.azim_deg, args.save_rendered_views_grid,
                      item(args.output_paths, i), item(args.source_image_paths, i),
                      item(args.side_by_side_paths, i), item(args.click_markers_paths, i),
                      args.marker_sphere_radius, args.marker_sphere_resolution,
                      args.marker_min_radius_frac)
    else:
        render_one(mesh, renderer, args.ply_path, args.save_dir, args.elev_deg, args.azim_deg,
                  args.save_rendered_views_grid, args.output_path, args.source_image_path,
                  args.side_by_side_path, args.click_markers_path, args.marker_sphere_radius,
                  args.marker_sphere_resolution, args.marker_min_radius_frac)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_path', type=str, default=None,
                         help='colored mesh .ply from color_mesh_from_knn.py. Required unless using '
                              'batch mode (--ply_paths).')
    parser.add_argument('--save_dir', type=str, default=None,
                         help='Required unless using batch mode (--ply_paths/--save_dirs).')
    parser.add_argument('--ply_paths', type=str, default=None,
                         help='BATCH MODE: comma-separated .ply paths (e.g. one per training-step '
                              'checkpoint) to render, sharing ONE mesh load + Renderer construction '
                              'instead of repeating it per .ply. Must be paired 1:1 with --save_dirs; '
                              '--source_image_paths/--click_markers_paths/--output_paths/'
                              '--side_by_side_paths are also 1:1 with this if given at all (omit '
                              'entirely to skip that output for every item in the batch). Mutually '
                              'exclusive with --ply_path/--save_dir/--source_image_path/'
                              '--click_markers_path/--output_path/--side_by_side_path.')
    parser.add_argument('--save_dirs', type=str, default=None,
                         help='BATCH MODE: comma-separated save_dir paths, 1:1 with --ply_paths.')
    parser.add_argument('--source_image_paths', type=str, default=None,
                         help='BATCH MODE: comma-separated --source_image_path values, 1:1 with '
                              '--ply_paths.')
    parser.add_argument('--click_markers_paths', type=str, default=None,
                         help='BATCH MODE: comma-separated --click_markers_path values, 1:1 with '
                              '--ply_paths.')
    parser.add_argument('--side_by_side_paths', type=str, default=None,
                         help='BATCH MODE: comma-separated --side_by_side_path values, 1:1 with '
                              '--ply_paths.')
    parser.add_argument('--output_paths', type=str, default=None,
                         help='BATCH MODE: comma-separated --output_path values, 1:1 with --ply_paths.')
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--save_rendered_views_grid', action='store_true',
                         help='also save the <ply_basename>_rendered_views.png labeled-grid figure. '
                              'Off by default since --source_image_path already produces a side-by-side '
                              'with the same views; turn on only when the grid (with elev/azim labels '
                              'per panel) is specifically wanted.')
    parser.add_argument('--output_path', type=str, default=None,
                         help='rendered-views grid path, only used with --save_rendered_views_grid; '
                              'defaults to <save_dir>/<ply_basename>_rendered_views.png')
    parser.add_argument('--render_res', type=int, default=224)
    parser.add_argument('--elev_deg', nargs='+', type=float, default=[0],
                         help='elevation angles (degrees) for the rendered views')
    parser.add_argument('--azim_deg', nargs='+', type=float, default=[90],
                         help='azimuth angles (degrees) for the rendered views. Default single view at '
                              "90: for this mesh/obj's built-in orientation, azim=0/180 are degenerate "
                              'edge-on views of the guitar (kept fixed regardless of --ply_path/--k so '
                              'runs stay comparable). Pass multiple values to render more angles.')
    parser.add_argument('--source_image_path', type=str, default=None,
                         help='image to place on the left of the side-by-side output: the plain '
                              '<mode>_color.png (pca_color.png or position_color.png) when coloring from '
                              'all pixels, or the '
                              '*_clicked_pixels.png (from color_mesh_from_knn.py --num_pixels/'
                              '--pixel_indices) when coloring from a handful of selected pixels. If '
                              'given, saves a side-by-side PNG of this image next to all the rendered '
                              'views above, in order.')
    parser.add_argument('--side_by_side_path', type=str, default=None,
                         help='defaults to <save_dir>/<ply_basename>_side_by_side.png')
    parser.add_argument('--click_markers_path', type=str, default=None,
                         help='*_click_markers.pth from color_mesh_from_knn.py --num_pixels/'
                              '--pixel_indices (--k neighbors per pixel, same k as was painted into the '
                              '.ply); if given, merges a colored sphere per (pixel, neighbor) pair into '
                              'the rendered mesh, since a '
                              'single colored vertex is too small to see clearly in a 224x224 render. '
                              'Sphere size scales with that neighbor\'s similarity relative to the '
                              'pixel\'s own top-1 match (see --marker_min_radius_frac), so you can '
                              'visually inspect the KNN spread of a chosen pixel, not just its winner.')
    parser.add_argument('--marker_sphere_radius', type=float, default=0.04,
                         help='radius of a pixel\'s top-1 (best-match) click-marker sphere, in the '
                              'mesh\'s unit-bounding-sphere coordinate frame (i.e. relative to the whole '
                              'mesh spanning ~radius 1); weaker neighbors are drawn smaller than this')
    parser.add_argument('--marker_min_radius_frac', type=float, default=0.35,
                         help='smallest allowed sphere radius, as a fraction of --marker_sphere_radius, '
                              "for a pixel's least-similar shown neighbor (prevents it from shrinking to "
                              'invisible)')
    parser.add_argument('--marker_sphere_resolution', type=int, default=12,
                         help='mesh resolution (facets) of each click-marker sphere')

    args = parser.parse_args()

    if args.ply_paths:
        if args.ply_path or args.save_dir:
            parser.error('--ply_paths (batch mode) is mutually exclusive with --ply_path/--save_dir')
        if not args.save_dirs:
            parser.error('--save_dirs is required with --ply_paths')
        args.ply_paths = args.ply_paths.split(',')
        args.save_dirs = args.save_dirs.split(',')
        if len(args.save_dirs) != len(args.ply_paths):
            parser.error(f'--ply_paths has {len(args.ply_paths)} entries but --save_dirs has '
                         f'{len(args.save_dirs)} -- must match 1:1')
        for name in ('source_image_paths', 'click_markers_paths', 'side_by_side_paths', 'output_paths'):
            value = getattr(args, name)
            if value is not None:
                parts = value.split(',')
                if len(parts) != len(args.ply_paths):
                    parser.error(f'--{name} has {len(parts)} entries but --ply_paths has '
                                 f'{len(args.ply_paths)} -- must match 1:1 (or omit --{name} entirely)')
                setattr(args, name, parts)
    else:
        if not (args.ply_path and args.save_dir):
            parser.error('--ply_path and --save_dir are required unless using batch mode '
                         '(--ply_paths/--save_dirs)')
        args.ply_paths = None

    main(args)
