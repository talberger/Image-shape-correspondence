"""
SMALL per-arm analysis in ONE process (Tal, 2026-09-15). Replaces the SMALL path of
scripts/analyze_isobbox_arm.sh, which ran six separate python processes and re-loaded the 8 GB photo
feature file ([1080, 810, 2304] float32) in five of them.

Stages, same outputs and the same numbers as the old stages (they call the same functions):
  1. back-projection top1k        pixel_to_vertex_nn.compute_topk_nn -> color_image_from_knn (k=1)
  2. soft-snap + coverage         color_mesh_from_soft_mapping.compute_soft_snap -> back-projection k=1
  3. similarity heatmap, ONE ROW  feature_similarity_heatmap (grid_rows 1 -> simheatmap_row.png), the
                                  strip compare_isobbox_arms.sh stacks -- it no longer recomputes it
  4. eval v2                      eval_geodesic_error.main, fed the in-mask features from memory
Dropped from SMALL by Tal's call: PCA, the 2x2 heatmap grid, and the 28 MB *_pixel_to_vertex_nn.pth
(nothing in the small path reads it; FULL mode still produces it).

The photo features are read through a cache of the MASK pixels only (212,549 x 2304 float32, ~2 GB),
written next to the feature file on first use. float32 is kept on purpose so every number is
bit-identical to the old path (top-1/top-2 similarity margins are ~0.003 here).
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.eval_geodesic_error as evalmod
import analysis.feature_similarity_heatmap as heatmod
from analysis.color_image_from_knn import back_project_colors, load_vertex_colors, save_side_by_side
from analysis.color_mesh_from_soft_mapping import compute_soft_snap
from analysis.pixel_coloring import build_output_image
from analysis.pixel_to_vertex_nn import compute_topk_nn, load_mask, load_vertex_features
from modules.mesh import Mesh
from modules.render import Renderer
from modules.utils import get_features_in_mask, loadmesh

T0 = time.time()


def stamp(msg):
    print(f'[{time.time() - T0:6.1f}s] {msg}', flush=True)


def load_inmask_features(features_path, mask_path, mask):
    """[N, F] float32 features of the mask pixels, in torch.where(mask) order, via an on-disk cache."""
    mb = os.path.splitext(os.path.basename(mask_path))[0]
    cache = os.path.splitext(features_path)[0] + f'_inmask_{mb}.pth'
    src_mtime = os.path.getmtime(features_path)
    if os.path.exists(cache):
        c = torch.load(cache, map_location='cpu')
        # Same source file (mtime) AND exactly the same mask pixels, not just the same count --
        # a mask file replaced under the same name must invalidate the cache.
        row_now, col_now = torch.where(mask)
        if c.get('source_mtime') == src_mtime and tuple(c['mask_shape']) == tuple(mask.shape) \
                and c['row'].shape == row_now.shape and bool((c['row'] == row_now).all()) \
                and bool((c['col'] == col_now).all()):
            stamp(f'in-mask feature cache: {cache}')
            return c['features'], c['row'], c['col'], c['feature_shape']
        stamp(f'cache {cache} is stale, rebuilding')
    full = torch.load(features_path, map_location='cpu')
    if tuple(full.shape[:2]) != tuple(mask.shape):
        raise ValueError(f'features {tuple(full.shape[:2])} != mask {tuple(mask.shape)}')
    feats, row, col = get_features_in_mask(full, mask)
    feats = feats.contiguous()
    shape = tuple(full.shape)
    del full
    torch.save({'features': feats, 'row': row, 'col': col, 'mask_shape': tuple(mask.shape),
                'feature_shape': shape, 'source_mtime': src_mtime, 'features_path': features_path}, cache)
    stamp(f'in-mask feature cache written: {cache} ({feats.shape[0]} x {feats.shape[1]})')
    return feats, row, col, shape


class InMaskFeatures:
    """Stands in for the full [H, W, F] tensor where eval_geodesic_error indexes features[rows, cols]."""

    def __init__(self, feats, row, col, shape):
        self.shape = shape
        self._feats = feats
        self._index = torch.full(shape[:2], -1, dtype=torch.long)
        self._index[row, col] = torch.arange(row.shape[0])

    def __getitem__(self, key):
        rows, cols = key
        idx = self._index[torch.as_tensor(rows).long(), torch.as_tensor(cols).long()]
        if (idx < 0).any():
            raise ValueError('eval pixel outside the mask -- the in-mask cache cannot serve it')
        return self._feats[idx]


def save_backprojection(out_png, mask_shape, row, col, vertex_colors, topk_idx, topk_sim, render_path):
    colors = back_project_colors(vertex_colors, topk_idx, topk_sim, 1)
    img = build_output_image(mask_shape, row, col,
                             np.clip(np.round(colors.numpy()), 0, 255).astype(np.uint8),
                             [255, 255, 255], None)
    side = os.path.splitext(out_png)[0] + '_side_by_side.png'
    save_side_by_side(render_path, img, side, [255, 255, 255])
    return side


def main(a):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mb = os.path.splitext(os.path.basename(a.mask_path))[0]
    chk_stem = os.path.splitext(os.path.basename(a.checker_path))[0]
    checker_render = os.path.splitext(a.checker_path)[0] + '_rendered_views.png'
    bp_dir = os.path.join(a.save_dir, 'back_projection')
    soft_dir = os.path.join(a.save_dir, 'color_mesh_from_soft_mapping')
    sim_dir = os.path.join(a.save_dir, 'features_analyze', 'sim_heatmap')
    for d in (bp_dir, soft_dir, sim_dir):
        os.makedirs(d, exist_ok=True)

    mask = load_mask(a.mask_path)
    feats, row, col, fshape = load_inmask_features(a.features_path, a.mask_path, mask)
    vertex_features = load_vertex_features(a.vertex_features_path, feats.shape[1])
    num_vertices = vertex_features.shape[0]
    vertex_colors, _ = load_vertex_colors(a.checker_path, num_vertices)
    stamp(f'loaded: {feats.shape[0]} mask pixels, {num_vertices} vertices, pred_f {a.vertex_features_path}')

    # 1. back-projection top1k -- only k=1 is read, so top_k=1
    topk_idx, topk_sim = compute_topk_nn(feats, vertex_features, 1, a.chunk_size, device)
    p = save_backprojection(os.path.join(bp_dir, f'{mb}_backproj_{chk_stem}_k1.png'), tuple(mask.shape),
                            row, col, vertex_colors, topk_idx, topk_sim, checker_render)
    stamp(f'[1/4] back-projection top1k: {p}')

    # 2. soft-snap (read T) + coverage
    norm_mesh = loadmesh(dir=a.obj_path, name=a.name)
    nearest, dist = compute_soft_snap(feats, vertex_features, norm_mesh.vertices, a.k_vertices_per_pixel,
                                      a.chunk_size, device, temperature=a.temperature)
    confidence = torch.exp(-dist)
    torch.save({'pixel_row': row, 'pixel_col': col, 'top1_vertex_idx': nearest, 'top1_similarity': confidence,
                'topk_vertex_idx': nearest.unsqueeze(1), 'topk_similarity': confidence.unsqueeze(1), 'top_k': 1,
                'num_vertices': num_vertices, 'mask_shape': tuple(mask.shape), 'nearest_distance': dist,
                'k_vertices_per_pixel': a.k_vertices_per_pixel, 'vertex_features_path': a.vertex_features_path,
                'features_path': a.features_path, 'mask_path': a.mask_path},
               os.path.join(soft_dir, f'{mb}_soft_snap_correspondence.pth'))
    p = save_backprojection(os.path.join(bp_dir, f'{mb}_backproj_{chk_stem}_soft_snap_k1.png'),
                            tuple(mask.shape), row, col, vertex_colors, nearest.unsqueeze(1),
                            confidence.unsqueeze(1), checker_render)
    k = int(torch.unique(nearest).numel())
    print(f'soft-snap coverage (read T={a.temperature:g}): {k}/{num_vertices} = {100 * k / num_vertices:.2f}%')
    stamp(f'[2/4] soft-snap back-projection: {p}')

    # 3. similarity heatmap, one row
    h = SimpleNamespace(vmin=None, vmax=None, colormap='viridis', min_peak_separation_px=50.0,
                        min_peak_separation_frac=0.15, background_color=[255, 255, 255],
                        transparent_background=None, marker_dot_radius_px=8, marker_sphere_radius=0.02,
                        marker_sphere_resolution=12, marker_min_radius_frac=1.0, elev_deg=[0], azim_deg=[90],
                        mask_path=a.mask_path, features_path=a.features_path, save_dir=sim_dir,
                        obj_path=a.obj_path, vertex_features_path=a.vertex_features_path)
    img_panels, img_labels = [], []
    qp = a.query_pixels
    for r, c in zip(qp[0::2], qp[1::2]):
        panel, tag = heatmod.process_pixel_query(r, c, h, mask, row, col, feats)
        img_panels.append(panel); img_labels.append(tag)
    mesh = Mesh(a.obj_path)
    verts, faces = mesh.vertices.detach().cpu().numpy(), mesh.faces.detach().cpu().numpy()
    render_mesh = loadmesh(a.obj_path, name='guitar')
    renderer = Renderer(dim=(224, 224))
    mesh_panels, mesh_labels = [], []
    for idx in a.query_vertices:
        panel, tag = heatmod.process_vertex_query(idx, h, verts, faces, vertex_features, render_mesh, renderer)
        mesh_panels.append(panel); mesh_labels.append(tag)
    heatmod.build_grid(img_panels, img_labels, mesh_panels, mesh_labels,
                       os.path.join(sim_dir, 'simheatmap_row.png'), nrows=1)
    stamp('[3/4] similarity heatmap row')

    # 4. eval v2
    if a.eval_save_path:
        evalmod.load_features = lambda _path: InMaskFeatures(feats, row, col, fshape)
        evalmod.main(SimpleNamespace(
            replot_from=None, eval_pairs_path=a.eval_pairs_path, features_path=a.features_path,
            vertex_features_path=a.vertex_features_path, run_label=a.run_label, geo_table_path=a.geo_table_path,
            obj_path=a.obj_path, name=a.name, k_vertices_per_pixel=a.k_vertices_per_pixel,
            temperature=a.temperature, eps_max=0.20, eps_step=0.01, variant='A',
            plot_path=a.eval_plot_path, save_path=a.eval_save_path))
        stamp('[4/4] eval v2')
    stamp('SMALL ANALYSIS DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', required=True, help='the arm\'s analyze/ dir')
    ap.add_argument('--vertex_features_path', required=True)
    ap.add_argument('--features_path', required=True)
    ap.add_argument('--mask_path', required=True)
    ap.add_argument('--obj_path', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--checker_path', required=True)
    ap.add_argument('--query_pixels', nargs='+', type=int, required=True)
    ap.add_argument('--query_vertices', nargs='+', type=int, required=True)
    ap.add_argument('--k_vertices_per_pixel', type=int, default=10)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--chunk_size', type=int, default=20000)
    ap.add_argument('--eval_pairs_path', default=None)
    ap.add_argument('--geo_table_path', default=None)
    ap.add_argument('--eval_save_path', default=None)
    ap.add_argument('--eval_plot_path', default=None)
    ap.add_argument('--run_label', default='(unlabeled run)')
    main(ap.parse_args())
