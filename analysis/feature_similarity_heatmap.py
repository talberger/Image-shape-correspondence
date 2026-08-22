"""
Cosine-similarity heatmap: pick one or more query points -- pixels in the
masked image, vertices on the mesh, or both -- and for each, compute cosine
similarity from its DINO feature to every OTHER point in the SAME domain
(masked pixels for a pixel query, mesh vertices for a vertex query; no
cross-domain pixel<->vertex coupling), visualized as a heatmap with the
query point itself marked.

Mirrors pca_feature_compare.py's --display split: image queries and vertex
queries are two independent, same-domain analyses, not a correspondence
check. Unlike a global PCA coloring, this shows the *whole* similarity
distribution for one specific query -- the diagnostic for ambiguity: a
symmetric point (e.g. left vs. right horn of the guitar) can show two
well-separated near-equal peaks instead of one dominant peak.

Every query point given (pixel and/or vertex) is combined into ONE grid
image: top row = image-domain heatmaps (query pixel marked with a ring),
bottom row = mesh-domain heatmaps (query vertex marked with a small sphere,
rendered). No separate per-query PNG is kept -- only the combined grid PNG,
plus each vertex query's .ply/.pth (the actual colored-mesh data, not just
a view).

Run (2 pixel queries + 2 vertex queries -> one combined grid):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/feature_similarity_heatmap.py \
  --query_pixels 151 94 571 206 --query_vertices 2760 1746 \
  --save_dir ./demo/guitar/bsb_frontcentered_ZR/features_analyze/sim_heatmap/ \
  --mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --vertex_features_path ./experiments/guitar/encoder_dino_frontcentered/pred_f.pth

--pixel_indices (indices into the np.where(mask)-sorted pixel array, same
convention as color_mesh_from_knn.py --pixel_indices) is also accepted for
pixel queries, e.g. to reuse the project's explicit9 tracked-pixel set, and
combines with --query_pixels if both are given.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pixel_coloring import load_features, load_mask, build_output_image
from analysis.mesh_coloring import load_vertex_features
from analysis.render_colored_mesh import add_click_marker_spheres, render_one_view
from modules.mesh import Mesh
from modules.utils import (get_img_name, get_features_in_mask, write_ply, loadmesh,
                            path_provenance_tag, setcolor_mesh_batched, device)
from modules.render import Renderer

QUERY_MARKER_COLOR = (255, 0, 0)


def cosine_similarity_to_target(query_feature, target_features):
    """query_feature: [F]. target_features: [N, F]. Returns sim: [N] in
    [-1, 1] -- same L2-normalize + dot product pixel_to_vertex_nn.py's
    compute_topk_nn uses, without the top-k truncation: here the whole
    distribution is the point, not just the best match(es)."""
    q = F.normalize(query_feature.unsqueeze(0), p=2, dim=-1)
    t = F.normalize(target_features, p=2, dim=-1)
    return (q @ t.T).squeeze(0)


def similarity_to_rgb(similarity, vmin, vmax, colormap):
    sim = similarity.detach().cpu().numpy()
    vmin = float(sim.min()) if vmin is None else vmin
    vmax = float(sim.max()) if vmax is None else vmax
    norm = np.clip((sim - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    cmap = matplotlib.cm.get_cmap(colormap)
    rgba = cmap(norm)
    return (rgba[:, :3] * 255).astype(np.uint8), vmin, vmax


def find_separated_peaks(similarity, positions, k, min_separation):
    """Greedy top-k local maxima at least min_separation apart (Euclidean, in
    `positions`' own units -- pixels for the image domain, mesh-vertex
    coordinates for the mesh domain). Two peaks surviving this filter close
    in similarity is the concrete symmetry-ambiguity signal; one dominant
    peak is a clean match."""
    sim = similarity.detach().cpu().numpy()
    order = np.argsort(-sim)
    peaks = []
    for idx in order:
        pos = positions[idx]
        if all(np.linalg.norm(pos - positions[p]) >= min_separation for p in peaks):
            peaks.append(int(idx))
        if len(peaks) >= k:
            break
    return peaks


def process_pixel_query(row, col, args, mask, mask_idx_row, mask_idx_col, features_in_mask):
    """Image-domain self-similarity heatmap for one query pixel. Returns
    (panel: PIL.Image, label: str) for the combined grid; also saves a .pth
    with the raw similarity data (no standalone PNG -- only the grid keeps a
    picture)."""
    query_idx = ((mask_idx_row == row) & (mask_idx_col == col)).nonzero()
    if query_idx.numel() == 0:
        raise ValueError(f'--query_pixels ({row}, {col}) is not inside the mask')
    query_feature = features_in_mask[query_idx[0, 0]]

    positions = np.stack([mask_idx_row.numpy(), mask_idx_col.numpy()], axis=1).astype(np.float32)
    similarity = cosine_similarity_to_target(query_feature, features_in_mask)
    colors, vmin, vmax = similarity_to_rgb(similarity, args.vmin, args.vmax, args.colormap)
    peaks = find_separated_peaks(similarity, positions, 3, args.min_peak_separation_px)

    img = build_output_image(mask.shape, mask_idx_row, mask_idx_col, colors,
                              args.background_color, args.transparent_background)
    panel = Image.fromarray(img).convert('RGB')
    draw = ImageDraw.Draw(panel)
    r = args.marker_dot_radius_px
    draw.ellipse([col - r, row - r, col + r, row + r], fill=QUERY_MARKER_COLOR)

    tag = f'pixel_row{row}_col{col}'
    print(f'[{tag} -> image] similarity: min={similarity.min():.4f} max={similarity.max():.4f} '
          f'mean={similarity.mean():.4f} (colormap range [{vmin:.4f}, {vmax:.4f}])')
    for rank, idx in enumerate(peaks):
        p_row, p_col = positions[idx]
        print(f'  peak {rank + 1}: (row={int(p_row)}, col={int(p_col)}), similarity={similarity[idx]:.4f}')

    mask_basename = os.path.splitext(os.path.basename(args.mask_path))[0]
    data_path = os.path.join(args.save_dir, f'{mask_basename}_simheatmap_{tag}_data.pth')
    os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)
    torch.save({
        'similarity': similarity.detach().cpu(),
        'query_row': row, 'query_col': col,
        'vmin': vmin, 'vmax': vmax,
        'features_path': args.features_path,
    }, data_path)
    print(f'  data saved to {data_path}')

    return panel, tag


def process_vertex_query(idx, args, verts, faces, vertex_features, render_mesh, renderer):
    """Mesh-domain self-similarity heatmap for one query vertex. Writes the
    real colored-mesh data (.ply/.pth, unmarked, indexed by original vertex
    ids) and returns (panel: PIL.Image, label: str) -- a small marker sphere
    is merged in only for the rendered grid panel, never baked into the
    saved .ply/.pth (same convention render_colored_mesh.py's own
    --click_markers_path already uses)."""
    similarity = cosine_similarity_to_target(vertex_features[idx], vertex_features)
    colors, vmin, vmax = similarity_to_rgb(similarity, args.vmin, args.vmax, args.colormap)
    bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    min_separation = args.min_peak_separation_frac * bbox_diag
    peaks = find_separated_peaks(similarity, verts, 3, min_separation)

    tag = f'vertex_idx{idx}'
    print(f'[{tag} -> mesh] similarity: min={similarity.min():.4f} max={similarity.max():.4f} '
          f'mean={similarity.mean():.4f} (colormap range [{vmin:.4f}, {vmax:.4f}])')
    for rank, p_idx in enumerate(peaks):
        print(f'  peak {rank + 1}: vertex {p_idx} at {tuple(round(float(c), 4) for c in verts[p_idx])}, '
              f'similarity={similarity[p_idx]:.4f}')

    obj_name = os.path.splitext(os.path.basename(args.obj_path))[0]
    output_path = os.path.join(args.save_dir, f'{obj_name}_simheatmap_{tag}_to_mesh_color.ply')
    colors_output_path = os.path.splitext(output_path)[0] + '.pth'
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    write_ply(output_path, verts, faces, colors.astype(np.int32))
    torch.save({
        'vertex_colors': torch.from_numpy(colors),
        'similarity': similarity.detach().cpu(),
        'query_vertex_idx': idx,
        'vmin': vmin, 'vmax': vmax,
        'obj_path': args.obj_path,
        'vertex_features_path': args.vertex_features_path,
    }, colors_output_path)
    print(f'  saved to {output_path} (colors: {colors_output_path})')

    marker_path = os.path.join(args.save_dir, f'_tmp_marker_{tag}.pth')
    torch.save({
        'vertex_idx': torch.tensor([[idx]], dtype=torch.long),
        'similarity': torch.tensor([[1.0]], dtype=torch.float32),
        'color': torch.tensor([QUERY_MARKER_COLOR], dtype=torch.float32),
    }, marker_path)
    marked_mesh, combined_colors = add_click_marker_spheres(
        render_mesh, colors.astype(np.float64) / 255.0, marker_path,
        sphere_radius=args.marker_sphere_radius, sphere_resolution=args.marker_sphere_resolution,
        min_radius_frac=args.marker_min_radius_frac)
    os.remove(marker_path)
    setcolor_mesh_batched(marked_mesh, torch.from_numpy(combined_colors).float().to(device).unsqueeze(0))
    rendered = render_one_view(renderer, marked_mesh, args.elev_deg[0], args.azim_deg[0])
    panel = Image.fromarray((np.clip(rendered, 0, 1) * 255).astype(np.uint8))

    return panel, tag


def build_grid(image_panels, image_labels, mesh_panels, mesh_labels, output_path):
    """Two independent 2-row blocks side by side -- image-domain queries on the
    left, mesh-domain queries on the right -- rather than one row of images
    stacked directly over one row of meshes. Stacking would visually suggest
    each image column corresponds to the mesh column below it, which isn't
    true: the two domains are queried independently (no pixel<->vertex
    coupling)."""
    nrows = 2
    img_cols = max(1, (len(image_panels) + nrows - 1) // nrows) if image_panels else 0
    mesh_cols = max(1, (len(mesh_panels) + nrows - 1) // nrows) if mesh_panels else 0
    ncols = img_cols + mesh_cols
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 4.5),
                             squeeze=False, constrained_layout=True)
    for ax in axs.flat:
        ax.axis('off')

    for i, (panel, label) in enumerate(zip(image_panels, image_labels)):
        r, c = divmod(i, img_cols)
        axs[r, c].imshow(panel)
        axs[r, c].set_title(label, fontsize=8)

    for i, (panel, label) in enumerate(zip(mesh_panels, mesh_labels)):
        r, c = divmod(i, mesh_cols)
        axs[r, img_cols + c].imshow(panel)
        axs[r, img_cols + c].set_title(label, fontsize=8)

    fig.savefig(output_path)
    plt.close(fig)
    print(f'combined grid ({len(image_panels)} image + {len(mesh_panels)} mesh panels) '
          f'saved to {output_path}')


def main(args):
    pixel_queries = []
    if args.query_pixels:
        if len(args.query_pixels) % 2 != 0:
            raise ValueError('--query_pixels must be given in ROW COL pairs')
        pixel_queries.extend(zip(args.query_pixels[0::2], args.query_pixels[1::2]))

    need_pixel_features = bool(pixel_queries) or bool(args.pixel_indices)
    need_vertex_features = bool(args.query_vertices)

    if need_pixel_features and args.mask_path is None:
        raise ValueError('--mask_path is required for --query_pixels/--pixel_indices')
    if need_vertex_features and args.vertex_features_path is None:
        raise ValueError('--vertex_features_path is required for --query_vertices '
                          '(no default on purpose)')
    if not need_pixel_features and not need_vertex_features:
        raise ValueError('give at least one of --query_pixels/--pixel_indices/--query_vertices')

    mask = mask_idx_row = mask_idx_col = features_in_mask = None
    if need_pixel_features:
        mask = load_mask(args.mask_path)
        img_name = get_img_name(args.img_path)
        features_path = args.features_path or os.path.join(args.save_dir, f'{img_name}_pix_dino_features.pth')
        pixel_features = load_features(features_path)
        print(f'pixel features: {features_path} (tag: {path_provenance_tag(features_path)})')
        if pixel_features.shape[:2] != mask.shape:
            raise ValueError(f'features spatial shape {tuple(pixel_features.shape[:2])} '
                              f'!= mask shape {tuple(mask.shape)}')
        features_in_mask, mask_idx_row, mask_idx_col = get_features_in_mask(pixel_features, mask)

        if args.pixel_indices:
            n_masked = features_in_mask.shape[0]
            for i in args.pixel_indices:
                if not (0 <= i < n_masked):
                    raise ValueError(f'--pixel_indices entry {i} out of range [0, {n_masked})')
                pixel_queries.append((int(mask_idx_row[i]), int(mask_idx_col[i])))

    verts = faces = vertex_features = None
    render_mesh = renderer = None
    if need_vertex_features:
        mesh = Mesh(args.obj_path)
        verts = mesh.vertices.detach().cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy()
        vertex_features = load_vertex_features(args.vertex_features_path, verts.shape[0])
        print(f'vertex features: {args.vertex_features_path} '
              f'(tag: {path_provenance_tag(args.vertex_features_path)})')
        for idx in args.query_vertices:
            if not (0 <= idx < vertex_features.shape[0]):
                raise ValueError(f'--query_vertices entry {idx} out of range [0, {vertex_features.shape[0]})')
        # Built once, reused across every vertex query -- mesh load + Renderer
        # construction is the ~120s kaolin JIT cost.
        render_mesh = loadmesh(args.obj_path, name='guitar')
        renderer = Renderer(dim=(args.render_res, args.render_res))

    image_panels, image_labels = [], []
    for row, col in pixel_queries:
        panel, tag = process_pixel_query(row, col, args, mask, mask_idx_row, mask_idx_col, features_in_mask)
        image_panels.append(panel)
        image_labels.append(tag)

    mesh_panels, mesh_labels = [], []
    for idx in args.query_vertices or []:
        panel, tag = process_vertex_query(idx, args, verts, faces, vertex_features, render_mesh, renderer)
        mesh_panels.append(panel)
        mesh_labels.append(tag)

    grid_output_path = args.grid_output_path or os.path.join(args.save_dir, 'simheatmap_grid.png')
    os.makedirs(os.path.dirname(os.path.abspath(grid_output_path)), exist_ok=True)
    build_grid(image_panels, image_labels, mesh_panels, mesh_labels, grid_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_pixels', nargs='+', type=int, default=None,
                         metavar='ROW/COL', help='flat list of ROW COL pairs, e.g. '
                              '--query_pixels 151 94 571 206 for two pixel queries')
    parser.add_argument('--pixel_indices', nargs='+', type=int, default=None,
                         help='additional pixel queries via indices into the '
                              'np.where(mask)-sorted pixel array (same convention as '
                              'color_mesh_from_knn.py --pixel_indices) -- combines with '
                              '--query_pixels if both are given')
    parser.add_argument('--query_vertices', nargs='+', type=int, default=None,
                         help='vertex indices to query, each rendered as its own panel')
    parser.add_argument('--img_path', type=str, default='./images/guitar_01.png',
                         help='used only to derive the default pixel features filename')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--features_path', type=str, default=None,
                         help='defaults to <save_dir>/<img_name>_pix_dino_features.pth')
    parser.add_argument('--mask_path', type=str, default=None,
                         help='required for --query_pixels/--pixel_indices')
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--vertex_features_path', type=str, default=None,
                         help='required for --query_vertices. No default on purpose.')
    parser.add_argument('--vmin', type=float, default=None,
                         help='colormap lower bound. Default: auto (min of each query\'s own '
                              'similarity values) for max local contrast per panel.')
    parser.add_argument('--vmax', type=float, default=None)
    parser.add_argument('--colormap', type=str, default='viridis',
                         help='matplotlib sequential colormap name -- sequential, not diverging, '
                              'since this project\'s similarity values sit in a narrow high band, '
                              'not spread around 0.')
    parser.add_argument('--min_peak_separation_px', type=float, default=50.0,
                         help='pixel queries only. Minimum pixel distance between reported peaks.')
    parser.add_argument('--min_peak_separation_frac', type=float, default=0.15,
                         help='vertex queries only. Minimum distance between reported peaks, as a '
                              'fraction of the mesh bounding-box diagonal.')
    parser.add_argument('--render_res', type=int, default=224, help='vertex queries only')
    parser.add_argument('--elev_deg', nargs='+', type=float, default=[0],
                         help='vertex queries only. Only the first value is used -- one rendered '
                              'view per grid panel.')
    parser.add_argument('--azim_deg', nargs='+', type=float, default=[90],
                         help='vertex queries only. Default 90, this project\'s confirmed front '
                              'angle for meshes/guitar.obj. Only the first value is used.')
    parser.add_argument('--marker_dot_radius_px', type=int, default=8,
                         help='pixel queries only. Radius (px) of the filled dot marking the query '
                              'pixel on its heatmap panel.')
    parser.add_argument('--marker_sphere_radius', type=float, default=0.02,
                         help='vertex queries only. Radius of the query-vertex marker sphere, in '
                              "the mesh's unit-bounding-sphere coordinate frame.")
    parser.add_argument('--marker_sphere_resolution', type=int, default=12, help='vertex queries only')
    parser.add_argument('--marker_min_radius_frac', type=float, default=1.0, help='vertex queries only')
    parser.add_argument('--grid_output_path', type=str, default=None,
                         help='defaults to <save_dir>/simheatmap_grid.png')
    parser.add_argument('--background_color', nargs=3, type=int, default=[255, 255, 255])
    parser.add_argument('--transparent_background', type=int, default=None,
                         help='if set (0-255), heatmap panels use this alpha for background pixels')

    args = parser.parse_args()
    main(args)
