"""
mesh coloring: color the mesh using pixel_coloring.py's per-pixel colors (`pixel_coloring.py`, either
`--mode pca` or `--mode position`) and pixel_to_vertex_nn.py's per-pixel top-K nearest-vertex
correspondences. For every masked pixel, its pixel color is painted onto its
top-K nearest vertices (from pixel_to_vertex_nn.py), weighted by cosine similarity. A vertex
touched by multiple (pixel, neighbor-rank) contributions gets their
similarity-weighted average; a vertex touched by none is colored neutral
gray. Output is an ASCII .ply, viewable in MeshLab.

Run (pca colors, pixel_coloring.py's default mode):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/color_mesh_from_knn.py \
  --mask_path ./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2.pth --k 1

Or with pixel_coloring.py's --mode position colors instead (e.g. as a correspondence sanity
check independent of the DINO features: smooth position colors should stay
roughly smooth across the mesh if the feature-based correspondence respects
spatial locality):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/color_mesh_from_knn.py \
  --mask_path ./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2.pth --k 1 \
  --color_mode position

To paint from only a handful of pixels (e.g. to trace exactly where each one
lands, rather than an all-pixels average that hides individual contributions):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/color_mesh_from_knn.py \
  --mask_path ./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2.pth --k 1 \
  --num_pixels 5 --seed 0
This also saves a `*_clicked_pixels.png`: the original photo with each of the
5 chosen pixels marked as a small circle in its own pixel color, so you can see
where on the real photo they came from alongside the mesh they painted.
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils import write_ply
from modules.mesh import Mesh


def check_mask_matches_knn(mask_path, knn, knn_path):
    """pixel_to_vertex_nn.py already stores exactly which (row, col) pixels it was
    computed from. Load the mask given now and check it marks the same pixels --
    catches a stale knn file computed against a different/older mask (this exact
    bug produced mismatched pixel selections across bsb_retrained/bsb_mapping_loss
    on 2026-08-08, silently, because nothing checked this before)."""
    mask = torch.load(mask_path, map_location='cpu')
    mask_row, mask_col = torch.where(mask)
    same = (mask_row.shape == knn['pixel_row'].shape
            and torch.equal(mask_row, knn['pixel_row'])
            and torch.equal(mask_col, knn['pixel_col']))
    if not same:
        raise ValueError(f'{knn_path} was computed from a different mask than --mask_path={mask_path} '
                          f'({mask_row.shape[0]} masked pixels now vs {knn["pixel_row"].shape[0]} stored) -- '
                          f're-run pixel_to_vertex_nn.py against the current mask')


def load_knn(knn_path, k, mask_path):
    knn = torch.load(knn_path, map_location='cpu')
    top_k = knn['top_k']
    if not (1 <= k <= top_k):
        raise ValueError(f'k={k} must be in [1, {top_k}] (top_k stored in {knn_path})')
    check_mask_matches_knn(mask_path, knn, knn_path)
    return knn


def load_pixel_colors(color_path, mask_shape, pixel_row, pixel_col):
    img = np.array(Image.open(color_path).convert('RGB'))
    if tuple(img.shape[:2]) != tuple(mask_shape):
        raise ValueError(f'color image shape {tuple(img.shape[:2])} != mask shape {tuple(mask_shape)}')
    return torch.from_numpy(img[pixel_row.numpy(), pixel_col.numpy()]).float()


def accumulate_vertex_colors(topk_vertex_idx, topk_similarity, pixel_colors, k, num_vertices,
                              softmax_temp=None):
    vertex_idx = topk_vertex_idx[:, :k].reshape(-1)
    sim_k = topk_similarity[:, :k]
    if softmax_temp is not None:
        # per-pixel softmax over its own k neighbors, so each pixel's weights sum to 1
        # before being scattered into the vertex accumulators below.
        weight = torch.softmax(sim_k / softmax_temp, dim=1).reshape(-1)
    else:
        weight = sim_k.reshape(-1)
    color = pixel_colors.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3)

    vertex_color_sum = torch.zeros(num_vertices, 3, dtype=torch.float32)
    vertex_weight_sum = torch.zeros(num_vertices, dtype=torch.float32)
    vertex_color_sum.index_add_(0, vertex_idx, color * weight.unsqueeze(1))
    vertex_weight_sum.index_add_(0, vertex_idx, weight)

    return vertex_color_sum, vertex_weight_sum


def save_clicked_pixels_image(img_path, pixel_row, pixel_col, colors, output_path,
                               radius=8, outline_width=2):
    """Draws one filled circle per (row, col) in its own `colors` RGB, with a
    black outline, on top of the original (non-PCA) photo. Lets you see
    exactly where each hand-picked pixel sits and what color it painted."""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    for i in range(len(pixel_row)):
        r, c = int(pixel_row[i]), int(pixel_col[i])
        color = tuple(int(round(x)) for x in colors[i])
        bbox = [c - radius, r - radius, c + radius, r + radius]
        draw.ellipse(bbox, fill=color, outline=(0, 0, 0), width=outline_width)
    img.save(output_path)


def select_pixels(args, n_masked):
    """Returns (selected_idx or None, selection_desc or None).
    selected_idx is a 0-based index array into the masked-pixel arrays;
    None means "use all masked pixels" (previous, default behavior)."""
    if args.pixel_indices is not None:
        selected_idx = np.array(sorted(set(args.pixel_indices)), dtype=np.int64)
        if selected_idx.min() < 0 or selected_idx.max() >= n_masked:
            raise ValueError(f'--pixel_indices must be in [0, {n_masked - 1}]')
        return selected_idx, f'explicit{len(selected_idx)}'
    if args.num_pixels is not None:
        if not (1 <= args.num_pixels <= n_masked):
            raise ValueError(f'--num_pixels must be in [1, {n_masked}]')
        rng = np.random.RandomState(args.seed)
        selected_idx = np.sort(rng.choice(n_masked, size=args.num_pixels, replace=False))
        return selected_idx, f'random{args.num_pixels}_seed{args.seed}'
    return None, None


_MESH_CACHE = {}


def _load_mesh(obj_path):
    """Cache the mesh per process so batch mode pays the load once, not once per checkpoint."""
    if obj_path not in _MESH_CACHE:
        _MESH_CACHE[obj_path] = Mesh(obj_path)
    return _MESH_CACHE[obj_path]


def main(args):
    mask_basename = os.path.splitext(os.path.basename(args.mask_path))[0]
    knn_path = args.knn_path or os.path.join(args.save_dir, f'{mask_basename}_pixel_to_vertex_nn.pth')
    color_path = args.color_path or os.path.join(
        args.save_dir, f'{mask_basename}_{args.color_mode}_color.png')

    knn = load_knn(knn_path, args.k, args.mask_path)
    num_vertices = knn['num_vertices']
    n_masked = knn['pixel_row'].shape[0]

    if n_masked == 0:
        raise ValueError(f'{knn_path} has zero masked pixels; nothing to color')

    selected_idx, selection_desc = select_pixels(args, n_masked)

    if args.output_path is not None:
        output_path = args.output_path
    elif selection_desc is not None:
        output_path = os.path.join(
            args.save_dir, f'{mask_basename}_mesh_color_{args.color_mode}_k{args.k}_pix_{selection_desc}.ply')
    else:
        output_path = os.path.join(args.save_dir, f'{mask_basename}_mesh_color_{args.color_mode}_k{args.k}.ply')

    if selected_idx is not None:
        selected_idx_t = torch.from_numpy(selected_idx)
        pixel_row = knn['pixel_row'][selected_idx_t]
        pixel_col = knn['pixel_col'][selected_idx_t]
        topk_vertex_idx = knn['topk_vertex_idx'][selected_idx_t]
        topk_similarity = knn['topk_similarity'][selected_idx_t]
    else:
        pixel_row = knn['pixel_row']
        pixel_col = knn['pixel_col']
        topk_vertex_idx = knn['topk_vertex_idx']
        topk_similarity = knn['topk_similarity']

    pixel_colors = load_pixel_colors(color_path, knn['mask_shape'], pixel_row, pixel_col)

    if selected_idx is not None:
        print(f'selected {len(selected_idx)}/{n_masked} masked pixels via {selection_desc}:')
        for local_i, orig_i in enumerate(selected_idx):
            r, c = int(pixel_row[local_i]), int(pixel_col[local_i])
            top1_v = int(topk_vertex_idx[local_i, 0])
            top1_s = float(topk_similarity[local_i, 0])
            print(f'  pixel #{orig_i} (row={r}, col={c}) -> top1 vertex {top1_v} (sim={top1_s:.4f})')

        clicked_pixels_path = args.clicked_pixels_output_path or os.path.join(
            args.save_dir,
            f'{mask_basename}_mesh_color_{args.color_mode}_k{args.k}_pix_{selection_desc}_clicked_pixels.png')
        save_clicked_pixels_image(args.img_path, pixel_row, pixel_col, pixel_colors.numpy(),
                                   clicked_pixels_path, radius=args.marker_radius,
                                   outline_width=args.marker_outline_width)
        print(f'clicked-pixels image saved to {clicked_pixels_path}')

        click_markers_path = args.click_markers_output_path or os.path.join(
            args.save_dir,
            f'{mask_basename}_mesh_color_{args.color_mode}_k{args.k}_pix_{selection_desc}_click_markers.pth')
        torch.save({
            'vertex_idx': topk_vertex_idx[:, :args.k].long(),  # [N_selected, k]
            'similarity': topk_similarity[:, :args.k],         # [N_selected, k]
            'color': pixel_colors,                             # [N_selected, 3], shared across a pixel's neighbors
        }, click_markers_path)
        print(f'click markers ({args.k} nearest vertices + similarity + color per selected pixel) '
              f'saved to {click_markers_path}')

    softmax_temp = args.softmax_temp if args.use_softmax else None
    vertex_color_sum, vertex_weight_sum = accumulate_vertex_colors(
        topk_vertex_idx, topk_similarity, pixel_colors, args.k, num_vertices,
        softmax_temp=softmax_temp)

    covered = vertex_weight_sum > 0
    num_covered = int(covered.sum().item())

    vertex_colors = np.tile(np.array(args.uncovered_color, dtype=np.float32), (num_vertices, 1))
    avg_color = (vertex_color_sum[covered] / vertex_weight_sum[covered].unsqueeze(1)).numpy()
    vertex_colors[covered.numpy()] = avg_color
    vertex_colors = np.clip(np.round(vertex_colors), 0, 255).astype(np.int32)

    mesh = _load_mesh(args.obj_path)
    verts = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    if verts.shape[0] != num_vertices:
        raise ValueError(f'mesh has {verts.shape[0]} vertices, expected {num_vertices} (from {knn_path})')

    n_painted = len(selected_idx) if selected_idx is not None else n_masked
    print(f'n_masked pixels: {n_masked}, n_painted from: {n_painted}, k: {args.k}')
    raw_sim = topk_similarity[:, :args.k]
    print(f'raw similarity: mean={raw_sim.mean().item():.4f} '
          f'min={raw_sim.min().item():.4f} max={raw_sim.max().item():.4f}')
    if softmax_temp is not None:
        actual_weight = torch.softmax(raw_sim / softmax_temp, dim=1)
        print(f'weighting: softmax(similarity / {softmax_temp:g}) per pixel over its own k neighbors '
              f'-- weight mean={actual_weight.mean().item():.4f} min={actual_weight.min().item():.4f} '
              f'max={actual_weight.max().item():.4f}')
    else:
        print('weighting: raw similarity (no softmax)')
    print(f'vertex coverage: {num_covered}/{num_vertices} vertices ({100.0 * num_covered / num_vertices:.2f}%)')

    write_ply(output_path, verts, faces, vertex_colors)
    print(f'saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=str, required=True,
                         help='used only to derive default paths below (same mask the pixel coloring and NN steps were run on)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='required unless --save_dirs (batch mode) is given')
    parser.add_argument('--knn_paths', type=str, default=None,
                        help='BATCH MODE: comma-separated --knn_path values. Pair with --save_dirs '
                             '(same length). One process, mesh loaded once -- the step-progression '
                             'script called this 12 times and paid 12 cold starts (~5.5 min).')
    parser.add_argument('--save_dirs', type=str, default=None,
                        help='BATCH MODE: comma-separated --save_dir values, one per --knn_paths entry.')
    parser.add_argument('--knn_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_pixel_to_vertex_nn.pth (pixel_to_vertex_nn.py output)')
    parser.add_argument('--color_mode', type=str, choices=['pca', 'position'], default='pca',
                         help="which of pixel_coloring.py's --mode outputs to paint the mesh with; "
                              'also selects the default --color_path filename')
    parser.add_argument('--color_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_<color_mode>_color.png '
                              '(pixel_coloring.py output)')
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--output_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_mesh_color_<color_mode>_k<k>.ply')
    parser.add_argument('--k', type=int, default=1,
                         help='number of nearest vertices per pixel to paint (must be <= top_k stored in the pixel_to_vertex_nn.py output)')
    parser.add_argument('--use_softmax', action='store_true',
                         help='instead of using each pixel\'s raw cosine similarities as weights '
                              'directly, take softmax(similarity / --softmax_temp) over the pixel\'s '
                              'own k neighbors first, so weights sum to 1 per pixel before scattering '
                              'into vertices. Off by default (raw similarity weighting, original '
                              'behavior).')
    parser.add_argument('--softmax_temp', type=float, default=1.0,
                         help='only used with --use_softmax. 1.0 is the neutral/standard choice '
                              '(no extra scaling beyond the softmax itself) -- lower sharpens toward '
                              'the single best-matching neighbor, higher flattens toward a uniform '
                              'average over the k neighbors.')
    parser.add_argument('--uncovered_color', nargs=3, type=int, default=[200, 200, 200],
                         help='RGB (0-255) for vertices with zero contributing pixels')
    parser.add_argument('--num_pixels', type=int, default=None,
                         help='if set, paint the mesh from only this many masked pixels instead of all of '
                              'them (selected per --pixel_selection); every other vertex is left uncovered. '
                              'Ignored if --pixel_indices is given.')
    parser.add_argument('--pixel_selection', type=str, default='random', choices=['random'],
                         help='how to pick --num_pixels pixels (only "random" for now)')
    parser.add_argument('--pixel_indices', type=int, nargs='+', default=None,
                         help='explicit 0-based indices into the masked-pixel array to paint from, instead '
                              'of --num_pixels/--pixel_selection (e.g. to re-inspect a specific pixel)')
    parser.add_argument('--seed', type=int, default=0,
                         help='random seed for --pixel_selection random')
    parser.add_argument('--img_path', type=str, default='./images/guitar_01.png',
                         help='original (uncolored) photo to draw --num_pixels/--pixel_indices click '
                              'markers on; only used when pixel-subsampling is active')
    parser.add_argument('--clicked_pixels_output_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_mesh_color_<color_mode>_k<k>_pix_'
                              '<selection>_clicked_pixels.png; only produced when pixel-subsampling is active')
    parser.add_argument('--click_markers_output_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_mesh_color_<color_mode>_k<k>_pix_'
                              '<selection>_click_markers.pth ({vertex_idx, similarity, color} per selected pixel, '
                              'top-k neighbors each -- same k as --k, so the markers always match '
                              'exactly what got painted into the .ply; for '
                              'render_colored_mesh.py --click_markers_path); only produced when '
                              'pixel-subsampling is active')
    parser.add_argument('--marker_radius', type=int, default=16,
                         help='radius in pixels of each clicked-pixel marker circle')
    parser.add_argument('--marker_outline_width', type=int, default=3,
                         help='outline width in pixels of each clicked-pixel marker circle')

    args = parser.parse_args()
    if not args.knn_paths and not args.save_dir:
        parser.error('--save_dir is required (or use --knn_paths/--save_dirs for batch mode)')
    if args.knn_paths or args.save_dirs:
        if not (args.knn_paths and args.save_dirs):
            parser.error('--knn_paths and --save_dirs must be given together')
        knn_paths = [x for x in args.knn_paths.split(',') if x]
        save_dirs = [x for x in args.save_dirs.split(',') if x]
        if len(knn_paths) != len(save_dirs):
            parser.error(f'--knn_paths has {len(knn_paths)} entries but --save_dirs has {len(save_dirs)}')
        for i, (kp, sd) in enumerate(zip(knn_paths, save_dirs), 1):
            print(f'=== batch {i}/{len(knn_paths)}: {kp} -> {sd} ===')
            args.knn_path, args.save_dir = kp, sd
            os.makedirs(sd, exist_ok=True)
            main(args)
    else:
        main(args)
