"""
mesh coloring, alternative method to color_mesh_from_knn.py: instead of painting a pixel's
color directly onto its top-K feature-nearest vertices, first reproduce the model's own soft
mapping (modules/mapping_loss.py's soft_map_to_vertices -- DPC Eq. 1-3, the same function
mapping_loss()/relational_loss() train against): a softmax-weighted average, over the pixel's
top-K feature-nearest vertices, of their POSITIONS -- giving one continuous 3D point y_hat per
pixel, generally not coincident with any actual vertex. Then hard-snap y_hat to whichever mesh
vertex is geometrically nearest to it (Euclidean distance, in the same bounding-sphere-normalized
space loadmesh()/training uses), and paint that single vertex with the pixel's color.

A pixel can therefore land on a vertex that was never among its own top-K feature neighbours --
this is a genuinely different vertex assignment from color_mesh_from_knn.py's, not just a
different weighting of the same one. Vertices touched by multiple pixels get an unweighted
average of their colors: the geometric snap distance is not a similarity score, so there is no
principled weight to use (unlike color_mesh_from_knn.py's cosine-similarity weighting).

Run:
cd $DISC_ROOT && python analysis/color_mesh_from_soft_mapping.py \
  --mask_path ./demo/guitar/bsb_origin/guitar_01_click_130_140_sam_mask_2.pth \
  --features_path ./demo/guitar/bsb_origin/guitar_01_pix_dino_features.pth \
  --vertex_features_path ./experiments/guitar/encoder_dino_relational_bbox_norm/pred_f_step300.pth \
  --save_dir ./demo/guitar/bsb_relational_bbox_norm/analyze \
  --color_mode position \
  --color_path ./demo/guitar/bsb_retrained/pixel_coloring/guitar_01_click_130_140_sam_mask_2_position_color.png
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.color_mesh_from_knn import load_pixel_colors
from analysis.pixel_to_vertex_nn import load_features, load_mask, load_vertex_features
from modules.mapping_loss import soft_map_to_vertices
from modules.mesh import Mesh
from modules.utils import get_features_in_mask, get_img_name, loadmesh, write_ply


def compute_soft_snap(features_in_mask, vertex_features, vertex_positions, k, chunk_size, device, temperature=1.0):
    """For each masked pixel: soft-map to a continuous 3D point (soft_map_to_vertices), then
    hard-snap to the nearest actual vertex by Euclidean distance. Chunked over pixels -- the
    per-chunk [chunk, V] distance matrix is the dominant allocation, same reason
    pixel_to_vertex_nn.py chunks its [chunk, V] similarity matrix.

    `vertex_positions` must be the SAME space soft_map_to_vertices used to build y_hat (loadmesh's
    bounding-sphere-normalized positions, matching training) -- both the averaging and the snap
    have to agree on what "nearest" means.
    """
    pixel_feats = F.normalize(features_in_mask.detach().to(device).float(), dim=-1)
    vert_feats = vertex_features.to(device)
    vert_pos = vertex_positions.to(device).float()

    n = pixel_feats.shape[0]
    nearest_vertex_idx = torch.empty(n, dtype=torch.long)
    nearest_distance = torch.empty(n, dtype=torch.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        print(f'soft-mapping pixels {start + 1} to {end} out of {n}')
        y_hat = soft_map_to_vertices(pixel_feats[start:end], vert_feats, vert_pos, k,
                                      temperature=temperature)  # [chunk, 3]
        dist2 = (y_hat.unsqueeze(1) - vert_pos.unsqueeze(0)).pow(2).sum(-1)            # [chunk, V]
        nearest_d2, nearest_idx = dist2.min(dim=1)
        nearest_vertex_idx[start:end] = nearest_idx.cpu()
        nearest_distance[start:end] = nearest_d2.clamp(min=0).sqrt().cpu()

    return nearest_vertex_idx, nearest_distance


def accumulate_vertex_colors(nearest_vertex_idx, pixel_colors, num_vertices):
    vertex_color_sum = torch.zeros(num_vertices, 3, dtype=torch.float32)
    vertex_count = torch.zeros(num_vertices, dtype=torch.float32)
    vertex_color_sum.index_add_(0, nearest_vertex_idx, pixel_colors)
    vertex_count.index_add_(0, nearest_vertex_idx, torch.ones(nearest_vertex_idx.shape[0]))
    return vertex_color_sum, vertex_count


def main(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    mask_basename = os.path.splitext(os.path.basename(args.mask_path))[0]
    img_name = get_img_name(args.img_path)

    features_path = args.features_path or os.path.join(args.save_dir, f'{img_name}_pix_dino_features.pth')
    features = load_features(features_path)
    mask = load_mask(args.mask_path)
    if features.shape[:2] != mask.shape:
        raise ValueError(f'features spatial shape {tuple(features.shape[:2])} != mask shape {tuple(mask.shape)}')

    vertex_features = load_vertex_features(args.vertex_features_path, features.shape[-1])
    num_vertices = vertex_features.shape[0]
    if not (1 <= args.k_vertices_per_pixel <= num_vertices):
        raise ValueError(f'--k_vertices_per_pixel={args.k_vertices_per_pixel} must be in [1, {num_vertices}]')

    color_path = args.color_path or os.path.join(args.save_dir, f'{mask_basename}_{args.color_mode}_color.png')
    output_path = args.output_path or os.path.join(
        args.save_dir, f'{mask_basename}_mesh_color_soft_snap_{args.color_mode}_k{args.k_vertices_per_pixel}.ply')

    features_in_mask, mask_idx_row, mask_idx_col = get_features_in_mask(features, mask)
    n_masked = features_in_mask.shape[0]
    if n_masked == 0:
        raise ValueError(f'{args.mask_path} has zero True pixels; nothing to color')

    # Normalized mesh: the space soft_map_to_vertices/training's y_hat live in, and where the
    # nearest-vertex snap below has to happen for the two to agree. `mesh.vertices` after
    # loadmesh() is in-place bounding-sphere normalized -- vertex order/count are untouched, only
    # positions are, so indices below still line up with the raw mesh used for the .ply output.
    norm_mesh = loadmesh(dir=args.obj_path, name=args.name)
    if norm_mesh.vertices.shape[0] != num_vertices:
        raise ValueError(f'mesh has {norm_mesh.vertices.shape[0]} vertices, expected {num_vertices} '
                          f'(from {args.vertex_features_path})')

    nearest_vertex_idx, nearest_distance = compute_soft_snap(
        features_in_mask, vertex_features, norm_mesh.vertices, args.k_vertices_per_pixel,
        args.chunk_size, device, temperature=args.temperature)

    # Persist the per-pixel decision in pixel_to_vertex_nn.py's own output shape (top_k=1: this
    # method assigns exactly one vertex per pixel, no top-K to store) so
    # color_image_from_knn.py's back-projection works against it completely unmodified via
    # --knn_path. There is no cosine similarity here (the snap is geometric nearest-vertex, not
    # feature similarity), so 'similarity' holds exp(-nearest_distance): positive, decreasing in
    # distance, bounded in (0, 1] -- a stand-in "confidence" that never triggers
    # color_image_from_knn.py's near-zero-weight fallback. It has no effect on the back-projected
    # color itself (k=1 averages a single term regardless of its weight); it only shows up in
    # that script's own diagnostic printout.
    correspondence_path = args.correspondence_output_path or os.path.join(
        args.save_dir, f'{mask_basename}_soft_snap_correspondence.pth')
    confidence = torch.exp(-nearest_distance)
    torch.save({
        'pixel_row': mask_idx_row,
        'pixel_col': mask_idx_col,
        'top1_vertex_idx': nearest_vertex_idx,
        'top1_similarity': confidence,
        'topk_vertex_idx': nearest_vertex_idx.unsqueeze(1),
        'topk_similarity': confidence.unsqueeze(1),
        'top_k': 1,
        'num_vertices': num_vertices,
        'mask_shape': tuple(mask.shape),
        'nearest_distance': nearest_distance,
        'k_vertices_per_pixel': args.k_vertices_per_pixel,
        'vertex_features_path': args.vertex_features_path,
        'features_path': features_path,
        'mask_path': args.mask_path,
    }, correspondence_path)
    print(f'per-pixel vertex correspondence saved to {correspondence_path}')

    pixel_colors = load_pixel_colors(color_path, tuple(mask.shape), mask_idx_row, mask_idx_col)

    vertex_color_sum, vertex_count = accumulate_vertex_colors(nearest_vertex_idx, pixel_colors, num_vertices)
    covered = vertex_count > 0
    num_covered = int(covered.sum().item())

    vertex_colors = np.tile(np.array(args.uncovered_color, dtype=np.float32), (num_vertices, 1))
    avg_color = (vertex_color_sum[covered] / vertex_count[covered].unsqueeze(1)).numpy()
    vertex_colors[covered.numpy()] = avg_color
    vertex_colors = np.clip(np.round(vertex_colors), 0, 255).astype(np.int32)

    # Raw (unnormalized) mesh for the .ply geometry -- same convention as color_mesh_from_knn.py
    # / render_colored_mesh.py, so this output renders at the same scale as everything else.
    raw_mesh = Mesh(args.obj_path)
    verts = raw_mesh.vertices.detach().cpu().numpy()
    faces = raw_mesh.faces.detach().cpu().numpy()
    if verts.shape[0] != num_vertices:
        raise ValueError(f'mesh has {verts.shape[0]} vertices, expected {num_vertices}')

    print(f'n_masked pixels: {n_masked}, k_vertices_per_pixel: {args.k_vertices_per_pixel}')
    print(f'snap distance (soft-mapped point -> nearest actual vertex, normalized space): '
          f'mean={nearest_distance.mean().item():.4f} min={nearest_distance.min().item():.4f} '
          f'max={nearest_distance.max().item():.4f}')
    print(f'vertex coverage: {num_covered}/{num_vertices} vertices ({100.0 * num_covered / num_vertices:.2f}%)')

    write_ply(output_path, verts, faces, vertex_colors)
    print(f'saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=str, required=True)
    parser.add_argument('--img_path', type=str, default='./images/guitar_01.png',
                         help='used only to derive the default pixel-features filename')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--features_path', type=str, default=None,
                         help='defaults to <save_dir>/<img_name>_pix_dino_features.pth')
    parser.add_argument('--vertex_features_path', type=str, required=True,
                         help='per-vertex DINO features from modules/encoder.py, shape [V, F] (a pred_f.pth)')
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')
    parser.add_argument('--color_mode', type=str, choices=['pca', 'position'], default='position',
                         help="which of pixel_coloring.py's --mode outputs to paint the mesh with; "
                              'also selects the default --color_path filename')
    parser.add_argument('--color_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_<color_mode>_color.png (pixel_coloring.py output)')
    parser.add_argument('--output_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_mesh_color_soft_snap_<color_mode>_'
                              'k<k_vertices_per_pixel>.ply')
    parser.add_argument('--correspondence_output_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_soft_snap_correspondence.pth -- the '
                              'per-pixel nearest-vertex decision, in pixel_to_vertex_nn.py\'s own output '
                              'shape (top_k=1), so analysis/color_image_from_knn.py can back-project '
                              'through it via --knn_path unmodified')
    parser.add_argument('--k_vertices_per_pixel', type=int, default=10,
                         help="how many feature-nearest vertices' positions get softmax-averaged into "
                              'y_hat per pixel (DPC Eq. 2-3 / soft_map_to_vertices). 10 matches the default '
                              'used across every phase2/train_*.py script, so this reproduces the same y_hat '
                              'training itself optimizes against.')
    parser.add_argument('--temperature', type=float, default=1.0,
                         help='softmax temperature for the soft mapping (see '
                               'modules/mapping_loss.py:soft_map_to_vertices). 1.0 = the plain '
                               'softmax every previous coverage number in this project used.')
    parser.add_argument('--chunk_size', type=int, default=20000,
                         help='number of masked pixels processed per chunk (soft-mapping + nearest-vertex search)')
    parser.add_argument('--uncovered_color', nargs=3, type=int, default=[200, 200, 200],
                         help='RGB (0-255) for vertices with zero contributing pixels')
    parser.add_argument('--device', type=str, default=None, help='defaults to cuda if available else cpu')

    args = parser.parse_args()
    main(args)
