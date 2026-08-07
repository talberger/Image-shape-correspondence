"""
pixel -> vertex nearest neighbours: for every pixel of a SAM-segmented object, find its nearest
neighbor (NN) vertex on the shape, by cosine similarity between the
pixel's per-pixel DINO feature and each vertex's per-vertex DINO
feature (from the trained Encoder in modules/encoder.py). Also records
the top-K nearest vertices per pixel (not just the top-1), so the mesh-coloring step can
both color the mesh from the top-1 match and inspect the KNN of a
chosen pixel.

Run:
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/pixel_to_vertex_nn.py \
  --mask_path ./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2.pth
"""

import argparse
import os
import sys
import warnings

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils import get_img_name, get_features_in_mask


def load_features(features_path):
    features = torch.load(features_path, map_location='cpu')
    if features.dtype != torch.float32:
        raise ValueError(f'expected float32 features, got {features.dtype}')
    if features.dim() != 3:
        raise ValueError(f'expected features of shape [H, W, F], got {tuple(features.shape)}')
    return features


def load_mask(mask_path):
    mask = torch.load(mask_path, map_location='cpu')
    if mask.dtype != torch.bool:
        raise ValueError(f'expected a bool mask, got {mask.dtype}')
    if mask.dim() != 2:
        raise ValueError(f'expected mask of shape [H, W], got {tuple(mask.shape)}')
    return mask


def load_vertex_features(vertex_features_path, expected_feature_dim):
    vertex_features = torch.load(vertex_features_path, map_location='cpu')
    if vertex_features.dtype != torch.float32:
        raise ValueError(f'expected float32 vertex features, got {vertex_features.dtype}')
    if vertex_features.dim() != 2:
        raise ValueError(f'expected vertex features of shape [V, F], got {tuple(vertex_features.shape)}')
    if vertex_features.shape[1] != expected_feature_dim:
        raise ValueError(f'vertex feature dim {vertex_features.shape[1]} != '
                          f'pixel feature dim {expected_feature_dim}')
    return vertex_features.detach()


def compute_topk_nn(pixel_features_in_mask, vertex_features, top_k, chunk_size, device):
    pixel_feats = pixel_features_in_mask.detach().to(device)
    vert_feats = vertex_features.to(device)

    pixel_norm = F.normalize(pixel_feats, p=2, dim=-1)
    vert_norm = F.normalize(vert_feats, p=2, dim=-1)

    n = pixel_norm.shape[0]
    topk_sim = torch.empty(n, top_k, dtype=torch.float32)
    topk_idx = torch.empty(n, top_k, dtype=torch.long)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        print(f'computing similarity for pixels {start + 1} to {end} out of {n}')
        chunk_sim = pixel_norm[start:end] @ vert_norm.T
        chunk_vals, chunk_idx = torch.topk(chunk_sim, top_k, dim=1)
        topk_sim[start:end] = chunk_vals.cpu()
        topk_idx[start:end] = chunk_idx.cpu()

    return topk_idx, topk_sim


def empty_output(mask_shape, top_k, num_vertices):
    return {
        'pixel_row': torch.empty(0, dtype=torch.long),
        'pixel_col': torch.empty(0, dtype=torch.long),
        'top1_vertex_idx': torch.empty(0, dtype=torch.long),
        'top1_similarity': torch.empty(0, dtype=torch.float32),
        'topk_vertex_idx': torch.empty(0, top_k, dtype=torch.long),
        'topk_similarity': torch.empty(0, top_k, dtype=torch.float32),
        'top_k': top_k,
        'num_vertices': num_vertices,
        'mask_shape': tuple(mask_shape),
    }


def main(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    img_name = get_img_name(args.img_path)

    features_path = args.features_path or os.path.join(args.save_dir, f'{img_name}_pix_dino_features.pth')
    features = load_features(features_path)

    mask = load_mask(args.mask_path)

    if features.shape[:2] != mask.shape:
        raise ValueError(f'features spatial shape {tuple(features.shape[:2])} '
                          f'!= mask shape {tuple(mask.shape)}')

    mask_basename = os.path.splitext(os.path.basename(args.mask_path))[0]
    output_path = args.output_path or os.path.join(args.save_dir, f'{mask_basename}_pixel_to_vertex_nn.pth')

    features_in_mask, mask_idx_row, mask_idx_col = get_features_in_mask(features, mask)
    n_masked = features_in_mask.shape[0]

    if args.chunk_size < 1:
        raise ValueError(f'chunk_size must be >= 1, got {args.chunk_size}')

    if n_masked == 0:
        warnings.warn('mask has zero True pixels; nothing to match')
        vertex_features_for_count = load_vertex_features(args.vertex_features_path, features.shape[-1])
        output = empty_output(mask.shape, args.top_k, vertex_features_for_count.shape[0])
        torch.save(output, output_path)
        print(f'saved to {output_path}')
        return

    vertex_features = load_vertex_features(args.vertex_features_path, features.shape[-1])
    num_vertices = vertex_features.shape[0]

    if args.top_k < 1 or args.top_k > num_vertices:
        raise ValueError(f'top_k={args.top_k} must be in [1, {num_vertices}]')

    topk_idx, topk_sim = compute_topk_nn(features_in_mask, vertex_features, args.top_k, args.chunk_size, device)

    top1_idx = topk_idx[:, 0]
    top1_sim = topk_sim[:, 0]
    unique_verts = torch.unique(top1_idx)
    num_unique = unique_verts.numel()

    print(f'n_masked pixels: {n_masked}')
    print(f'pixel feature dim: {features.shape[-1]}, vertex feature dim: {vertex_features.shape[1]}, '
          f'num vertices: {num_vertices}')
    print(f'top-1 similarity: mean={top1_sim.mean().item():.4f} '
          f'min={top1_sim.min().item():.4f} max={top1_sim.max().item():.4f}')
    print(f'vertex coverage: {num_unique}/{num_vertices} vertices matched at least once '
          f'({100.0 * num_unique / num_vertices:.2f}%)')
    print(f'mean pixels per matched vertex: {n_masked / num_unique:.1f}')

    output = {
        'pixel_row': mask_idx_row,
        'pixel_col': mask_idx_col,
        'top1_vertex_idx': top1_idx,
        'top1_similarity': top1_sim,
        'topk_vertex_idx': topk_idx,
        'topk_similarity': topk_sim,
        'top_k': args.top_k,
        'num_vertices': num_vertices,
        'mask_shape': tuple(mask.shape),
        # Provenance: without this the file is indistinguishable from one built on
        # a different encoder, and the only way to tell was to recompute the
        # similarities against each candidate pred_f.pth and see which matched.
        'vertex_features_path': args.vertex_features_path,
        'features_path': features_path,
        'mask_path': args.mask_path,
    }
    torch.save(output, output_path)
    print(f'saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./images/guitar_01.png',
                         help='used only to derive the default pixel features filename')
    parser.add_argument('--save_dir', type=str, default='./demo/guitar/bsb/')
    parser.add_argument('--features_path', type=str, default=None,
                         help='defaults to <save_dir>/<img_name>_pix_dino_features.pth')
    parser.add_argument('--mask_path', type=str,
                         default='./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2.pth')
    parser.add_argument('--vertex_features_path', type=str,
                         default='./experiments/guitar/encoder_dino/pred_f.pth',
                         help='per-vertex DINO features from modules/encoder.py, shape [V, F]. '
                              'Defaults to OUR phase-1 retrain. It used to default to '
                              './demo/guitar/encoder_dino/pred_f.pth, which is the BSB authors\' '
                              'downloaded file (download_demo_data.sh:23-24) -- everything produced '
                              'before 2026-08-02 was therefore about their encoder, not ours.')
    parser.add_argument('--output_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_pixel_to_vertex_nn.pth')
    parser.add_argument('--top_k', type=int, default=10,
                         help='number of nearest vertices to keep per pixel')
    parser.add_argument('--chunk_size', type=int, default=20000,
                         help='number of masked pixels processed per similarity-matmul chunk')
    parser.add_argument('--device', type=str, default=None,
                         help='defaults to cuda if available else cpu')

    args = parser.parse_args()
    main(args)
