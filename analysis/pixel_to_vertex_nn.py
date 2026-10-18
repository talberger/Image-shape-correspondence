"""
pixel -> vertex nearest neighbours: for every pixel of a SAM-segmented object, find its nearest
neighbor (NN) vertex on the shape, by cosine similarity between the
pixel's per-pixel DINO feature and each vertex's per-vertex DINO
feature (from the trained Encoder in modules/encoder.py). Also records
the top-K nearest vertices per pixel (not just the top-1), so the mesh-coloring step can
both color the mesh from the top-1 match and inspect the KNN of a
chosen pixel.

Run:
cd $DISC_ROOT && python analysis/pixel_to_vertex_nn.py \
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


def process_one(vertex_features_path, output_path, features_in_mask, mask_idx_row, mask_idx_col,
                mask_shape, pixel_feature_dim, top_k, chunk_size, device, features_path, mask_path):
    """One (vertex_features_path -> output_path) match, given an already-loaded,
    already-masked pixel feature set. Factored out of main() so the batch path
    below (many vertex-feature checkpoints against one pixel-feature file) and
    the single-file path share identical logic -- the single-file path is just
    a batch of size 1, not a separately maintained copy.
    """
    n_masked = features_in_mask.shape[0]
    vertex_features = load_vertex_features(vertex_features_path, pixel_feature_dim)
    num_vertices = vertex_features.shape[0]

    if top_k < 1 or top_k > num_vertices:
        raise ValueError(f'top_k={top_k} must be in [1, {num_vertices}]')

    topk_idx, topk_sim = compute_topk_nn(features_in_mask, vertex_features, top_k, chunk_size, device)

    top1_idx = topk_idx[:, 0]
    top1_sim = topk_sim[:, 0]
    unique_verts = torch.unique(top1_idx)
    num_unique = unique_verts.numel()

    print(f'n_masked pixels: {n_masked}')
    print(f'pixel feature dim: {pixel_feature_dim}, vertex feature dim: {vertex_features.shape[1]}, '
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
        'top_k': top_k,
        'num_vertices': num_vertices,
        'mask_shape': tuple(mask_shape),
        # Provenance: without this the file is indistinguishable from one built on
        # a different encoder, and the only way to tell was to recompute the
        # similarities against each candidate pred_f.pth and see which matched.
        'vertex_features_path': vertex_features_path,
        'features_path': features_path,
        'mask_path': mask_path,
    }
    torch.save(output, output_path)
    print(f'saved to {output_path}')


def main(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if args.save_dir is None and args.features_path is None:
        raise ValueError('--features_path is required when --save_dir is not given '
                         '(save_dir is only used to derive its default)')
    if args.save_dir is None and not args.vertex_features_paths and args.output_path is None:
        raise ValueError('--output_path is required when --save_dir is not given, unless '
                         'using batch mode (--output_paths)')

    img_name = get_img_name(args.img_path)

    features_path = args.features_path or os.path.join(args.save_dir, f'{img_name}_pix_dino_features.pth')
    features = load_features(features_path)

    mask = load_mask(args.mask_path)

    if features.shape[:2] != mask.shape:
        raise ValueError(f'features spatial shape {tuple(features.shape[:2])} '
                          f'!= mask shape {tuple(mask.shape)}')

    if args.chunk_size < 1:
        raise ValueError(f'chunk_size must be >= 1, got {args.chunk_size}')

    # Batch mode: many vertex-feature checkpoints (e.g. one per training step)
    # matched against the SAME pixel features/mask in one process, so the
    # multi-GB pixel feature file (loaded above, once) is never re-read from
    # disk per checkpoint the way running this script once per checkpoint
    # would. Off by default -- single-file mode below is untouched byte for
    # byte, so every existing run_*_analysis.sh driver keeps working as-is.
    if args.vertex_features_paths or args.output_paths:
        if not (args.vertex_features_paths and args.output_paths):
            raise ValueError('--vertex_features_paths and --output_paths must both be given together')
        vertex_features_paths = args.vertex_features_paths.split(',')
        output_paths = args.output_paths.split(',')
        if len(vertex_features_paths) != len(output_paths):
            raise ValueError(f'--vertex_features_paths has {len(vertex_features_paths)} entries but '
                              f'--output_paths has {len(output_paths)} -- must match 1:1')

        features_in_mask, mask_idx_row, mask_idx_col = get_features_in_mask(features, mask)
        n_masked = features_in_mask.shape[0]
        if n_masked == 0:
            warnings.warn('mask has zero True pixels; nothing to match')
            for vfp, out_path in zip(vertex_features_paths, output_paths):
                vertex_features_for_count = load_vertex_features(vfp, features.shape[-1])
                torch.save(empty_output(mask.shape, args.top_k, vertex_features_for_count.shape[0]), out_path)
                print(f'saved to {out_path}')
            return

        for i, (vfp, out_path) in enumerate(zip(vertex_features_paths, output_paths)):
            print(f'=== batch {i + 1}/{len(vertex_features_paths)}: {vfp} -> {out_path} ===')
            process_one(vfp, out_path, features_in_mask, mask_idx_row, mask_idx_col, mask.shape,
                       features.shape[-1], args.top_k, args.chunk_size, device, features_path, args.mask_path)
        return

    mask_basename = os.path.splitext(os.path.basename(args.mask_path))[0]
    output_path = args.output_path or os.path.join(args.save_dir, f'{mask_basename}_pixel_to_vertex_nn.pth')

    features_in_mask, mask_idx_row, mask_idx_col = get_features_in_mask(features, mask)
    n_masked = features_in_mask.shape[0]

    if n_masked == 0:
        warnings.warn('mask has zero True pixels; nothing to match')
        vertex_features_for_count = load_vertex_features(args.vertex_features_path, features.shape[-1])
        output = empty_output(mask.shape, args.top_k, vertex_features_for_count.shape[0])
        torch.save(output, output_path)
        print(f'saved to {output_path}')
        return

    process_one(args.vertex_features_path, output_path, features_in_mask, mask_idx_row, mask_idx_col, mask.shape,
               features.shape[-1], args.top_k, args.chunk_size, device, features_path, args.mask_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./images/guitar_01.png',
                         help='used only to derive the default pixel features filename')
    parser.add_argument('--save_dir', type=str, default=None,
                         help='required unless --features_path AND (--output_path or '
                              '--output_paths) are all given explicitly, since it is only '
                              'used to derive those defaults')
    parser.add_argument('--features_path', type=str, default=None,
                         help='defaults to <save_dir>/<img_name>_pix_dino_features.pth')
    parser.add_argument('--mask_path', type=str, required=True)
    parser.add_argument('--vertex_features_path', type=str, default=None,
                         help='per-vertex DINO features from modules/encoder.py, shape [V, F]. '
                              'Required unless --vertex_features_paths (batch mode) is used -- '
                              'which encoder a run used should always be explicit, never silently '
                              'picked by a default.')
    parser.add_argument('--output_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_pixel_to_vertex_nn.pth')
    parser.add_argument('--vertex_features_paths', type=str, default=None,
                         help='BATCH MODE: comma-separated list of vertex-feature files (e.g. one '
                              'per training-step checkpoint) to match against the SAME pixel '
                              'features/mask in this one process -- the multi-GB --features_path '
                              'file is loaded once and reused, instead of once per invocation of '
                              'this script. Must be paired 1:1 with --output_paths. Mutually '
                              'exclusive with --vertex_features_path/--output_path.')
    parser.add_argument('--output_paths', type=str, default=None,
                         help='BATCH MODE: comma-separated output paths, 1:1 with '
                              '--vertex_features_paths.')
    parser.add_argument('--top_k', type=int, default=100,
                         help='number of nearest vertices to keep per pixel')
    parser.add_argument('--chunk_size', type=int, default=20000,
                         help='number of masked pixels processed per similarity-matmul chunk')
    parser.add_argument('--device', type=str, default=None,
                         help='defaults to cuda if available else cpu')

    args = parser.parse_args()
    if not (args.vertex_features_paths or args.output_paths) and not args.vertex_features_path:
        parser.error('--vertex_features_path is required unless using batch mode '
                     '(--vertex_features_paths/--output_paths)')
    main(args)
