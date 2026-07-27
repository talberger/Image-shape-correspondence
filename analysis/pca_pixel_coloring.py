"""
A1: color every pixel of a SAM-segmented object by its DINO feature,
reduced to RGB via PCA(3), following the pixel-coloring technique from
Amir et al., "Deep ViT Features as Dense Visual Descriptors"
(arXiv 2112.05814), code: github.com/ShirAmir/dino-vit-features/pca.py
(PCA fit directly on raw descriptors, then independent per-channel
min-max normalization to [0, 255]).

Run:
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/pca_pixel_coloring.py \
  --mask_path ./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2.pth
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

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


def pca_colors(features_in_mask, n_components, random_state):
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(features_in_mask.detach().numpy())
    print('explained_variance_ratio_:', pca.explained_variance_ratio_)

    colors = np.zeros_like(reduced, dtype=np.float32)
    for c in range(n_components):
        col = reduced[:, c]
        col_min, col_max = col.min(), col.max()
        col_range = col_max - col_min
        if col_range < 1e-12:
            warnings.warn(f'PCA channel {c} has ~zero variance; setting it to 0')
            continue
        colors[:, c] = (col - col_min) / col_range

    return (colors * 255).astype(np.uint8)


def build_output_image(shape_hw, mask_idx_row, mask_idx_col, pixel_colors,
                        background_color, transparent_background):
    h, w = shape_hw
    if transparent_background is not None:
        img = np.zeros((h, w, 4), dtype=np.uint8)
        img[:, :, :3] = background_color
        img[:, :, 3] = transparent_background
        img[mask_idx_row, mask_idx_col, :3] = pixel_colors
        img[mask_idx_row, mask_idx_col, 3] = 255
    else:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = background_color
        img[mask_idx_row, mask_idx_col] = pixel_colors

    return img


def main(args):
    img_name = get_img_name(args.img_path)

    features_path = args.features_path or os.path.join(args.save_dir, f'{img_name}_pix_dino_features.pth')
    features = load_features(features_path)

    mask = load_mask(args.mask_path)

    if features.shape[:2] != mask.shape:
        raise ValueError(f'features spatial shape {tuple(features.shape[:2])} '
                          f'!= mask shape {tuple(mask.shape)}')

    mask_basename = os.path.splitext(os.path.basename(args.mask_path))[0]
    output_path = args.output_path or os.path.join(args.save_dir, f'{mask_basename}_pca_color.png')

    features_in_mask, mask_idx_row, mask_idx_col = get_features_in_mask(features, mask)
    n_masked = features_in_mask.shape[0]

    transparent_background = None
    if args.transparent_background is not None:
        transparent_background = args.transparent_background

    if n_masked == 0:
        warnings.warn('mask has zero True pixels; writing a background-only image')
        img = build_output_image(mask.shape, mask_idx_row, mask_idx_col,
                                  np.zeros((0, 3), dtype=np.uint8),
                                  args.background_color, transparent_background)
        Image.fromarray(img).save(output_path)
        print(f'saved to {output_path}')
        return

    if n_masked < args.n_components:
        raise ValueError(f'mask has only {n_masked} True pixels, '
                          f'fewer than n_components={args.n_components}')

    pixel_colors = pca_colors(features_in_mask, args.n_components, args.random_state)

    img = build_output_image(mask.shape, mask_idx_row, mask_idx_col, pixel_colors,
                              args.background_color, transparent_background)
    Image.fromarray(img).save(output_path)
    print(f'saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./images/guitar_01.png',
                         help='used only to derive the default features filename')
    parser.add_argument('--save_dir', type=str, default='./demo/guitar/bsb/')
    parser.add_argument('--features_path', type=str, default=None,
                         help='defaults to <save_dir>/<img_name>_pix_dino_features.pth')
    parser.add_argument('--mask_path', type=str,
                         default='./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2.pth')
    parser.add_argument('--output_path', type=str, default=None,
                         help='defaults to <save_dir>/<mask_basename>_pca_color.png')
    parser.add_argument('--n_components', type=int, default=3)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--background_color', nargs=3, type=int, default=[255, 255, 255])
    parser.add_argument('--transparent_background', type=int, default=None,
                         help='if set (0-255), output is RGBA with this alpha for background pixels')

    args = parser.parse_args()
    main(args)
