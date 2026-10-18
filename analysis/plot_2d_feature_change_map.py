"""Where do the features change? Per-pixel map of delta = 1 - cos(f_p, f_q), q at --steps pixels from p, averaged over
the 8 directions (in-mask q only). Step 1 is the cost field of plot_2d_feature_geodesic_paths.py; a larger step
compares patch against patch. Each map is divided by its own median, so 1 = a typical pixel.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_geodesic_tables import largest_connected_component
from plot_2d_feature_geodesic_paths import STEPS, load_features


def change_map(feats, mask, step):
    H, W = mask.shape
    total, count = np.zeros((H, W)), np.zeros((H, W))
    rows, cols = np.nonzero(mask)
    for dr, dc in STEPS:
        nr, nc = rows + dr * step, cols + dc * step
        ok = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        ok[ok] = mask[nr[ok], nc[ok]]
        d = 1.0 - (feats[rows[ok], cols[ok]] * feats[nr[ok], nc[ok]]).sum(1)
        np.add.at(total, (rows[ok], cols[ok]), np.clip(d, 0.0, None))
        np.add.at(count, (rows[ok], cols[ok]), 1)
    out = np.full((H, W), np.nan)
    out[count > 0] = total[count > 0] / count[count > 0]
    return out


def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    img = np.array(Image.open(args.image_path).convert('RGB'))
    mask, _ = largest_connected_component(torch.load(args.object_mask_path, map_location='cpu').numpy().astype(bool))
    feats = load_features(args.features_path, *mask.shape)
    points = [tuple(args.points[i:i + 2]) for i in range(0, len(args.points), 2)]
    fig, axes = plt.subplots(1, 1 + len(args.steps), squeeze=False,
                             figsize=((1 + len(args.steps)) * img.shape[1] / 62, img.shape[0] / 62 + 0.6), dpi=110)
    axes[0, 0].imshow(img)
    axes[0, 0].contour(mask, [0.5], colors='yellow', linewidths=0.8)
    axes[0, 0].set_title('photo', fontsize=9)
    for ax, step in zip(axes[0, 1:], args.steps):
        m = change_map(feats, mask, step)
        med = float(np.nanmedian(m))
        rel = m / med
        print(f'step {step}: median delta {med:.2e}, relative p90 {np.nanpercentile(rel, 90):.2f}, '
              f'p99 {np.nanpercentile(rel, 99):.2f}, max {np.nanmax(rel):.2f}')
        ax.imshow(img, alpha=0.25)
        im = ax.imshow(rel, cmap='magma', vmin=0.0, vmax=float(np.nanpercentile(rel, 99.5)))
        ax.set_title(f'feature change over {step} px (/ median {med:.1e})', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes[0]:
        for i, p in enumerate(points):
            ax.plot(p[1], p[0], '*' if i == 0 else 'o', ms=14 if i == 0 else 8, mfc='cyan' if i == 0 else 'w', mec='k', mew=1.2)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(args.title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output_path)
    print(f'saved {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--object_mask_path', type=str, required=True)
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--points', type=int, nargs='+', required=True, help='row col row col ... (the first one is the centre)')
    parser.add_argument('--steps', type=int, nargs='+', required=True)
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--output_path', type=str, required=True)
    main(parser.parse_args())
