"""Setup figure for the crowded-parts study: the chosen pixels, their 2D geodesic paths, the feature map and one
feature-similarity heatmap per chosen pixel.

Top row:    photo + chosen pixels | paths, training metric (beta 0) | paths, feature step cost (--beta, --power) | feature PCA
Bottom row: cos(f_p, f_q) of every mask pixel p against each chosen pixel q (centre, then the neighbours).
The paths and their step cost are the ones of plot_2d_feature_geodesic_paths.py.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_geodesic_tables import largest_connected_component
from plot_2d_feature_geodesic_paths import load_features, mask_edges, pca_rgb

COLOURS = ['magenta', 'lime', 'orange', 'cyan', 'white', 'red']


def draw_points(ax, centre, nbrs):
    for i, nb in enumerate(nbrs):
        col = COLOURS[i % len(COLOURS)]
        ax.plot(nb[1], nb[0], 'o', ms=9, mfc=col, mec='k', mew=1.2)
        ax.text(nb[1] + 6, nb[0] - 6, f'N{i + 1}', color=col, fontsize=9, weight='bold',
                bbox=dict(fc='k', alpha=0.6, pad=1.0, lw=0))
    ax.plot(centre[1], centre[0], '*', ms=16, mfc='cyan', mec='k', mew=1.2)
    ax.text(centre[1] + 6, centre[0] - 6, 'C', color='cyan', fontsize=9, weight='bold',
            bbox=dict(fc='k', alpha=0.6, pad=1.0, lw=0))


def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    img = np.array(Image.open(args.image_path).convert('RGB'))
    mask, _ = largest_connected_component(torch.load(args.object_mask_path, map_location='cpu').numpy().astype(bool))
    feats = load_features(args.features_path, *mask.shape)
    centre = tuple(args.center)
    nbrs = [tuple(args.neighbors[i:i + 2]) for i in range(0, len(args.neighbors), 2)]
    points = [('C', centre)] + [(f'N{i + 1}', nb) for i, nb in enumerate(nbrs)]
    for _, p in points:
        if not mask[p]:
            raise ValueError(f'{p} (row, col) is not on the mask')

    src, dst, length, delta, p2n, rows, cols = mask_edges(mask, feats)
    rel = delta / float(np.median(delta))
    n, c = len(rows), p2n[centre]

    ncol = max(4, len(points))
    fig, axes = plt.subplots(2, ncol, squeeze=False, figsize=(ncol * img.shape[1] / 62, 2 * img.shape[0] / 62), dpi=110)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 0]
    ax.imshow(img)
    ax.contour(mask, [0.5], colors='yellow', linewidths=0.8)
    draw_points(ax, centre, nbrs)
    ax.set_title('chosen pixels (row, col)\n' + '\n'.join(f'{name} {p}' for name, p in points), fontsize=8)

    print('beta | power | neighbour | path px | cost | cost / diameter | diameter')
    for j, (beta, power) in enumerate([(0.0, 1.0), (args.beta, args.power)]):
        w = length * (1.0 + beta * rel ** power)
        graph = scipy.sparse.csr_matrix((w, (src, dst)), shape=(n, n))
        dist, pred = scipy.sparse.csgraph.dijkstra(graph, indices=c, directed=False, return_predecessors=True)
        diameter = float(scipy.sparse.csgraph.dijkstra(graph, indices=int(np.argmax(dist)), directed=False).max())
        ax = axes[0, 1 + j]
        ax.imshow(img)
        ax.contour(mask, [0.5], colors='yellow', linewidths=0.8)
        lines = []
        for i, nb in enumerate(nbrs):
            node, path = p2n[nb], []
            while node != c:
                path.append(node)
                node = pred[node]
            path.append(node)
            pr, pc = rows[path], cols[path]
            px = float(np.hypot(np.diff(pr), np.diff(pc)).sum())
            cost = float(dist[p2n[nb]])
            print(f'{beta:g} | {power:g} | N{i + 1} {nb} | {px:.1f} | {cost:.1f} | {cost / diameter:.3f} | {diameter:.1f}')
            lines.append(f'N{i + 1}: {px:.0f} px, norm {cost / diameter:.3f}')
            ax.plot(pc, pr, '-', color='k', lw=4.0)
            ax.plot(pc, pr, '-', color=COLOURS[i % len(COLOURS)], lw=2.2)
        draw_points(ax, centre, nbrs)
        head = 'paths, beta 0 (training metric)' if beta == 0 else f'paths, feature step cost, beta {beta:g}, power {power:g}'
        ax.set_title(head + '\n' + '\n'.join(lines), fontsize=8)

    ax = axes[0, 3]
    ax.imshow(pca_rgb(feats, mask))
    ax.contour(mask, [0.5], colors='k', linewidths=0.8)
    draw_points(ax, centre, nbrs)
    ax.set_title('feature map (PCA, 3 components -> RGB)', fontsize=8)
    for ax in axes[0, 4:]:
        ax.axis('off')

    sims = [np.where(mask, feats @ feats[p], np.nan) for _, p in points]
    vmin = float(min(np.nanmin(s) for s in sims))
    print('cosine similarity between the chosen pixels:')
    for (name, p), s in zip(points, sims):
        print(f'  {name}: ' + ', '.join(f'{other} {s[q]:.3f}' for other, q in points))
    for k, ((name, p), s) in enumerate(zip(points, sims)):
        ax = axes[1, k]
        ax.imshow(img, alpha=0.25)
        im = ax.imshow(s, cmap='turbo', vmin=vmin, vmax=1.0)
        draw_points(ax, centre, nbrs)
        ax.plot(p[1], p[0], 'o', ms=20, mfc='none', mec='w', mew=2.0)
        ax.set_title(f'cos similarity to {name} {p}\n' +
                     ', '.join(f'{other} {s[q]:.2f}' for other, q in points if other != name), fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes[1, len(points):]:
        ax.axis('off')

    fig.suptitle(f'{args.title}  setup: chosen pixels, 2D geodesic paths, feature map, similarity heatmaps', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output_path)
    print(f'saved {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--object_mask_path', type=str, required=True)
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--center', type=int, nargs=2, required=True, help='row col')
    parser.add_argument('--neighbors', type=int, nargs='+', required=True, help='row col row col ...')
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--power', type=float, required=True, help='1 = linear feature cost, 2 = squared')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--output_path', type=str, required=True)
    main(parser.parse_args())
