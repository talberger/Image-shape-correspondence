"""Original vs feature-cost 2D geodesic paths for several centres, each with a few neighbours INSIDE its own fine segment.

Checks whether the feature step cost (build_geodesic_tables.py --feature_geodesic) lengthens paths that stay inside one part.
Row 1 = original metric, then one row per beta of the feature cost; one column per centre; fine-segment borders in white.
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
from plot_2d_feature_geodesic_paths import load_features, mask_edges

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def main(args):
    img = np.asarray(Image.open(args.image_path).convert('RGB'))
    mask, _ = largest_connected_component(torch.load(args.object_mask_path, map_location='cpu').numpy().astype(bool))
    seg = torch.load(args.fine_seg_path, map_location='cpu').numpy()
    feats = load_features(args.features_path, *mask.shape)
    src, dst, length, delta, p2n, rows, cols = mask_edges(mask, feats)
    rel = delta / float(np.median(delta))
    n = len(rows)
    graphs = [('ORIGINAL metric', scipy.sparse.csr_matrix((length, (src, dst)), shape=(n, n)))]
    for beta in args.betas:
        if args.threshold_percentiles:      # WALL ONLY: a step is penalised (x (1 + beta)) only above this percentile of the feature change
            for q in args.threshold_percentiles:
                t = float(np.percentile(rel, q))
                graphs.append((f'WALL ONLY: steps above percentile {q:g} (change > {t:.1f}x median) cost x{1 + beta:g}',
                               scipy.sparse.csr_matrix((length * (1.0 + beta * (rel > t)), (src, dst)), shape=(n, n))))
        else:
            graphs.append((f'FEATURE cost, beta {beta:g}, power {args.power:g}',
                           scipy.sparse.csr_matrix((length * (1.0 + beta * rel ** args.power), (src, dst)), shape=(n, n))))
    nrow = len(graphs)
    centres = [tuple(args.centers[i:i + 2]) for i in range(0, len(args.centers), 2)]
    cross = [tuple(args.cross_points[i:i + 2]) for i in range(0, len(args.cross_points), 2)]     # last column: centre + given neighbours, any segment
    if cross:
        centres.append(cross[0])
    fig, axes = plt.subplots(nrow, len(centres), squeeze=False, figsize=(len(centres) * img.shape[1] / 55, nrow * img.shape[0] / 50), dpi=110)
    colours = ['magenta', 'lime', 'orange', 'cyan', 'yellow', 'red', 'white', 'deepskyblue']
    for j, c in enumerate(centres):
        nbrs = cross[1:] if (cross and j == len(centres) - 1) else []
        for dr, dc in ([] if nbrs else DIRECTIONS):       # same-segment neighbours at --radius px; shrink the radius until the pixel is in the segment
            for r in np.arange(args.radius, args.radius * 0.4, -4):
                q = (int(round(c[0] + dr * r / np.hypot(dr, dc))), int(round(c[1] + dc * r / np.hypot(dr, dc))))
                if 0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1] and mask[q] and (args.any_segment or seg[q] == seg[c]):
                    nbrs.append(q)
                    break
        same = np.array([seg[q] == seg[c] for q in nbrs])
        px = np.zeros((nrow, len(nbrs)))
        for k, (name, graph) in enumerate(graphs):
            _, pred = scipy.sparse.csgraph.dijkstra(graph, indices=p2n[c], directed=False, return_predecessors=True)
            ax = axes[k, j]
            ax.imshow(img)
            ax.contour(seg * mask, levels=np.arange(0.5, seg.max() + 1), colors='white', linewidths=1.0)
            for i, q in enumerate(nbrs):
                node, path = p2n[q], []
                while node != p2n[c]:
                    path.append(node)
                    node = pred[node]
                path.append(node)
                pr, pc = rows[path], cols[path]
                px[k, i] = np.hypot(np.diff(pr), np.diff(pc)).sum()
                ax.plot(pc, pr, '-', color='k', lw=3.6)
                ax.plot(pc, pr, '-', color=colours[i], lw=2.0)
                ax.plot(q[1], q[0], 'o' if same[i] else 's', ms=8 if same[i] else 10, mfc=colours[i], mec='k' if same[i] else 'white', mew=1.1 if same[i] else 2.0)
            ax.plot(c[1], c[0], '*', ms=16, mfc='cyan', mec='k', mew=1.2)
            ax.set_xticks([]); ax.set_yticks([])
            head = f'{name}\n' + ('ACROSS PARTS: ' if (cross and j == len(centres) - 1) else '') + f'centre {c}, segment {int(seg[c])}; path px: ' + ' '.join(f'{v:.0f}' for v in px[k])
            if k >= 1:
                ratio = px[k] / px[0]
                head += f'\nlength / original: ' + ' '.join(f'{v:.1f}' for v in ratio) + f'  (mean {np.mean(ratio):.2f})'
                if same.any() and (~same).any():
                    head += f'\nmean ratio: same segment {np.mean(ratio[same]):.2f} ({same.sum()}) | other segment {np.mean(ratio[~same]):.2f} ({(~same).sum()})'
            ax.set_title(head, fontsize=8)
        print(f'centre {c} seg {int(seg[c])}: mean / max length ratio per row: ' +
              ' | '.join(f'{graphs[k][0]}: {np.mean(px[k] / px[0]):.2f} / {np.max(px[k] / px[0]):.1f}'
                         + (f' (same {np.mean((px[k] / px[0])[same]):.2f}' if same.any() else ' (same -')
                         + (f', other {np.mean((px[k] / px[0])[~same]):.2f})' if (~same).any() else ', other -)')
                         + (' ratios ' + ' '.join(f'{v:.1f}' for v in px[k] / px[0]) if (cross and j == len(centres) - 1) else '')
                         for k in range(1, nrow)))
    scope = 'ANY segment (circle = same segment, square = other segment)' if args.any_segment else 'its OWN fine segment'
    fig.suptitle(f'{args.title}  paths from each centre to neighbours in {scope} (white = segment borders)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output_path)
    print(f'saved {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--object_mask_path', type=str, required=True)
    parser.add_argument('--fine_seg_path', type=str, required=True)
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--centers', type=int, nargs='+', required=True, help='row col row col ...')
    parser.add_argument('--radius', type=float, default=60.0, help='straight-line distance of the neighbours, px')
    parser.add_argument('--any_segment', action='store_true', help='neighbours may lie in any fine segment (default: the centre\'s own)')
    parser.add_argument('--betas', type=float, nargs='+', required=True, help='one row per beta, under the original metric')
    parser.add_argument('--power', type=float, default=8.0)
    parser.add_argument('--threshold_percentiles', type=float, nargs='+', default=None,
                        help='WALL ONLY cost instead of the power law: one row per (beta, percentile)')
    parser.add_argument('--cross_points', type=int, nargs='+', default=[],
                        help='extra last column: centre row col, then neighbours row col ... (not restricted to the centre\'s segment)')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--output_path', type=str, required=True)
    main(parser.parse_args())
