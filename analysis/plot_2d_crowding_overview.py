"""Crowded-parts study, pod 46 (2026-10-13): every feature-cost variant that was tried, in one figure, plus a summary.

Step cost = px * (1 + beta * rel ** power). A: rel = (1 - cos between the step's two pixels) / median.
B: rel = 1 - cos to the start pixel C (raw in the power-1 row, / median in the A-vs-B row, as they were run).
The number under each panel is the PATH LENGTH IN PIXELS / the original pixel diameter: the features choose the
route, the distance is its length (Tal's choice, 2026-10-13).
--overview_path: one row per experiment, column 0 = the original training metric.
--summary_path:  original vs the chosen variant (--chosen_beta, --chosen_power), feature PCA, feature-change map,
                 cos-similarity heatmap of C and of each neighbour.
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
from plot_2d_feature_geodesic_paths import load_features, mask_edges, pca_rgb, span_rel
from plot_2d_feature_change_map import change_map

COLOURS = ['magenta', 'lime', 'orange', 'cyan', 'white', 'red']


def main(args):
    img = np.array(Image.open(args.image_path).convert('RGB'))
    mask, _ = largest_connected_component(torch.load(args.object_mask_path, map_location='cpu').numpy().astype(bool))
    feats = load_features(args.features_path, *mask.shape)
    centre = tuple(args.center)
    nbrs = [tuple(args.neighbors[i:i + 2]) for i in range(0, len(args.neighbors), 2)]
    points = [('C', centre)] + [(f'N{i + 1}', nb) for i, nb in enumerate(nbrs)]
    src, dst, length, delta, p2n, rows, cols = mask_edges(mask, feats)
    f = feats[rows, cols]
    n, c = len(rows), p2n[centre]
    u = np.clip(1.0 - f @ feats[centre], 0.0, None)
    u_edge = 0.5 * (u[src] + u[dst])
    rel = {'A': delta / float(np.median(delta)), 'B raw': u_edge, 'B': u_edge / float(np.median(u))}

    def solve(extra):
        """Paths from C under step cost px * (1 + extra) and their pixel lengths."""
        with np.errstate(over='ignore'):
            graph = scipy.sparse.csr_matrix((length * (1.0 + extra), (src, dst)), shape=(n, n))
        _, pred = scipy.sparse.csgraph.dijkstra(graph, indices=c, directed=False, return_predecessors=True)
        out = []
        for nb in nbrs:
            node, path = p2n[nb], []
            while node != c:
                path.append(node)
                node = pred[node]
            path.append(node)
            pr, pc = rows[path], cols[path]
            out.append((pr, pc, float(np.hypot(np.diff(pr), np.diff(pc)).sum())))
        return out

    def term(kind, beta, power, span=1):
        r = rel[kind] if span == 1 else span_rel(f, src, dst, rows, cols, p2n, span)
        with np.errstate(over='ignore'):
            return beta * r ** power

    def draw(ax, bg, paths, outline='yellow'):
        ax.imshow(bg)
        ax.contour(mask, [0.5], colors=outline, linewidths=0.6)
        for i, (pr, pc, _) in enumerate(paths):
            ax.plot(pc, pr, '-', color='k', lw=3.2)
            ax.plot(pc, pr, '-', color=COLOURS[i % len(COLOURS)], lw=1.8)
            ax.plot(nbrs[i][1], nbrs[i][0], 'o', ms=7, mfc=COLOURS[i % len(COLOURS)], mec='k', mew=1.0)
        ax.plot(centre[1], centre[0], '*', ms=13, mfc='cyan', mec='k', mew=1.0)
        ax.set_xticks([]); ax.set_yticks([])

    original = solve(np.zeros(len(length)))
    diam = args.pixel_diameter
    ratios = lambda paths: '  '.join(f'N{i + 1} {p[2] / diam:.2f}' for i, p in enumerate(paths))

    g = lambda v: f'{v:g}'
    experiments = [
        ('A, power 1: beta', [(f'beta {g(b)}', term('A', b, 1)) for b in (10, 30, 100, 300, 500, 1000)]),
        ('B (unlike C), power 1: beta', [(f'beta {g(b)}', term('B raw', b, 1)) for b in (1, 3, 10, 30, 100)]),
        ('A, beta 10: power', [(f'power {g(p)}', term('A', 10, p)) for p in (2, 4, 8, 12, 16, 24, 32, 64, 128, 256)]),
        ('A, power 4: beta', [(f'beta {g(b)}', term('A', b, 4)) for b in (0.01, 1, 10, 100, 1000)]),
        ('A, power 8: beta', [(f'beta {g(b)}', term('A', b, 8)) for b in (0.01, 1, 10, 100, 1000, 1e6, 1e9, 1e12)]),
        ('A, power 32: beta', [(f'beta {g(b)}', term('A', b, 32)) for b in (1e3, 1e6, 1e9, 1e12)]),
        ('beta 10, power 8: A / B / A + B', [('A', term('A', 10, 8)), ('B', term('B', 10, 8)),
                                             ('A + B', term('A', 10, 8) + term('B', 10, 8))]),
        ('A, beta 10, power 8: change measured over (px)\n[mask-border artefact, not final]',
         [(f'{s} px', term('A', 10, 8, s)) for s in (7, 14, 28)]),
    ]
    chosen_key = ('A, power 8: beta', f'beta {g(args.chosen_beta)}')

    if args.overview_path:
        os.makedirs(os.path.dirname(args.overview_path), exist_ok=True)
        ncol = 1 + max(len(v) for _, v in experiments)
        w = 2.3
        fig, axes = plt.subplots(len(experiments), ncol, squeeze=False,
                                 figsize=(ncol * w, len(experiments) * (w * img.shape[0] / img.shape[1] + 0.45)), dpi=100)
        for ax in axes.ravel():
            ax.axis('off')
        for r, (name, variants) in enumerate(experiments):
            ax = axes[r, 0]
            ax.axis('on')
            draw(ax, img, original)
            ax.set_title(f'ORIGINAL (no features)\n{ratios(original)}', fontsize=7)
            ax.set_ylabel(name, fontsize=9, weight='bold')
            for k, (label, extra) in enumerate(variants):
                paths = solve(extra)
                print(f'{name.splitlines()[0]} | {label} | {ratios(paths)}', flush=True)
                ax = axes[r, 1 + k]
                ax.axis('on')
                draw(ax, img, paths)
                ax.set_title(f'{label}\n{ratios(paths)}', fontsize=7)
                if (name, label) == chosen_key and args.chosen_power == 8:
                    for sp in ax.spines.values():
                        sp.set_edgecolor('lime'); sp.set_linewidth(4)
        fig.suptitle(f'{args.title}  every variant tried. Step cost = px * (1 + beta * rel ** power); '
                     'A: rel = feature change between the two pixels of the step / median;  B: rel = 1 - cos to C.\n'
                     'Numbers = path length in pixels / original pixel diameter '
                     f'({diam:g} px).  Green frame = chosen (A, beta {args.chosen_beta:g}, power {args.chosen_power:g}).',
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(args.overview_path)
        print(f'saved {args.overview_path}')

    if args.summary_path:
        os.makedirs(os.path.dirname(args.summary_path), exist_ok=True)
        new = solve(term('A', args.chosen_beta, args.chosen_power))
        ncol = max(4, len(points))
        fig, axes = plt.subplots(2, ncol, squeeze=False, figsize=(ncol * img.shape[1] / 62, 2 * img.shape[0] / 62 + 0.8), dpi=110)
        for ax in axes.ravel():
            ax.axis('off')
        lines = lambda paths: '\n'.join(f'N{i + 1}: {p[2]:.0f} px = {p[2] / diam:.2f} of the diameter' for i, p in enumerate(paths))
        for k, (bg, paths, title, outline) in enumerate([
                (img, original, 'ORIGINAL path: training metric, no features\n' + lines(original), 'yellow'),
                (img, new, f'NEW path: feature step cost, beta {args.chosen_beta:g}, power {args.chosen_power:g}\n' + lines(new), 'yellow'),
                (pca_rgb(feats, mask), new, f'feature map (PCA -> RGB), {args.features_label}\nwith the new paths', 'k')]):
            axes[0, k].axis('on')
            draw(axes[0, k], bg, paths, outline)
            axes[0, k].set_title(title, fontsize=8)
        ax = axes[0, 3]
        ax.axis('on')
        m = change_map(feats, mask, 1)
        m = m / np.nanmedian(m)
        draw(ax, np.ones_like(img) * 255, new, 'k')
        im = ax.imshow(m, cmap='magma', vmin=0.0, vmax=float(np.nanpercentile(m, 99.5)))
        for i, (pr, pc, _) in enumerate(new):
            ax.plot(pc, pr, '-', color=COLOURS[i % len(COLOURS)], lw=1.8)
        ax.set_title(f'where the features change (1 = typical pixel), {args.features_label}\nthe bright ridges are the "walls" the new paths avoid', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        sims = [np.where(mask, feats @ feats[p], np.nan) for _, p in points]
        vmin = float(min(np.nanmin(s) for s in sims))
        for k, ((name, p), s) in enumerate(zip(points, sims)):
            ax = axes[1, k]
            ax.axis('on')
            ax.imshow(img, alpha=0.25)
            im = ax.imshow(s, cmap='turbo', vmin=vmin, vmax=1.0)
            for i, nb in enumerate(nbrs):
                ax.plot(nb[1], nb[0], 'o', ms=9, mfc=COLOURS[i % len(COLOURS)], mec='k', mew=1.2)
            ax.plot(centre[1], centre[0], '*', ms=16, mfc='cyan', mec='k', mew=1.2)
            ax.plot(p[1], p[0], 'o', ms=20, mfc='none', mec='w', mew=2.0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'cos similarity to {name} {p}, {args.features_label}\n' +
                         ', '.join(f'{other} {s[q]:.2f}' for other, q in points if other != name), fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(f'{args.title}  SUMMARY: original 2D geodesic vs feature-aware 2D geodesic.  '
                     f'Step cost = px * (1 + {args.chosen_beta:g} * (feature change of the step / median) ** {args.chosen_power:g}); '
                     f'distance = path length in pixels / original diameter ({diam:g} px)', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(args.summary_path)
        print(f'saved {args.summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--object_mask_path', type=str, required=True)
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--center', type=int, nargs=2, required=True, help='row col')
    parser.add_argument('--neighbors', type=int, nargs='+', required=True, help='row col row col ...')
    parser.add_argument('--pixel_diameter', type=float, required=True, help='true 2D diameter of the original metric, px')
    parser.add_argument('--chosen_beta', type=float, required=True)
    parser.add_argument('--chosen_power', type=float, required=True)
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--features_label', type=str, default='frozen DINO only',
                        help='which features --features_path holds, written in the panel titles')
    parser.add_argument('--overview_path', type=str, default=None)
    parser.add_argument('--summary_path', type=str, default=None)
    main(parser.parse_args())
