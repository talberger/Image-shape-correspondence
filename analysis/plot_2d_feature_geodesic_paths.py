"""2D geodesic paths with a FEATURE-dependent step cost, for one centre pixel and a few neighbours.

Same graph as plot_2d_geodesic_paths.py (8-connected over the mask's largest connected component), but the
edge p -> q costs
    |p - q| * (1 + beta * (delta(p, q) / median_delta) ** power),    delta = 1 - cos(f_p, f_q)
with f the frozen per-pixel features. beta = 0 is the training's metric. median_delta is the median of delta over
all edges, so beta is "how many pixels one typical feature step is worth". Each configuration gets its own
diameter (double sweep in the same metric), so norm_d_2D stays comparable with the training's.
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

STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def load_features(path, H, W):
    f = torch.load(path, map_location='cpu')
    f = f.float().squeeze()
    if f.shape[:2] != (H, W):     # (C, H, W) -> (H, W, C)
        f = f.permute(1, 2, 0)
    assert f.shape[:2] == (H, W), (f.shape, H, W)
    return torch.nn.functional.normalize(f, dim=-1).numpy()


def mask_edges(mask, feats):
    """Directed edge list (src, dst, pixel length, delta = 1 - cos) of the 8-connected mask graph."""
    H, W = mask.shape
    rows, cols = np.nonzero(mask)
    p2n = -np.ones((H, W), dtype=np.int64)
    p2n[rows, cols] = np.arange(len(rows))
    f = feats[rows, cols]
    src, dst, length, delta = [], [], [], []
    for dr, dc in STEPS:
        nr, nc = rows + dr, cols + dc
        ok = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        s = np.nonzero(ok)[0]
        d = p2n[nr[ok], nc[ok]]
        on = d >= 0
        s, d = s[on], d[on]
        src.append(s); dst.append(d)
        length.append(np.full(len(s), np.hypot(dr, dc)))
        delta.append(np.clip(1.0 - (f[s] * f[d]).sum(1), 0.0, None))
    return (np.concatenate(src), np.concatenate(dst), np.concatenate(length), np.concatenate(delta),
            p2n, rows, cols)


def span_rel(f, src, dst, rows, cols, p2n, span):
    """Option A over a wider span. delta of the step p -> q measured over `span` pixels centred on the step: 1 - cos
    between the pixel (span - 1) // 2 behind p and the one as far ahead of q (an end that leaves the mask falls back to
    p / q -- KNOWN ARTEFACT: the span shrinks along the mask border, so the border becomes a cheap corridor).
    f = per-node features. Divided by its median."""
    dr, dc = rows[dst] - rows[src], cols[dst] - cols[src]
    h = (span - 1) // 2
    ends = []
    for base, sign in ((src, -1), (dst, 1)):
        r, cc = rows[base] + sign * h * dr, cols[base] + sign * h * dc
        ok = (r >= 0) & (r < p2n.shape[0]) & (cc >= 0) & (cc < p2n.shape[1])
        node = base.copy()
        node[ok] = np.where(p2n[r[ok], cc[ok]] >= 0, p2n[r[ok], cc[ok]], base[ok])
        ends.append(node)
    d = np.concatenate([1.0 - (f[ends[0][i:i + 50000]] * f[ends[1][i:i + 50000]]).sum(1)     # chunked: RAM
                        for i in range(0, len(src), 50000)]).clip(0.0, None)
    return d / float(np.median(d))


def pca_rgb(feats, mask):
    x = feats[mask]
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x[::max(1, len(x) // 5000)], full_matrices=False)
    y = x @ vt[:3].T
    lo, hi = np.percentile(y, 1, axis=0), np.percentile(y, 99, axis=0)
    rgb = np.ones(mask.shape + (3,))
    rgb[mask] = np.clip((y - lo) / (hi - lo), 0, 1)
    return rgb


def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    img = np.array(Image.open(args.image_path).convert('RGB'))
    mask, _ = largest_connected_component(torch.load(args.object_mask_path, map_location='cpu').numpy().astype(bool))
    feats = load_features(args.features_path, *mask.shape)
    centre = tuple(args.center)
    nbrs = [tuple(args.neighbors[i:i + 2]) for i in range(0, len(args.neighbors), 2)]
    for p in [centre] + nbrs:
        if not mask[p]:
            raise ValueError(f'{p} (row, col) is not on the mask')

    src, dst, length, delta, p2n, rows, cols = mask_edges(mask, feats)
    med = float(np.median(delta))
    print(f'edges {len(src)}, delta = 1 - cos: median {med:.2e}, p90 {np.percentile(delta, 90):.2e}, '
          f'p99 {np.percentile(delta, 99):.2e}, max {delta.max():.2e}')
    f = feats[rows, cols]
    def step_rel(span):
        return delta / med if span == 1 else span_rel(f, src, dst, rows, cols, p2n, span)

    # option B: dissimilarity to the START pixel (mean over the step's two pixels), divided by its median
    u = np.clip(1.0 - f @ feats[centre], 0.0, None)
    start_rel = 0.5 * (u[src] + u[dst]) / float(np.median(u))
    n = len(rows)
    c = p2n[centre]

    configs = [(0.0, 1.0, 'step', 1)] + [(b, p, m, sp) for m in args.cost_modes for sp in (args.delta_spans if m != 'start' else [1])
                                         for p in args.powers for b in args.betas]
    rels = {sp: step_rel(sp) for sp in args.delta_spans}
    backgrounds = [('photo', img), ('feature PCA', pca_rgb(feats, mask))]
    points = [('C', centre)] + [(f'N{i + 1}', nb) for i, nb in enumerate(nbrs)]
    nrow = len(backgrounds) + int(args.similarity_row)
    ncol = max(len(configs), len(points)) if args.similarity_row else len(configs)
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, figsize=(ncol * img.shape[1] / 62, nrow * img.shape[0] / 62), dpi=110)
    for ax in axes.ravel():
        ax.axis('off')
    colours = ['magenta', 'lime', 'orange', 'cyan', 'white', 'red']
    print('beta | power | neighbour | path px | cost | cost / diameter | diameter | cost / pixel diameter | path px / pixel diameter')
    px_diameter = None
    for j, (beta, power, mode, span) in enumerate(configs):
        extra = np.zeros(len(length))
        if mode in ('step', 'both') and beta > 0:
            extra = extra + beta * rels[span] ** power
        if mode in ('start', 'both') and beta > 0:
            extra = extra + beta * start_rel ** power
        w = length * (1.0 + extra)
        graph = scipy.sparse.csr_matrix((w, (src, dst)), shape=(n, n))
        dist, pred = scipy.sparse.csgraph.dijkstra(graph, indices=c, directed=False, return_predecessors=True)
        far = int(np.argmax(dist))      # double sweep, same metric
        diameter = float(scipy.sparse.csgraph.dijkstra(graph, indices=far, directed=False).max())
        if beta == 0:
            px_diameter = diameter
        paths, lines = [], []
        for i, nb in enumerate(nbrs):
            node, path = p2n[nb], []
            while node != c:
                path.append(node)
                node = pred[node]
            path.append(node)
            pr, pc = rows[path], cols[path]
            px = float(np.hypot(np.diff(pr), np.diff(pc)).sum())
            cost = float(dist[p2n[nb]])
            print(f'{mode} span {span} | {beta:g} | {power:g} | N{i + 1} {nb} | {px:.1f} | {cost:.1f} | {cost / diameter:.3f} | {diameter:.1f} | {cost / px_diameter:.3f} | {px / px_diameter:.3f}')
            paths.append((pr, pc))
            # path length / pixel diameter = the features choose the route, the distance is its length in pixels
            lines.append(f'N{i + 1}: {px:.0f} px = {px / px_diameter:.2f} of the pixel diameter')
        for k, (name, bg) in enumerate(backgrounds):
            ax = axes[k, j]
            ax.axis('on')
            ax.imshow(bg)
            ax.contour(mask, [0.5], colors='yellow' if k == 0 else 'k', linewidths=0.8)
            for i, (pr, pc) in enumerate(paths):
                col = colours[i % len(colours)]
                ax.plot(pc, pr, '-', color='k', lw=4.0)
                ax.plot(pc, pr, '-', color=col, lw=2.2)
                ax.plot(nbrs[i][1], nbrs[i][0], 'o', ms=9, mfc=col, mec='k', mew=1.2)
                ax.text(nbrs[i][1] + 6, nbrs[i][0] - 6, f'N{i + 1}', color=col, fontsize=9, weight='bold',
                        bbox=dict(fc='k', alpha=0.6, pad=1.0, lw=0))
            ax.plot(centre[1], centre[0], '*', ms=16, mfc='cyan', mec='k', mew=1.2)
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                name_of = {'step': 'A: step' + (f' over {span} px' if span > 1 else ''), 'start': 'B: unlike C',
                           'both': 'A + B' + (f', A over {span} px' if span > 1 else '')}
                head = ('ORIGINAL: training metric, no features (beta 0)' if beta == 0 else
                        f'{name_of[mode]}, beta {beta:g}, power {power:g}')
                ax.set_title(head + '\n' + '\n'.join(lines), fontsize=8)
    if args.similarity_row:     # cos(f_p, f_q) of every mask pixel p against each chosen pixel q
        sims = [np.where(mask, feats @ feats[p], np.nan) for _, p in points]
        vmin = float(min(np.nanmin(s) for s in sims))
        for k, ((name, p), s) in enumerate(zip(points, sims)):
            ax = axes[-1, k]
            ax.axis('on')
            ax.imshow(img, alpha=0.25)
            im = ax.imshow(s, cmap='turbo', vmin=vmin, vmax=1.0)
            for i, nb in enumerate(nbrs):
                ax.plot(nb[1], nb[0], 'o', ms=9, mfc=colours[i % len(colours)], mec='k', mew=1.2)
            ax.plot(centre[1], centre[0], '*', ms=16, mfc='cyan', mec='k', mew=1.2)
            ax.plot(p[1], p[0], 'o', ms=20, mfc='none', mec='w', mew=2.0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'cos similarity to {name} {p}\n' +
                         ', '.join(f'{other} {s[q]:.2f}' for other, q in points if other != name), fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f'{args.title}  feature-cost 2D geodesics from C {centre} (row, col); '
                 'step cost = px * (1 + beta * rel ** power); A: rel = (1 - cos between the step\'s pixels) / median, '
                 'B: rel = (1 - cos to C) / median', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.output_path)
    print(f'saved {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--object_mask_path', type=str, required=True)
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--center', type=int, nargs=2, required=True, help='row col')
    parser.add_argument('--neighbors', type=int, nargs='+', required=True, help='row col row col ...')
    parser.add_argument('--betas', type=float, nargs='+', required=True)
    parser.add_argument('--powers', type=float, nargs='+', required=True)
    parser.add_argument('--cost_modes', choices=['step', 'start', 'both'], nargs='+', default=['step'],
                        help='step = A, dissimilarity between the pixels of a step; start = B, dissimilarity to the centre pixel; '
                             'both = the two terms added')
    parser.add_argument('--delta_spans', type=int, nargs='+', default=[1],
                        help='option A: measure the step\'s feature change over this many pixels (1 = the two pixels of the step)')
    parser.add_argument('--similarity_row', action='store_true',
                        help='add a bottom row: cosine-similarity heatmap of the centre and of each neighbour')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--output_path', type=str, required=True)
    main(parser.parse_args())
