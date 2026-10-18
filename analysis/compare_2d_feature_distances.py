"""Side by side: the training's 2D geodesic vs feature-aware 2D distances, from one centre pixel to a few neighbours.

Columns (each normalised by its OWN diameter, double sweep in the same metric, so the numbers are comparable;
columns 4-5 use a robust 99th-percentile diameter, see embedding_column):
  0  training metric     8-connected mask graph, step cost = pixel length                          (full resolution)
  1  feature cost        step cost = px * (1 + beta * (delta / median) ** power), delta = 1 - cos  (full resolution)
  2  fast marching       the same cost as a per-pixel field (a pixel's mean delta to its neighbours), solved as a
                         continuous Eikonal equation: paths are not tied to the 8 grid directions  (full resolution)
  3  Isomap              k nearest neighbours in FEATURE space only, edge = feature distance       (--grid_stride grid)
  4  biharmonic          spatial 8-neighbour graph, affinity exp(-delta / median); all paths count (--grid_stride grid)
  5  spectral            dense feature affinity max(cos, 0), first --num_eig Laplacian eigenvectors (--grid_stride grid)
Top row: the path where the method has one. Bottom row: the distance from the centre to every pixel.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
import skfmm
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_geodesic_tables import largest_connected_component
from plot_2d_feature_geodesic_paths import STEPS, load_features, mask_edges


def walk(pred, start, end):
    node, path = end, []
    while node != start:
        path.append(node)
        node = pred[node]
    path.append(node)
    return path


def shortest_path_column(graph, c, nodes):
    """Distances from c normalised by the double-sweep diameter, and the paths to `nodes`."""
    dist, pred = scipy.sparse.csgraph.dijkstra(graph, indices=c, directed=False, return_predecessors=True)
    diameter = float(scipy.sparse.csgraph.dijkstra(graph, indices=int(np.argmax(dist)), directed=False).max())
    return dist / diameter, [walk(pred, c, n) for n in nodes]


def fast_marching_column(mask, cost, centre, nbrs):
    """Travel time from centre under the per-pixel cost field, normalised by its double-sweep diameter, and the paths
    to `nbrs` (0.5 px steps down the travel time). The source is the zero contour around the centre pixel, so every
    time is ~0.5 px short. Pixels attached only diagonally are unreachable on the 4-neighbour stencil and stay nan."""
    def travel(p):
        phi = np.ones(mask.shape)
        phi[p] = -1
        return skfmm.travel_time(np.ma.MaskedArray(phi, ~mask), 1.0 / cost).filled(np.nan)
    T = travel(centre)
    far = np.unravel_index(np.nanargmax(T), T.shape)
    T = T / np.nanmax(travel(far))
    g = np.gradient(np.where(np.isnan(T), np.nanmax(T), T))     # off the object = high, so the descent stays inside
    paths = []
    for nb in nbrs:
        if np.isnan(T[nb]):
            raise ValueError(f'{nb} is not reachable by fast marching')
        p, path = np.array(nb, dtype=np.float64), [nb]
        for _ in range(20 * sum(mask.shape)):
            if np.hypot(*(p - centre)) < 1.0:
                break
            d = np.array([scipy.ndimage.map_coordinates(gi, p[:, None], order=1)[0] for gi in g])
            p = p - 0.5 * d / max(np.linalg.norm(d), 1e-12)
            path.append(tuple(p))
        else:
            print(f'WARNING: fast marching path from {nb} did not reach the centre (stopped at {p.round(1)})')
        path.append(centre)
        paths.append(tuple(np.array(path).T))
    return T, paths


def embedding_column(emb, c):
    """Euclidean distance in an embedding, normalised by a ROBUST diameter: 99th percentiles instead of maxima in the
    double sweep. A weakly attached node (a mane-hair pixel) is very far from everything in these embeddings, and the
    true maximum then squeezes every other distance to ~0."""
    dist = np.linalg.norm(emb - emb[c], axis=1)
    far = int(np.argsort(dist)[int(0.99 * (len(dist) - 1))])
    return dist / np.percentile(np.linalg.norm(emb - emb[far], axis=1), 99)


def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    img = np.array(Image.open(args.image_path).convert('RGB'))
    mask, _ = largest_connected_component(torch.load(args.object_mask_path, map_location='cpu').numpy().astype(bool))
    H, W = mask.shape
    feats = load_features(args.features_path, H, W)
    centre = tuple(args.center)
    nbrs = [tuple(args.neighbors[i:i + 2]) for i in range(0, len(args.neighbors), 2)]
    columns = []   # (title, normalised distance map (H, W) with nan off the object, paths as (rows, cols) or None, values)

    # ---- 0-1: shortest paths on the full-resolution mask graph ----
    src, dst, length, delta, p2n, rows, cols = mask_edges(mask, feats)
    rel = delta / np.median(delta)
    n = len(rows)
    ids = [p2n[p] for p in nbrs]
    for title, w in [('training metric\n(pixel length only)', length),
                     (f'feature cost (8-direction graph)\nbeta {args.beta:g}, power {args.power:g}',
                      length * (1 + args.beta * rel ** args.power))]:
        d, paths = shortest_path_column(scipy.sparse.csr_matrix((w, (src, dst)), shape=(n, n)), p2n[centre], ids)
        dmap = np.full((H, W), np.nan)
        dmap[rows, cols] = d
        columns.append((title, dmap, [(rows[p], cols[p]) for p in paths], [float(d[i]) for i in ids]))

    # ---- 2: fast marching on the same cost, as a per-pixel field ----
    cost = np.ones((H, W))
    cost[rows, cols] = 1 + args.beta * (np.bincount(src, delta, n) / np.bincount(src, minlength=n) / np.median(delta)) ** args.power
    dmap, paths = fast_marching_column(mask, cost, centre, nbrs)
    columns.append((f'fast marching (continuous)\nbeta {args.beta:g}, power {args.power:g}', dmap, paths,
                    [float(dmap[p]) for p in nbrs]))

    # ---- 3-5: on a coarse grid (the features are upsampled from the 14 px patch grid anyway) ----
    s = args.grid_stride
    grid = np.zeros_like(mask)
    grid[centre[0] % s::s, centre[1] % s::s] = True
    grid &= mask
    gr, gc = np.nonzero(grid)
    g2n = -np.ones((H, W), dtype=np.int64)
    g2n[gr, gc] = np.arange(len(gr))
    es, ed = [], []
    for dr, dc in STEPS:
        nr, nc = gr + dr * s, gc + dc * s
        ok = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        a, b = np.nonzero(ok)[0], g2n[nr[ok], nc[ok]]
        es.append(a[b >= 0]); ed.append(b[b >= 0])
    es, ed = np.concatenate(es), np.concatenate(ed)
    # keep the grid's largest connected component (thin parts can fall between grid nodes)
    _, lab = scipy.sparse.csgraph.connected_components(
        scipy.sparse.csr_matrix((np.ones(len(es)), (es, ed)), shape=(len(gr), len(gr))), directed=False)
    keep = lab == np.bincount(lab).argmax()
    remap = -np.ones(len(gr), dtype=np.int64)
    remap[keep] = np.arange(keep.sum())
    gr, gc = gr[keep], gc[keep]
    on = keep[es] & keep[ed]
    es, ed = remap[es[on]], remap[ed[on]]
    m = len(gr)
    f = feats[gr, gc].astype(np.float64)
    cos = f @ f.T

    def snap(p):
        return int(np.argmin((gr - p[0]) ** 2 + (gc - p[1]) ** 2))
    gc_id, gids = snap(centre), [snap(p) for p in nbrs]
    print(f'coarse grid: stride {s}, {m} nodes; neighbours snapped to {[(int(gr[i]), int(gc[i])) for i in gids]}')

    def to_map(d):     # nearest grid node for every mask pixel
        filled = np.full((H, W), np.nan)
        filled[gr, gc] = d
        idx = scipy.ndimage.distance_transform_edt(np.isnan(filled), return_distances=False, return_indices=True)
        out = filled[idx[0], idx[1]]
        out[~mask] = np.nan
        return out

    # 3: Isomap
    k = args.isomap_k
    while True:
        order = np.argsort(-cos, axis=1)[:, 1:k + 1]
        a = np.repeat(np.arange(m), k)
        b = order.ravel()
        g = scipy.sparse.csr_matrix((np.sqrt(np.clip(2 - 2 * cos[a, b], 1e-12, None)), (a, b)), shape=(m, m))
        if scipy.sparse.csgraph.connected_components(g, directed=False)[0] == 1:
            break
        k += 2
    d, paths = shortest_path_column(g, gc_id, gids)
    columns.append((f'Isomap: feature-space\nnearest neighbours (k {k})', to_map(d), [(gr[p], gc[p]) for p in paths],
                    [float(d[i]) for i in gids]))

    # 4: biharmonic distance on the spatial graph with feature affinities
    dl = np.clip(1 - cos[es, ed], 0, None)
    wgt = np.exp(-dl / np.median(dl))
    Wsp = scipy.sparse.csr_matrix((wgt, (es, ed)), shape=(m, m)).toarray()
    Wsp = np.maximum(Wsp, Wsp.T)
    lam, phi = scipy.linalg.eigh(np.diag(Wsp.sum(1)) - Wsp)
    d = embedding_column(phi[:, 1:] / lam[1:], gc_id)
    columns.append(('biharmonic: spatial graph,\nfeature affinity (all paths)', to_map(d), None, [float(d[i]) for i in gids]))

    # 5: spectral embedding of the dense feature affinity (Deep Spectral Methods)
    Wf = np.clip(cos, 0, None)
    np.fill_diagonal(Wf, 0)
    deg = Wf.sum(1)
    _, phi = scipy.linalg.eigh(np.diag(deg) - Wf, np.diag(deg), subset_by_index=[0, args.num_eig])
    d = embedding_column(phi[:, 1:], gc_id)
    columns.append((f'spectral: dense feature affinity,\n{args.num_eig} eigenvectors', to_map(d), None, [float(d[i]) for i in gids]))

    # ---- figure ----
    colours = ['magenta', 'lime', 'orange', 'cyan', 'white', 'red']
    fig, axes = plt.subplots(2, len(columns), figsize=(len(columns) * W / 60, 2 * H / 60 + 1.0), dpi=110)
    print('method | ' + ' | '.join(f'N{i + 1} {nb}' for i, nb in enumerate(nbrs)))
    for j, (title, dmap, paths, vals) in enumerate(columns):
        print(title.replace('\n', ' ') + ' | ' + ' | '.join(f'{v:.3f}' for v in vals))
        axes[0, j].imshow(img)
        axes[0, j].contour(mask, [0.5], colors='yellow', linewidths=0.8)
        im = axes[1, j].imshow(dmap, cmap='turbo', vmin=0, vmax=1)
        axes[1, j].contour(np.nan_to_num(dmap, nan=2.0), np.arange(0.1, 1.0, 0.1), colors='k', linewidths=0.4, alpha=0.6)
        for k_ax in range(2):
            ax = axes[k_ax, j]
            if paths is not None and k_ax == 0:
                for i, (pr, pc) in enumerate(paths):
                    ax.plot(pc, pr, '-', color='k', lw=4.0)
                    ax.plot(pc, pr, '-', color=colours[i % len(colours)], lw=2.2)
            for i, nb in enumerate(nbrs):
                col = colours[i % len(colours)]
                ax.plot(nb[1], nb[0], 'o', ms=9, mfc=col, mec='k', mew=1.2)
                ax.text(nb[1] + 7, nb[0] - 7, f'N{i + 1} {vals[i]:.2f}', color=col, fontsize=9, weight='bold',
                        bbox=dict(fc='k', alpha=0.65, pad=1.0, lw=0))
            ax.plot(centre[1], centre[0], '*', ms=16, mfc='cyan', mec='k', mew=1.2)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, j].set_title(title + ('' if paths is not None else '\n(distance only, no path)'), fontsize=9)
    axes[0, 0].set_ylabel('path', fontsize=10)
    axes[1, 0].set_ylabel('distance from C / diameter', fontsize=10)
    fig.suptitle(f'{args.title}  distance from C {centre} (row, col), each method divided by its own diameter', fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.colorbar(im, ax=axes[1, :], orientation='horizontal', fraction=0.03, pad=0.02, aspect=80)
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
    parser.add_argument('--grid_stride', type=int, required=True)
    parser.add_argument('--isomap_k', type=int, required=True)
    parser.add_argument('--num_eig', type=int, required=True)
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--output_path', type=str, required=True)
    main(parser.parse_args())
