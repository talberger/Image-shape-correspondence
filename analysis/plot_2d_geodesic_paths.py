"""Draw the 2D geodesic paths the relational loss measures, for one centre pixel and a few neighbours.

The distances come from analysis/build_geodesic_tables.py::build_2d_table itself (the function that
writes the training table), with the centre as its only table row. The paths are Dijkstra predecessors
on the same graph (8-connected over the mask's largest connected component, edges 1 / sqrt(2)), and
each path's length is asserted equal to the table value.
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
from build_geodesic_tables import build_2d_table, largest_connected_component, occlusion_domain


def mask_graph(mask):
    """Same graph as build_2d_table: 8-connected, weight = pixel distance."""
    H, W = mask.shape
    rows, cols = np.nonzero(mask)
    p2n = -np.ones((H, W), dtype=np.int64)
    p2n[rows, cols] = np.arange(len(rows))
    gr, gc, gw = [], [], []
    for dr, dc, w in [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, np.sqrt(2)),
                      (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]:
        nr, nc = rows + dr, cols + dc
        ok = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        src = np.nonzero(ok)[0]
        node = p2n[nr[ok], nc[ok]]
        on = node >= 0
        gr.append(src[on]); gc.append(node[on]); gw.append(np.full(on.sum(), w))
    graph = scipy.sparse.csr_matrix((np.concatenate(gw), (np.concatenate(gr), np.concatenate(gc))),
                                    shape=(len(rows), len(rows)))
    return graph, p2n, rows, cols


def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    img = np.array(Image.open(args.image_path).convert('RGB'))
    visible = torch.load(args.object_mask_path, map_location='cpu').numpy().astype(bool)
    if args.geodesic_domain_mask_path:
        visible, domain = occlusion_domain(
            visible, torch.load(args.geodesic_domain_mask_path, map_location='cpu').numpy())
    else:
        visible, _ = largest_connected_component(visible)
        domain = visible

    centre = tuple(args.center)
    nbrs = [tuple(args.neighbors[i:i + 2]) for i in range(0, len(args.neighbors), 2)]
    for p in [centre] + nbrs:
        if not visible[p]:
            raise ValueError(f'{p} (row, col) is not a pixel the training can sample')

    # The training table, one row: the centre.
    table, _, _, p2n_t, diameter = build_2d_table(
        torch.tensor([centre], dtype=torch.int64), domain, compute_true_diameter=True,
        endpoint_mask=None if domain is visible else visible)

    graph, p2n, rows, cols = mask_graph(domain)
    dist, pred = scipy.sparse.csgraph.dijkstra(graph, indices=p2n[centre], directed=False,
                                               return_predecessors=True)

    fig, ax = plt.subplots(figsize=(img.shape[1] / 20, img.shape[0] / 20), dpi=110)
    ax.imshow(img)
    ax.contour(domain, [0.5], colors='yellow', linewidths=1.0)
    colours = ['magenta', 'lime', 'orange', 'cyan', 'white', 'red']
    print(f'true 2D geodesic diameter = {diameter:.2f} px')
    print('neighbour (row, col) | geodesic px | euclid px | geo / euclid | norm_d_2D = geo / diameter')
    for i, nb in enumerate(nbrs):
        d_table = float(table[0, p2n_t[nb]])
        node, path = p2n[nb], []
        while node != p2n[centre]:
            path.append(node)
            node = pred[node]
        path.append(node)
        pr, pc = rows[path], cols[path]
        length = float(np.hypot(np.diff(pr), np.diff(pc)).sum())
        assert abs(length - d_table) < 1e-2 and abs(dist[p2n[nb]] - d_table) < 1e-2, (length, d_table)
        euc = float(np.hypot(nb[0] - centre[0], nb[1] - centre[1]))
        print(f'N{i + 1} {nb} | {d_table:.1f} | {euc:.1f} | {d_table / euc:.3f} | {d_table / diameter:.3f}')
        c = colours[i % len(colours)]
        ax.plot(pc, pr, '-', color=c, lw=2.5)
        ax.plot(nb[1], nb[0], 'o', ms=11, mfc=c, mec='k', mew=1.5)
        ax.text(nb[1] + 6, nb[0] - 6, f'N{i + 1}  geo {d_table:.0f} px  (euclid {euc:.0f})\nnorm_d_2D {d_table / diameter:.3f}',
                color=c, fontsize=10, weight='bold', bbox=dict(fc='k', alpha=0.6, pad=1.5, lw=0))
    ax.plot(centre[1], centre[0], '*', ms=20, mfc='cyan', mec='k', mew=1.5)
    ax.set_xticks(range(0, img.shape[1], 20)); ax.set_yticks(range(0, img.shape[0], 20))
    ax.grid(color='w', lw=0.4, alpha=0.35); ax.tick_params(labelsize=8)
    ax.set_title(f'{args.title}  2D geodesic paths from C {centre} (row, col); diameter {diameter:.0f} px', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output_path)
    print(f'saved {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--object_mask_path', type=str, required=True)
    parser.add_argument('--geodesic_domain_mask_path', type=str, default=None)
    parser.add_argument('--center', type=int, nargs=2, required=True, help='row col')
    parser.add_argument('--neighbors', type=int, nargs='+', required=True, help='row col row col ...')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--output_path', type=str, required=True)
    main(parser.parse_args())
