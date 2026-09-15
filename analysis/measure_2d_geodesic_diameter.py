"""
Measure the TRUE 2D geodesic diameter of the mask, over all pixel pairs.

2026-09-08, Tal's question. The `--geodesic` relational loss normalizes d_2D by `geo_diag_2d`,
which build_geodesic_tables.py computes as the max over the table it happens to build --
[2048 sampled centers x all masked pixels]. That is a max over 2048 SOURCES, not over all pairs, so
it can only ever UNDERESTIMATE the real diameter: if no sampled center sits at either end of the
longest path, the longest path is never measured. The 3D side has no such issue -- its table is all
3886x3886 pairs, so its `diagonal` is the true max.

An underestimated mu_2D inflates every norm_d_2D, and the loss then pushes the 3D side to spread
further than it should -- which is the direction the observed behaviour actually goes (coverage up,
accuracy down). Hence: measure it.

All-pairs Dijkstra over 212k nodes is not affordable, so this uses the standard double sweep:
Dijkstra from a node -> its farthest node u -> Dijkstra from u -> its farthest node v. d(u,v) is a
lower bound on the diameter, and iterating until it stops improving is exact for trees and
extremely tight for a compact planar region like an object mask. Each sweep is one Dijkstra over
the same 8-connected graph build_2d_table uses (1.0 orthogonal, sqrt(2) diagonal), so the numbers
are directly comparable to the stored `diagonal` -- this is not a different distance.

Run:
cd /workspace/bsb && python analysis/measure_2d_geodesic_diameter.py \
  --mask_path demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OFFSETS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]


def build_graph(mask):
    """Byte-for-byte the same 8-connected graph build_2d_table constructs."""
    H, W = mask.shape
    rows, cols = np.nonzero(mask)
    n = rows.shape[0]
    pixel_to_node = -np.ones((H, W), dtype=np.int64)
    pixel_to_node[rows, cols] = np.arange(n)

    r_list, c_list, w_list = [], [], []
    for dr, dc, w in OFFSETS:
        nr, nc = rows + dr, cols + dc
        valid = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        src = np.nonzero(valid)[0]
        nbr = pixel_to_node[nr[valid], nc[valid]]
        on = nbr >= 0
        r_list.append(src[on]); c_list.append(nbr[on])
        w_list.append(np.full(on.sum(), w, dtype=np.float64))

    graph = scipy.sparse.csr_matrix(
        (np.concatenate(w_list), (np.concatenate(r_list), np.concatenate(c_list))), shape=(n, n))
    return graph, pixel_to_node, rows, cols


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mask_path', required=True)
    p.add_argument('--geo_table_2d_path', default=None,
                   help='if given, compares against the diameter stored there')
    p.add_argument('--max_sweeps', type=int, default=8)
    args = p.parse_args()

    m = torch.load(args.mask_path, map_location='cpu')
    mask = m['mask'] if isinstance(m, dict) and 'mask' in m else m
    mask = mask.numpy() if torch.is_tensor(mask) else np.asarray(mask)
    mask = mask.astype(bool)

    # The stored table drops disconnected specks; mirror that so node counts line up.
    graph, pixel_to_node, rows, cols = build_graph(mask)
    n_comp, comp = scipy.sparse.csgraph.connected_components(graph, directed=False)
    if n_comp > 1:
        keep = np.argmax(np.bincount(comp))
        sub = np.nonzero(comp == keep)[0]
        print(f'{n_comp} components; keeping the largest ({sub.size}/{comp.size} px)')
        graph = graph[sub][:, sub]
        rows, cols = rows[sub], cols[sub]
    print(f'graph: {graph.shape[0]} nodes, {graph.nnz} edges')

    src = 0
    best = 0.0
    seen = set()
    for sweep in range(args.max_sweeps):
        d = scipy.sparse.csgraph.dijkstra(graph, indices=src, directed=False)
        d[~np.isfinite(d)] = -1
        far = int(np.argmax(d))
        far_d = float(d[far])
        print(f'  sweep {sweep + 1}: from ({rows[src]},{cols[src]}) -> farthest '
              f'({rows[far]},{cols[far]}) at {far_d:.2f} px')
        best = max(best, far_d)
        if far in seen:
            break
        seen.add(src)
        src = far

    print(f'\nTRUE 2D geodesic diameter (double sweep): {best:.2f} px')
    if args.geo_table_2d_path:
        stored = float(torch.load(args.geo_table_2d_path, map_location='cpu')['diagonal'])
        print(f'stored geo_diag_2d (max over sampled centers): {stored:.2f} px')
        print(f'underestimate: {best - stored:.2f} px = {100 * (best - stored) / best:.2f}%')
        print(f'implied mu_3d_scale-equivalent correction: {stored / best:.4f}')


if __name__ == '__main__':
    main()
