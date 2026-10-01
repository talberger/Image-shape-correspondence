"""
Proportion diagnostic (2026-09-16, saved as a script 2026-09-17): 3D geodesic extent of the set of
vertices hit by the top1 pixel->vertex match, as a fraction of the mesh geodesic diameter D.
Reports max and p99 of the pairwise geodesic distances over the unique top1 vertices.
Reference values on the bear: SHIC 0.706/0.641, old mu=0.6 0.624/0.488, MAIN 0.795/0.654, ZS 0.907/0.727.
"""
import argparse
import sys

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--geo_table_3d_path', required=True)
parser.add_argument('--knn_paths', nargs='+', required=True)
parser.add_argument('--labels', nargs='+', default=None)
args = parser.parse_args()

g = torch.load(args.geo_table_3d_path, map_location='cpu')
table, D = g['table'].float(), float(g['diagonal'])
labels = args.labels or args.knn_paths
for label, path in zip(labels, args.knn_paths):
    c = torch.load(path, map_location='cpu')
    v = torch.unique(c['top1_vertex_idx'])
    sub = table[v][:, v]
    iu = torch.triu_indices(len(v), len(v), 1)
    d = sub[iu[0], iu[1]]
    p99 = torch.quantile(d[torch.randperm(len(d))[:5_000_000]] if len(d) > 5_000_000 else d, 0.99)
    print(f'{label}: top1 vertices {len(v)}/{c["num_vertices"]}  extent max/D {d.max().item() / D:.3f}  p99/D {p99.item() / D:.3f}')
    sys.stdout.flush()
