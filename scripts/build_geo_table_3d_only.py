"""Build ONLY the 3D (mesh-surface) geodesic table.

analysis/build_geodesic_tables.py has --skip_3d but no --skip_2d, and the 2D table is pixel-space
(mask + fine segments only) so it is completely mesh-INDEPENDENT -- a mesh change must not force a
1.7 GB rebuild of it. This driver reuses build_3d_table() verbatim and writes the same file format.
"""
import argparse, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.build_geodesic_tables import build_3d_table

ap = argparse.ArgumentParser()
ap.add_argument('--obj_path', required=True)
ap.add_argument('--name', required=True, help='output is <out_dir>/<name>_geo_table_3d.pth')
ap.add_argument('--t_coef', type=float, required=True)
ap.add_argument('--undershoot_tol', type=float, default=0.15)
ap.add_argument('--out_dir', required=True)
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
table, diag = build_3d_table(args.obj_path, args.name, args.t_coef, args.undershoot_tol)
path = os.path.join(args.out_dir, f'{args.name}_geo_table_3d.pth')
torch.save({'table': table, 'diagonal': diag, 'obj_path': args.obj_path}, path)
print(f'saved 3D table {tuple(table.shape)} (diagonal={diag:.5f}, t_coef={args.t_coef}) to {path}')
