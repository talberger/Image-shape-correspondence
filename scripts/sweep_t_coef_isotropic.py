"""Re-sweep the heat-method t_coef for a NEW mesh (2026-09-02).

build_geodesic_tables.build_3d_table's docstring is explicit: t_coef=20.0 was tuned against
meshes/guitar.obj's ~500x edge-length ratio, and "if this mesh's geometry ever changes, re-sweep
rather than assuming 20.0 still holds". The isotropic remesh takes that ratio from ~500x to ~30x
(edge-length CV 1.497 -> 0.230), so the old value has no reason to still be right.

Metric: undershoot of Euclidean distance, which geodesic can never be below -- so any undershoot is
pure approximation error. Measured on a random subset of source vertices (the full table is V solves;
a subset is enough to rank t_coef).
"""
import argparse, os, sys, time
import numpy as np
import potpourri3d as pp3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils import loadmesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj_path', required=True)
    ap.add_argument('--name', default='guitar')
    ap.add_argument('--n_sources', type=int, default=250)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--t_coefs', type=float, nargs='+',
                    default=[1, 2, 3, 5, 8, 10, 15, 20, 30, 50])
    args = ap.parse_args()

    mesh = loadmesh(dir=args.obj_path, name=args.name)
    verts = mesh.vertices.detach().cpu().numpy().astype(np.float64)
    faces = mesh.faces.detach().cpu().numpy().astype(np.int32)
    V = len(verts)
    rng = np.random.default_rng(args.seed)
    src = rng.choice(V, size=min(args.n_sources, V), replace=False)
    euc = np.sqrt(((verts[src][:, None, :] - verts[None, :, :]) ** 2).sum(-1))

    print(f'{args.obj_path}: V={V}, sweeping t_coef on {len(src)} random sources\n')
    print(f'{"t_coef":>7} {"undershoot %":>13} {"mean":>9} {"p99":>9} {"WORST":>9} {"sec":>6}')
    rows = []
    for tc in args.t_coefs:
        t0 = time.time()
        solver = pp3d.MeshHeatMethodDistanceSolver(verts, faces, t_coef=tc)
        d = np.stack([solver.compute_distance(int(v)) for v in src])
        u = euc - d
        pos = u[u > 0]
        frac = 100 * (u > 0).mean()
        mean = pos.mean() if pos.size else 0.0
        p99 = np.percentile(pos, 99) if pos.size else 0.0
        worst = u.max()
        rows.append((tc, frac, mean, p99, worst))
        print(f'{tc:>7.1f} {frac:>12.2f}% {mean:>9.5f} {p99:>9.5f} {worst:>9.5f} {time.time()-t0:>6.1f}')
    best = min(rows, key=lambda r: r[4])
    print(f'\nlowest worst-case undershoot: t_coef={best[0]:g} (worst {best[4]:.5f})')


if __name__ == '__main__':
    main()
