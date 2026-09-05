"""Isotropic remesh of meshes/guitar.obj (2026-09-02, Tal's request).

The original mesh is badly anisotropic -- edge lengths span 0.0006..0.32 (500x) and the neck is
carried by a handful of very large triangles, so there are almost no sample points there. This runs
MeshLab's isotropic explicit remesher (via pymeshlab) and bisects the target edge length to land the
vertex count near the original 4036 (Tal: "4300 is fine, it is flexible").

Writes a NEW file; meshes/guitar.obj is never touched.
"""
import argparse
import numpy as np
import pymeshlab as ml


def stats(v, f):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    L = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    a = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) / 2
    return dict(V=len(v), F=len(f), edge_mean=L.mean(), edge_med=np.median(L), edge_min=L.min(),
                edge_max=L.max(), edge_cv=L.std() / L.mean(), area_total=a.sum(),
                area_ratio=a.max() / np.median(a))


def show(tag, s):
    print(f'{tag:24s} V={s["V"]:5d} F={s["F"]:5d}  edge mean {s["edge_mean"]:.4f} med {s["edge_med"]:.4f} '
          f'min {s["edge_min"]:.4f} max {s["edge_max"]:.4f} CV {s["edge_cv"]:.3f}  '
          f'area(max/med) {s["area_ratio"]:.1f}  total area {s["area_total"]:.4f}')


def remesh(src, targetlen, iterations):
    ms = ml.MeshSet()
    ms.load_new_mesh(src)
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=ml.PureValue(targetlen), iterations=iterations,
        adaptive=False,          # isotropic: uniform edge length, not curvature-adaptive
        checksurfdist=True,      # keep the remesh on the original surface
        maxsurfdist=ml.PercentageValue(0.5))
    m = ms.current_mesh()
    return ms, m.vertex_matrix(), m.face_matrix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='meshes/guitar.obj')
    ap.add_argument('--out', default='meshes/guitar_isotropic.obj')
    ap.add_argument('--target_vertices', type=int, default=4036)
    ap.add_argument('--tolerance', type=float, default=0.04, help='accept within this fraction')
    ap.add_argument('--iterations', type=int, default=10)
    ap.add_argument('--max_bisect', type=int, default=14)
    args = ap.parse_args()

    ms0 = ml.MeshSet(); ms0.load_new_mesh(args.src)
    m0 = ms0.current_mesh()
    s0 = stats(m0.vertex_matrix(), m0.face_matrix())
    show('ORIGINAL', s0)

    # equilateral-triangle estimate: F ~ 2V, area per face = total/F, A = sqrt(3)/4 * L^2
    lo_guess = np.sqrt(4 * (s0['area_total'] / (2 * args.target_vertices)) / np.sqrt(3))
    print(f'\nanalytic target edge length for {args.target_vertices} vertices: {lo_guess:.5f}')

    # bisect on targetlen: larger targetlen -> fewer vertices
    lo, hi = lo_guess * 0.4, lo_guess * 2.5
    best = None
    for i in range(args.max_bisect):
        mid = 0.5 * (lo + hi)
        ms, v, f = remesh(args.src, mid, args.iterations)
        s = stats(v, f)
        show(f'  targetlen {mid:.5f}', s)
        if best is None or abs(s['V'] - args.target_vertices) < abs(best[2]['V'] - args.target_vertices):
            best = (mid, ms, s)
        if abs(s['V'] - args.target_vertices) <= args.tolerance * args.target_vertices:
            break
        if s['V'] > args.target_vertices:
            lo = mid   # too many vertices -> longer edges
        else:
            hi = mid
    targetlen, ms, s = best
    ms.save_current_mesh(args.out, save_vertex_normal=False, save_vertex_color=False)
    print(f'\nchosen targetlen {targetlen:.5f}, {args.iterations} iterations')
    show('ORIGINAL', s0)
    show('REMESHED', s)
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
