"""Add the guitar neck's MISSING RINGS to meshes/guitar.obj, changing nothing else.

Why (2026-09-16, Tal): the neck of guitar.obj is a tube of 13 rings of 36 vertices, spaced
~0.05-0.066 apart -- except between y=-0.0267 and y=+0.2944, where there are NO rings at all.
That 0.321 stretch (~15% of the guitar's length) is spanned by a single quad strip of 72 long
triangles, so a whole run of neck pixels has almost no vertex to map to. That is the leading
suspect for why the old mesh's back-projection shows one long uniform cell on the neck.

What this does: inserts `--num_rings` evenly spaced rings into that one strip, at the spacing the
rest of the neck already uses, and re-triangulates the strip. Nothing else in the mesh is touched.

Exactness. Every new vertex is placed on an EXISTING MESH EDGE (a rail of the quad strip), at a
linear parameter along it, so it lies exactly on the original surface -- the geometry is unchanged,
only the sampling. Original vertices keep their indices and coordinates byte-for-byte; the new ones
are appended. Only the strip's 72 faces are replaced (by num_bands*72); all other faces are copied
through verbatim.

Frame. Deliberately NO `.normalizer.json` sidecar (Tal, 2026-09-16: "do what Itai's original code
did"). The mesh loads through the historical centroid normalizer like guitar.obj always has. The
cost, stated up front: the centroid depends on the vertex set, so the added vertices shift the
frame slightly (reported at the end of this script) -- see PROJECT_NOTES.

Run:
  python analysis/densify_neck.py --obj_path ./meshes/guitar.obj \
      --out_path ./meshes/guitar_neckrings.obj --num_rings 5
"""
import argparse
import os

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def read_obj(path):
    """v / vn / f only -- guitar.obj has no vt, and its `f` entries are bare vertex indices."""
    V, N, F = [], [], []
    for ln in open(path):
        p = ln.split()
        if not p:
            continue
        if p[0] == 'v':
            V.append([float(x) for x in p[1:4]])
        elif p[0] == 'vn':
            N.append([float(x) for x in p[1:4]])
        elif p[0] == 'f':
            F.append([int(x.split('/')[0]) - 1 for x in p[1:4]])
        elif p[0] in ('vt', 'usemtl', 'mtllib', 's', 'o', 'g'):
            raise ValueError(f'{path} carries a {p[0]!r} record this writer would drop')
    return np.array(V), np.array(N), np.array(F, dtype=np.int64)


def write_obj(path, V, N, F):
    """Same interleaved v/vn layout guitar.obj uses, so a diff against the original is readable."""
    with open(path, 'w') as fh:
        for v, n in zip(V, N):
            fh.write('v %.6f %.6f %.6f\n' % tuple(v))
            fh.write('vn %.6f %.6f %.6f\n' % tuple(n))
        for f in F:
            fh.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))


def find_gap_strip(V, F, x_halfwidth, y_lo, y_hi, ring_tol):
    """Locate the longest ring-to-ring gap in the neck and return (ringA, ringB, strip face ids).

    Rings are found by clustering neck vertices on y; the gap is the largest spacing between
    consecutive clusters. Returns the two rings bounding it, ordered low-y first.
    """
    neck = np.where((np.abs(V[:, 0]) < x_halfwidth) & (V[:, 1] > y_lo) & (V[:, 1] < y_hi))[0]
    order = neck[np.argsort(V[neck, 1])]
    rings = [[order[0]]]
    for idx in order[1:]:
        if V[idx, 1] - V[rings[-1][-1], 1] < ring_tol:
            rings[-1].append(idx)
        else:
            rings.append([idx])
    centers = np.array([V[r, 1].mean() for r in rings])
    gap = int(np.argmax(np.diff(centers)))
    A, B = np.array(rings[gap]), np.array(rings[gap + 1])
    As, Bs = set(A.tolist()), set(B.tolist())
    strip = [i for i, f in enumerate(F)
             if set(f) <= (As | Bs) and set(f) & As and set(f) & Bs]
    return A, B, np.array(strip), float(centers[gap + 1] - centers[gap]), centers


def rail_matching(V, F, A, B, strip):
    """Pair each ring-A vertex with one ring-B vertex ACROSS AN EXISTING MESH EDGE.

    The strip's cross edges are 36 rails plus 36 diagonals and nothing distinguishes them by
    length, so the rails are recovered as the perfect matching that best lines the two rings up
    angularly around the tube axis -- restricted to edges that actually exist, which is what keeps
    every interpolated point on the surface.
    """
    As, Bs = set(A.tolist()), set(B.tolist())
    cross = set()
    for i in strip:
        f = F[i]
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            if u in As and v in Bs:
                cross.add((int(u), int(v)))
            elif v in As and u in Bs:
                cross.add((int(v), int(u)))

    axis = np.array([0.0, 1.0, 0.0])
    def angles(R):
        c = V[R].mean(0)
        d = V[R] - c
        d = d - np.outer(d @ axis, axis)
        return np.arctan2(d[:, 2], d[:, 0])
    angA, angB = angles(A), angles(B)
    posA = {int(a): k for k, a in enumerate(A)}
    posB = {int(b): k for k, b in enumerate(B)}

    BIG = 1e6
    cost = np.full((len(A), len(B)), BIG)
    for a, b in cross:
        d = abs(angA[posA[a]] - angB[posB[b]])
        cost[posA[a], posB[b]] = min(d, 2 * np.pi - d)
    r, c = linear_sum_assignment(cost)
    if cost[r, c].max() >= BIG:
        raise RuntimeError('no perfect rail matching along existing cross edges')
    return {int(A[i]): int(B[j]) for i, j in zip(r, c)}


def ring_cycle(F, strip, A):
    """Ring A's 36 vertices in cyclic order, walked along the strip's in-ring edges."""
    As = set(A.tolist())
    adj = defaultdict(set)
    for i in strip:
        f = F[i]
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            if u in As and v in As:
                adj[int(u)].add(int(v))
                adj[int(v)].add(int(u))
    if any(len(adj[int(a)]) != 2 for a in A):
        raise RuntimeError('ring A is not a simple cycle')
    start = int(A[0])
    order, prev = [start], None
    while True:
        nxt = [x for x in adj[order[-1]] if x != prev][0]
        prev = order[-1]
        if nxt == start:
            break
        order.append(nxt)
    if len(order) != len(A):
        raise RuntimeError(f'ring A cycle covered {len(order)} of {len(A)} vertices')
    return order


def densify(V, N, F, A, B, strip, rails, num_rings):
    """Insert num_rings rings along the rails and re-triangulate the strip's 36 quads.

    Each new vertex sits at a fixed parameter along an existing RAIL edge, so it lies exactly on
    the original surface. The quads themselves are re-triangulated, which is the one place the
    surface is not reproduced bit-for-bit: a quad here is non-planar by at most 1.2e-4 (measured),
    and the sub-quads follow the bilinear patch rather than the original single diagonal. main()
    reports the number.
    """
    bands = num_rings + 1
    Vn, Nn = list(V), list(N)
    new = {}
    for a in A:
        a = int(a)
        b = rails[a]
        new[a] = []
        for k in range(1, bands):
            t = k / bands
            Vn.append((1 - t) * V[a] + t * V[b])
            nrm = (1 - t) * N[a] + t * N[b]
            Nn.append(nrm / max(np.linalg.norm(nrm), 1e-12))
            new[a].append(len(Vn) - 1)

    def chain(a):
        """Vertex index at each of the bands+1 levels: ring A, the new rings, then ring B."""
        return [a] + new[a] + [rails[a]]

    dropped = set(strip.tolist())
    Fn = [list(f) for i, f in enumerate(F) if i not in dropped]
    order = ring_cycle(F, strip, A)
    ref = {i: [int(x) for x in F[i]] for i in strip}
    for k in range(len(order)):
        a0, a1 = order[k], order[(k + 1) % len(order)]
        c0, c1 = chain(a0), chain(a1)
        # a strip triangle of this quad, to copy the winding from
        quad = {a0, a1, rails[a0], rails[a1]}
        src = next(f for f in ref.values() if set(f) <= quad)
        for j in range(bands):
            Fn.extend(_orient([(c0[j], c1[j], c1[j + 1]),
                               (c0[j], c1[j + 1], c0[j + 1])], src, V, Vn))
    return np.array(Vn), np.array(Nn), np.array(Fn, dtype=np.int64)


def _orient(tris, ref, V, Vn):
    """Give each new triangle the same outward winding as the strip triangle it came from."""
    Vn = np.asarray(Vn)
    rn = np.cross(V[ref[1]] - V[ref[0]], V[ref[2]] - V[ref[0]])
    out = []
    for t in tris:
        p = Vn[list(t)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        out.append(list(t) if float(n @ rn) >= 0 else [t[0], t[2], t[1]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj_path', default='./meshes/guitar.obj')
    ap.add_argument('--out_path', required=True)
    ap.add_argument('--num_rings', type=int, default=5,
                    help='rings to insert into the gap (5 -> spacing 0.0535, matching the '
                         '0.059/0.066 of the neighbouring neck)')
    ap.add_argument('--x_halfwidth', type=float, default=0.075)
    ap.add_argument('--y_lo', type=float, default=-0.30)
    ap.add_argument('--y_hi', type=float, default=0.55)
    ap.add_argument('--ring_tol', type=float, default=0.004)
    args = ap.parse_args()

    V, N, F = read_obj(args.obj_path)
    print(f'loaded {args.obj_path}: {len(V)} vertices, {len(F)} faces')

    A, B, strip, gap, centers = find_gap_strip(V, F, args.x_halfwidth, args.y_lo, args.y_hi,
                                               args.ring_tol)
    spacing = np.diff(centers)
    print(f'neck rings: {len(centers)}, spacing median {np.median(spacing):.4f}')
    print(f'gap: {gap:.4f} between y={V[A, 1].mean():+.4f} and y={V[B, 1].mean():+.4f}, '
          f'|A|={len(A)} |B|={len(B)}, strip faces={len(strip)}')
    if len(strip) != 2 * len(A):
        raise RuntimeError(f'expected a quad strip of {2 * len(A)} faces, found {len(strip)}')

    rails = rail_matching(V, F, A, B, strip)
    Vn, Nn, Fn = densify(V, N, F, A, B, strip, rails, args.num_rings)

    # --- checks that must hold before this mesh is allowed near a training run ---
    assert np.array_equal(Vn[:len(V)], V), 'original vertices moved'
    expect_faces = len(F) - len(strip) + len(A) * (args.num_rings + 1) * 2
    assert len(Fn) == expect_faces, f'face count {len(Fn)} != {expect_faces}'
    added = len(Vn) - len(V)
    # every new vertex lies on the rail edge it came from -> on the original surface
    maxdev = 0.0
    for j, a in enumerate(A):
        a = int(a)
        seg = Vn[rails[a]] - Vn[a]
        for k in range(args.num_rings):
            w = Vn[len(V) + j * args.num_rings + k] - Vn[a]
            maxdev = max(maxdev, float(np.linalg.norm(w - seg * (w @ seg) / (seg @ seg))))
    # the one real surface change: the strip's quads are re-triangulated, and they are not exactly
    # planar, so the new faces follow the bilinear patch instead of the original diagonal
    order = ring_cycle(F, strip, A)
    nonplanar = 0.0
    for k in range(len(order)):
        a0, a1 = order[k], order[(k + 1) % len(order)]
        P = V[[a0, a1, rails[a1], rails[a0]]]
        nrm = np.cross(P[1] - P[0], P[2] - P[0])
        nrm /= np.linalg.norm(nrm)
        nonplanar = max(nonplanar, abs(float((P[3] - P[0]) @ nrm)))
    # manifold check: every edge used by exactly two faces
    ec = Counter()
    for f in Fn:
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            ec[(min(u, v), max(u, v))] += 1
    bad = [e for e, c in ec.items() if c != 2]
    print(f'checks: +{added} vertices, faces {len(F)} -> {len(Fn)}, '
          f'max off-rail deviation {maxdev:.2e}, non-manifold edges {len(bad)}')
    print(f'surface change: bounded by the strip quads\' non-planarity, max {nonplanar:.2e} '
          f'(guitar height ~1.80); every new vertex is exactly on a rail edge')
    if bad:
        raise RuntimeError(f'{len(bad)} edges are not shared by exactly two faces')

    def frame(x):
        s = x.mean(0)
        return s, float(np.linalg.norm(x - s, axis=1).max())
    s0, r0 = frame(V)
    s1, r1 = frame(Vn)
    print('frame drift under the historical centroid normalizer (no sidecar, by design): '
          f'centroid {np.linalg.norm(s1 - s0):.5f} (dY {s1[1] - s0[1]:+.5f}), '
          f'radius {r0:.6f} -> {r1:.6f} ({100 * (r1 - r0) / r0:+.3f}%)')

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    write_obj(args.out_path, Vn, Nn, Fn)
    print(f'wrote {args.out_path}: {len(Vn)} vertices, {len(Fn)} faces')


if __name__ == '__main__':
    main()
