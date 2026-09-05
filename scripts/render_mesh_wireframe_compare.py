"""Side-by-side wireframe comparison of two meshes (2026-09-02).

Front view = projection onto the XY plane; the guitar's Z extent is 0.12 against 1.80 in Y, so this
IS the front view. Draws every triangle edge, which is the point -- the question being reviewed is
where the sample points are, not what the surface looks like shaded.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def load_obj(path):
    v, f = [], []
    for line in open(path):
        if line.startswith('v '):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            f.append([int(p.split('/')[0]) for p in line.split()[1:4]])
    return np.array(v), np.array(f) - 1


def segments(v, f):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return v[e][:, :, :2]


def panel(ax, v, f, title, xlim=None, ylim=None, lw=0.3, dots=False):
    ax.add_collection(LineCollection(segments(v, f), colors='#2b6cb0', linewidths=lw, alpha=0.85))
    if dots:
        ax.plot(v[:, 0], v[:, 1], '.', color='#c53030', markersize=1.2)
    ax.set_xlim(*(xlim if xlim else (v[:, 0].min(), v[:, 0].max())))
    ax.set_ylim(*(ylim if ylim else (v[:, 1].min(), v[:, 1].max())))
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh_a', required=True)
    ap.add_argument('--mesh_b', required=True)
    ap.add_argument('--label_a', default='A')
    ap.add_argument('--label_b', default='B')
    ap.add_argument('--output_path', required=True)
    ap.add_argument('--zoom_ymin', type=float, default=None, help='neck-zoom band, world Y')
    ap.add_argument('--zoom_ymax', type=float, default=None)
    args = ap.parse_args()

    va, fa = load_obj(args.mesh_a)
    vb, fb = load_obj(args.mesh_b)

    lo = min(va[:, 1].min(), vb[:, 1].min()); hi = max(va[:, 1].max(), vb[:, 1].max())
    zy0 = args.zoom_ymin if args.zoom_ymin is not None else lo + 0.62 * (hi - lo)
    zy1 = args.zoom_ymax if args.zoom_ymax is not None else hi
    zx = (min(va[:, 0].min(), vb[:, 0].min()), max(va[:, 0].max(), vb[:, 0].max()))

    fig, axes = plt.subplots(2, 2, figsize=(11, 13))
    for col, (v, f, lab) in enumerate([(va, fa, args.label_a), (vb, fb, args.label_b)]):
        panel(axes[0][col], v, f, f'{lab}\n{len(v)} vertices, {len(f)} faces', lw=0.25)
        panel(axes[1][col], v, f, f'{lab} — neck / headstock zoom (vertices in red)',
              xlim=zx, ylim=(zy0, zy1), lw=0.6, dots=True)
    fig.suptitle('Isotropic remesh — front view (XY), every triangle edge drawn', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=170, bbox_inches='tight')
    print('saved ->', args.output_path)


if __name__ == '__main__':
    main()
