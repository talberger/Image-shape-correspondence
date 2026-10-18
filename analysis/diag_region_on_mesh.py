"""Where do the pixels of an image BOX land on the WHOLE mesh, per arm? (Tal, 2026-10-06)
Rows = arms; columns = photo (box pixels coloured by image row) | mesh from the x- flank | from the x+ flank | front view.
Landing point = the soft-snap vertex of each pixel (top1_vertex_idx of a *_soft_snap_correspondence.pth).

Usage: python analysis/diag_region_on_mesh.py --img <png> --box R0 R1 C0 C1 --arm 'ZS=<correspondence.pth>' ... --out f.png
"""
import argparse, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from modules.utils import loadmesh
ap = argparse.ArgumentParser()
ap.add_argument('--img', required=True); ap.add_argument('--box', type=int, nargs=4, required=True)
ap.add_argument('--arm', action='append', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--obj_path', default='./demo/shic/shapes/horse.obj'); ap.add_argument('--name', default='shic_horse')
a = ap.parse_args()
m = loadmesh(dir=a.obj_path, name=a.name, load_rings=False); V = m.vertices.cpu().numpy(); F = m.faces.cpu().numpy()
img = np.asarray(Image.open(a.img).convert('RGB')).astype(float) / 255
R0, R1, C0, C1 = a.box; cmap = plt.cm.viridis
VIEWS = [('mesh, x- flank', (2, 1), 0, -1), ('mesh, x+ flank', (2, -1), 0, 1), ('mesh, front view', (0, 1), 2, 1)]   # (h axis, sign), depth axis, sign
def view(ax, h, d, ds, pts, col):
    P = np.stack([V[:, h[0]] * h[1], V[:, 1], V[:, d] * ds], 1); T = P[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    sh = .6 + .4 * np.abs(n @ (np.array([.3, .5, .8]) / np.linalg.norm([.3, .5, .8]))); o = np.argsort(T[:, :, 2].mean(1))
    ax.add_collection(PolyCollection(T[o][:, :, :2], facecolors=np.outer(sh[o], [.86, .86, .86]), edgecolors='face', linewidths=.2))
    front = P[pts, 2] >= 0 if d == 0 else np.ones(len(pts), bool)                     # landing vertices on the half facing the viewer
    ax.scatter(P[pts[front], 0], P[pts[front], 1], s=7, c=col[front], linewidths=0, zorder=4)
    ax.set_xlim(P[:, 0].min() - .03, P[:, 0].max() + .03); ax.set_ylim(P[:, 1].min() - .03, P[:, 1].max() + .03); ax.set_aspect('equal'); ax.axis('off')
arms = [x.split('=', 1) for x in a.arm]
fig, axs = plt.subplots(len(arms), 4, figsize=(21, 5.2 * len(arms)), gridspec_kw={'width_ratios': [.8, 1.5, 1.5, .8]})
for i, (lab, p) in enumerate(arms):
    k = torch.load(p, map_location='cpu'); r, c, v = k['pixel_row'].numpy(), k['pixel_col'].numpy(), k['top1_vertex_idx'].numpy()
    s = (r >= R0) & (r <= R1) & (c >= C0) & (c <= C1); r, c, v = r[s], c[s], v[s]
    col = cmap((r - R0) / max(R1 - R0, 1))
    o = img * .5 + .3; o[r, c] = col[:, :3]
    ax = axs[i, 0]; ax.imshow(o[max(R0 - 15, 0):R1 + 5, C0 - 25:C1 + 25], interpolation='nearest'); ax.axis('off')
    ax.set_title(('box pixels, colour = image row\n' if i == 0 else '') + lab, fontsize=15, weight='bold')
    for j, (t, h, d, ds) in enumerate(VIEWS):
        view(axs[i, j + 1], h, d, ds, v, col)
        axs[i, j + 1].set_title(t + '  (only points on the side facing the viewer)', fontsize=13)
fig.tight_layout(); fig.savefig(a.out, dpi=100); print('saved', a.out)
