"""Alternative mesh colourings for judging LEG mapping (Tal, 2026-10-06), shown as colour key + back-projection per arm.

  legs4      body grey, each of the 4 legs one colour
  legbands   body grey, each leg its own hue, split into height bands (same leg + same height = one colour)
  shicstyle  legbands on the legs + a side-view checker on the body (as in SHIC's figures)

Usage: python analysis/leg_colorings.py --tag lvis184389 --run_dir <instance dir> --out_dir <dir>
         --arm 'ZS=<correspondence.pth>' ... (soft-snap / SHIC *_correspondence.pth files, top1_vertex_idx)
"""
import argparse, colorsys, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from modules.utils import loadmesh

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True); ap.add_argument('--run_dir', required=True); ap.add_argument('--out_dir', required=True)
ap.add_argument('--obj_path', default='./demo/shic/shapes/horse.obj'); ap.add_argument('--name', default='shic_horse')
ap.add_argument('--arm', action='append', required=True, metavar='LABEL=CORRESPONDENCE_PTH')
ap.add_argument('--leg_cut', type=float, default=None, help='mesh height (y) below which a vertex is a leg; default = found automatically')
ap.add_argument('--bands', type=int, default=6)
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

mesh = loadmesh(dir=a.obj_path, name=a.name, load_rings=False)
V = mesh.vertices.cpu().numpy(); F = mesh.faces.cpu().numpy()
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
E_ = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
def comps(cut):
    keep = V[:, 1] < cut; e = E_[keep[E_[:, 0]] & keep[E_[:, 1]]]
    n, lab = connected_components(sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V),) * 2), directed=False)
    lab = np.where(keep, lab, -1); ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
    return lab, ids[cnt >= 40]
if a.leg_cut is None:                                   # the highest cut that still gives exactly 4 separate legs
    for cut in np.arange(0.0, -0.4, -0.005):
        lab, big = comps(cut)
        if len(big) == 4: a.leg_cut = float(cut); break
lab, big = comps(a.leg_cut)
assert len(big) == 4, 'leg cut %.3f gives %d components' % (a.leg_cut, len(big))
leg = np.zeros(len(V), int)                              # 1 front x+ | 2 front x- | 3 hind x+ | 4 hind x-
for c in big:
    m = lab == c; leg[m] = 1 + 2 * int(V[m, 2].mean() < 0) + int(V[m, 0].mean() < 0)
NAMES = {1: 'front, x+ side', 2: 'front, x- side', 3: 'hind, x+ side', 4: 'hind, x- side'}
print('leg cut y < %.3f | vertices per leg:' % a.leg_cut, {NAMES[k]: int((leg == k).sum()) for k in NAMES})

HUE = {1: 0.00, 2: 0.11, 3: 0.60, 4: 0.36}               # red, orange, blue, green
GREY = np.array([.78, .78, .78])
Y0 = V[leg > 0, 1].min()
def legs4(X=V, L=leg):
    c = np.tile(GREY, (len(X), 1))
    for k in HUE: c[L == k] = colorsys.hsv_to_rgb(HUE[k], .9, .95)
    return c
def legbands(X=V, L=leg, body=None):
    c = np.tile(GREY, (len(X), 1)) if body is None else body.copy()
    for k in HUE:
        m = L == k; b = np.clip(((X[m, 1] - Y0) / (a.leg_cut - Y0) * a.bands).astype(int), 0, a.bands - 1)
        val = np.where(b % 2 == 0, 1.0, 0.55); sat = 0.45 + 0.5 * b / (a.bands - 1)      # alternate light/dark, paler towards the hoof
        c[m] = np.array([colorsys.hsv_to_rgb(HUE[k], s_, v_) for s_, v_ in zip(sat, val)]).reshape(-1, 3)
    return c
def shicstyle(X=V, L=leg):
    pal = np.array([[.55, .12, .30], [.20, .35, .15], [.90, .75, .30], [.25, .55, .80], [.85, .50, .15], [.20, .30, .65]])
    cell = 0.105; i = np.floor(X[:, 2] / cell).astype(int); j = np.floor(X[:, 1] / cell).astype(int)
    return legbands(X, L, pal[(i * 2 + j * 3) % len(pal)])
FN = {'legs4': legs4, 'legbands': legbands, 'shicstyle': shicstyle}
COLS = {k: f() for k, f in FN.items()}
# colour key only: every face split 16x and coloured at its centroid, so cell borders are straight lines rather than
# following the triangle edges. The back-projection still reads the per-VERTEX colours above.
T3 = V[F]; FL = np.array([np.bincount(leg[f], minlength=5).argmax() for f in F])
for _ in range(2):
    A, B, C = T3[:, 0], T3[:, 1], T3[:, 2]; ab, bc, ca = (A + B) / 2, (B + C) / 2, (C + A) / 2
    T3 = np.concatenate([np.stack(t, 1) for t in ((A, ab, ca), (ab, B, bc), (ca, bc, C), (ab, bc, ca))]); FL = np.tile(FL, 4)
TITLE = {'legs4': '1. one colour per leg', 'legbands': '2. per leg: hue = which leg, bands = height',
         'shicstyle': '3. as 2 on the legs + side-view checker on the body'}

def render(ax, name, side):
    """orthographic side view, not mirrored; side = +1 looks at the x+ flank, -1 at the x- flank"""
    P = np.stack([-T3[..., 2] * side, T3[..., 1], T3[..., 0] * side], -1)
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = .55 + .45 * np.abs(n @ (np.array([.3, .5, .8]) / np.linalg.norm([.3, .5, .8])))
    rgb = FN[name](T3.mean(1), FL)
    o = np.argsort(P[:, :, 2].mean(1))
    ax.add_collection(PolyCollection(P[o][:, :, :2], facecolors=np.clip(rgb[o] * shade[o, None], 0, 1), edgecolors='face', linewidths=.1, rasterized=True))
    ax.set_xlim(P[..., 0].min() - .02, P[..., 0].max() + .02); ax.set_ylim(P[..., 1].min() - .02, P[..., 1].max() + .02); ax.set_aspect('equal'); ax.axis('off')

T, D = a.tag, a.run_dir
img = np.asarray(Image.open(f'{D}/{T}.png').convert('RGB')).astype(float) / 255
gt = torch.load(f'{D}/{T}_gt.pth')
arms = [x.split('=', 1) for x in a.arm]
corr = {l: torch.load(p, map_location='cpu') for l, p in arms}
for name, vc in COLS.items():
    torch.save({'vertex_colors': torch.from_numpy((vc * 255).astype(np.uint8)), 'mode': name, 'obj_path': a.obj_path, 'leg_cut': a.leg_cut},
               f'{a.out_dir}/horse_mesh_{name}_color.pth')
    fig, axs = plt.subplots(1, 3 + len(arms), figsize=(4.4 * (3 + len(arms)), 3.9), gridspec_kw={'width_ratios': [1, 1, 1.15] + [1.15] * len(arms)})
    render(axs[0], name, +1); axs[0].set_title('colour key: x+ flank', fontsize=11)
    render(axs[1], name, -1); axs[1].set_title('colour key: x- flank', fontsize=11)
    axs[2].imshow(img); axs[2].set_title('crop + GT points', fontsize=11)
    for ax, (l, _) in zip(axs[3:], arms):
        k = corr[l]; o = img * 0.55 + 0.25
        o[k['pixel_row'].numpy(), k['pixel_col'].numpy()] = vc[k['top1_vertex_idx'].numpy()]
        ax.imshow(o, interpolation='nearest'); ax.set_title(l, fontsize=11)
    for ax in axs[2:]:
        ax.axis('off')
        for i, (r, c) in enumerate(zip(gt['rows'].tolist(), gt['cols'].tolist())):
            ax.plot(c, r, 'o', ms=7, mfc='none', mec='k', mew=2.4); ax.plot(c, r, 'o', ms=7, mfc='none', mec='w', mew=1.2)
    fig.suptitle(f'{T}   colouring {TITLE[name]}', fontsize=13, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(f'{a.out_dir}/{T}_colouring_{name}.png', dpi=115); plt.close(fig)
    print('saved', f'{a.out_dir}/{T}_colouring_{name}.png')
