"""Per-segment mu, READABLE (Tal, 2026-09-27): one panel per run, on the photo.
  isotropic run  -> every segment tinted, labelled with its single mu.
  directional run-> per segment the ellipse with its two principal axes drawn and labelled: the mu a pixel pair gets
                    when its image offset runs ALONG that axis (red = major, blue = minor). The hybrid's cross-segment
                    ellipse is drawn in the top-right corner (magenta).
Usage: python analysis/plot_mu_annotated.py --fine_seg_path F --img_path I --runs LABEL=RUN_DIR ... --out fig.png"""
import argparse, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.ndimage import distance_transform_edt


def rep_pixel(mask):
    """the segment's most INTERIOR pixel (max distance to its border) -- the median can land on a neighbour."""
    return np.unravel_index(np.argmax(distance_transform_edt(mask)), mask.shape)

ap = argparse.ArgumentParser()
ap.add_argument('--fine_seg_path', required=True); ap.add_argument('--img_path', required=True)
ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--min_frac', type=float, default=0.01); ap.add_argument('--radius', type=float, default=40,
                help='pixels drawn for mu = 1')
ap.add_argument('--pad', type=int, default=40)
ap.add_argument('--ncols', type=int, default=99); ap.add_argument('--fs', type=int, default=10)
a = ap.parse_args()
fs = torch.load(a.fine_seg_path, map_location='cpu').numpy(); img = plt.imread(a.img_path)
tot = (fs > 0).sum()
segs = [int(s) for s in np.unique(fs) if s > 0 and (fs == s).sum() / tot >= a.min_frac]
rr, cc = np.nonzero(fs > 0)
r0, r1 = max(rr.min() - a.pad, 0), min(rr.max() + a.pad, fs.shape[0]); c0, c1 = max(cc.min() - a.pad, 0), min(cc.max() + a.pad, fs.shape[1])
cmap = plt.cm.tab20
tint = np.zeros(fs.shape + (4,))
for i, s in enumerate(segs): tint[fs == s] = (*cmap(i % 20)[:3], .35)
BOX = dict(boxstyle='round,pad=0.25', fc='white', alpha=.9, ec='0.5', lw=.5)
# callout slots: a column in the empty right part of the crop, one per segment, ordered by the segment's row
order = sorted(segs, key=lambda s: rep_pixel(fs == s)[0])
slot = {s: (c1 - 0.14 * (c1 - c0), r0 + (i + .7) * (r1 - r0) * 0.75 / len(order)) for i, s in enumerate(order)}

GPOS = (r0 + 3.6 * a.radius, c0 + 0.45 * (c1 - c0))
nc = min(len(a.runs), a.ncols); nr = -(-len(a.runs) // nc)
fig, axs = plt.subplots(nr, nc, figsize=(8 * nc, nr * (8 * (r1 - r0) / (c1 - c0) + 0.9)), squeeze=False)
axs = axs.ravel()
for ax in axs[len(a.runs):]: ax.axis('off')
for ax, spec in zip(axs, a.runs):
    label, d = spec.split('=', 1)
    ax.imshow(img); ax.imshow(tint); ax.set_xlim(c0, c1); ax.set_ylim(r1, r0); ax.set_xticks([]); ax.set_yticks([])
    if os.path.exists(os.path.join(d, 'mu_aniso.pth')):
        A = torch.load(os.path.join(d, 'mu_aniso.pth'), map_location='cpu')
        items = [(s, rep_pixel(fs == s), s) for s in segs]
        if A['major'].numel() == 1:                                    # global_aniso: ONE ellipse for all pairs (2026-09-28)
            items = [('global', GPOS, 0)]
        elif A['scope'] in ('segment_aniso_xglobal', 'global_then_segment_aniso'):
            items.append(('xglobal', GPOS, -1))   # empty top area, away from the parts (2026-09-28)
        for name, (cy, cx), g in items:
            M, m, th = float(A['major'][g]), float(A['minor'][g]), float(A['major_angle_deg'][g])
            # image direction th: 0 = along rows (down), 90 = along columns (right) -> (dx, dy) = (sin, cos)
            ux, uy = math.sin(math.radians(th)), math.cos(math.radians(th))
            col = 'magenta' if name in ('xglobal', 'global') else 'k'
            ax.add_patch(Ellipse((cx, cy), 2 * a.radius * M, 2 * a.radius * m, angle=math.degrees(math.atan2(uy, ux)),
                                 fill=False, ec=col, lw=1.5))
            ax.add_patch(Ellipse((cx, cy), 2 * a.radius, 2 * a.radius, fill=False, ec='white', lw=.8, ls='--'))
            for L, (vx, vy), c in ((M, (ux, uy), 'red'), (m, (-uy, ux), 'blue')):
                ax.plot([cx - a.radius * L * vx, cx + a.radius * L * vx], [cy - a.radius * L * vy, cy + a.radius * L * vy], c=c, lw=2)
            if name == 'global':
                tx, ty, txt = cx, cy - a.radius * 3.2, 'ONE global ellipse (all pairs)'
            elif name == 'xglobal':
                tx, ty, txt = cx, cy - a.radius * 3.2, 'cross-seg pairs'
            else:
                (tx, ty), txt = slot[name], 'seg %d' % name
                ax.plot([cx, tx], [cy, ty], c='0.2', lw=1.0, ls=':')
            dy = 0.028 * (r1 - r0)                                                   # one text line, in image rows
            ax.text(tx, ty - dy, txt, fontsize=a.fs, ha='center', va='center', bbox=BOX)
            ax.text(tx, ty, 'along %.2f' % M, color='red', fontsize=a.fs, fontweight='bold', ha='center', va='center', bbox=BOX)
            ax.text(tx, ty + dy, 'across %.2f' % m, color='blue', fontsize=a.fs, fontweight='bold', ha='center', va='center', bbox=BOX)
        ax.set_title('%s\nred = mu ALONG the major axis, blue = mu ACROSS it (white dashed = mu 1)' % label, fontsize=a.fs)
    else:
        mu = torch.load(os.path.join(d, 'mu_scale.pth'), map_location='cpu')['mu_scale']
        for s in segs:
            cy, cx = rep_pixel(fs == s)
            tx, ty = slot[s]
            ax.plot(cx, cy, 'ko', ms=4); ax.plot([cx, tx], [cy, ty], c='0.2', lw=1.0, ls=':')
            ax.text(tx, ty, 'seg %d\nmu %.2f' % (s, mu[s]), fontsize=a.fs, fontweight='bold', ha='center', va='center', bbox=BOX)
        ax.set_title('%s\none mu per segment, same in every direction' % label, fontsize=a.fs)
fig.tight_layout(); fig.savefig(a.out, dpi=120); print('saved', a.out)
