"""Figure for the pix2gestalt occlusion test (2026-10-07). One row per instance:
crop | visible LVIS mask (green = the part our training sees today, red = dropped) | geometric gap fill |
pix2gestalt's synthesized whole object (sample 1) | its amodal mask for each sample (green = visible, blue = added, orange = visible but NOT covered).
Run in the bsb env: python /workspace/pix2gestalt_repo/make_test_figure.py --out_dir /workspace/pix2gestalt_repo/lvis_test/out_steps50
"""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import ndimage
sys.path.insert(0, '/workspace/bsb/analysis')
from gapfill_preview import gap_fill
S8 = np.ones((3, 3))

ap = argparse.ArgumentParser(); ap.add_argument('--out_dir', required=True); ap.add_argument('--inputs', default='/workspace/pix2gestalt_repo/lvis_test/inputs')
a = ap.parse_args()
names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{a.out_dir}/*.npz'))
timing = json.load(open(f'{a.out_dir}/timing.json')) if os.path.exists(f'{a.out_dir}/timing.json') else {}

def connected(vis, full):   # share of the visible pixels that sit in the largest connected piece of `full`
    lab, _ = ndimage.label(full, structure=S8); c = np.bincount(lab[vis & full]); c[0] = 0 if len(c) else 0
    return c.max() / vis.sum() if len(c) else 0.0

rows = []
for half, chunk in enumerate([names[:5], names[5:]]):
    if not chunk: continue
    n_s = np.load(f'{a.out_dir}/{chunk[0]}.npz')['amodal'].shape[0]
    fig, axes = plt.subplots(len(chunk), 4 + n_s, figsize=(3.1 * (4 + n_s), 3.3 * len(chunk)), squeeze=False)
    for r, name in enumerate(chunk):
        d = np.load(f'{a.inputs}/{name}.npz'); o = np.load(f'{a.out_dir}/{name}.npz'); img = d['image']; vis = d['mask'] > 0
        lab, n = ndimage.label(vis, structure=S8); big = lab == np.bincount(lab[vis]).argmax()
        A = np.full(img.shape, 255, np.uint8); A[vis] = (235, 150, 150); A[big] = (160, 215, 160)
        gf, rad, gfrac = gap_fill(vis); B = np.full(img.shape, 255, np.uint8); B[vis] = (160, 215, 160); B[gf & ~vis] = (60, 90, 230)
        ax = axes[r]
        ax[0].imshow(img); ax[0].set_title(name.replace('_', ' '), fontsize=11)
        ax[1].imshow(A); ax[1].set_title(f'visible mask: {n} parts, training sees {100 * big.sum() / vis.sum():.0f}%', fontsize=9)
        ax[2].imshow(B); ax[2].set_title(f'geometric fill: {100 * gfrac:.0f}% connected', fontsize=9)
        ax[3].imshow(o['preds'][0]); ax[3].set_title('pix2gestalt: whole object (sample 1)', fontsize=9)
        res = dict(name=name, parts=int(n), seen_today=float(big.sum() / vis.sum()), geometric=float(gfrac), samples=[])
        for k in range(n_s):
            am = o['amodal'][k]; C = np.full(img.shape, 255, np.uint8); C[am] = (60, 90, 230); C[vis & am] = (160, 215, 160); C[vis & ~am] = (240, 150, 40)
            cov = (vis & am).sum() / vis.sum(); con = connected(vis, am); res['samples'].append(dict(covers_visible=float(cov), connected=float(con), area_ratio=float(am.sum() / vis.sum())))
            ax[4 + k].imshow(C); ax[4 + k].set_title(f'amodal mask {k + 1}: {100 * con:.0f}% connected,\ncovers {100 * cov:.0f}% of visible, area x{am.sum() / vis.sum():.2f}', fontsize=9)
        rows.append(res)
        for x in ax: x.axis('off')
    fig.tight_layout(); p = f'{a.out_dir}/pix2gestalt_test_{half + 1}.png'; fig.savefig(p, dpi=80); plt.close(fig); print('saved', p)
json.dump(rows, open(f'{a.out_dir}/summary.json', 'w'), indent=1)
for q in rows:
    print(f"{q['name']:14s} parts {q['parts']:2d} today {100 * q['seen_today']:3.0f}% | geometric {100 * q['geometric']:3.0f}% | p2g connected "
          + ' '.join(f"{100 * s['connected']:3.0f}%" for s in q['samples']) + ' | covers visible ' + ' '.join(f"{100 * s['covers_visible']:3.0f}%" for s in q['samples'])
          + f" | {timing.get('per_instance_s', {}).get(q['name'], float('nan')):.0f} s")
