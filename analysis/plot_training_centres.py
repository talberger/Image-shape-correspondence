"""Where the training's centre pixels are, vs the object mask (Tal, 2026-10-07: occluded masks split into parts and the 2D geodesic
table keeps only the largest one). Left: crop + centres + GT points. Right: mask parts -- green = in the geodesic table (seen by
the training), red = dropped -- with the same centres.
Usage: python analysis/plot_training_centres.py --run_dir data/densepose_lvis/runs_debug_merge0.015/lvis206034 --tag lvis206034 --output_path X.png"""
import argparse, os
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument('--run_dir', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--output_path', required=True)
a = ap.parse_args(); D, T = a.run_dir, a.tag
img = np.asarray(Image.open(f'{D}/{T}.png').convert('RGB'))
mask = torch.load(f'{D}/{T}_mask.pth').numpy()
t = torch.load(f'{D}/{T}_mask_geo_table_2d_seed0.pth')
safe = t['geodesic_safe_mask'].numpy(); c = t['coords'].numpy(); g = torch.load(f'{D}/{T}_gt.pth')
n_parts = ndimage.label(mask, structure=np.ones((3, 3)))[1]
inside = safe[c[:, 0], c[:, 1]].sum()
dom = t.get('geodesic_domain')   # occlusion domain (amodal mask), None for an ordinary table
vis = np.full(img.shape, 255, np.uint8)
if dom is not None: vis[dom.numpy()] = (60, 90, 230)
vis[mask & ~safe] = (235, 150, 150); vis[safe] = (160, 215, 160)
fig, ax = plt.subplots(1, 2, figsize=(15, 8.5))
ax[0].imshow(img); ax[1].imshow(vis)
for A in ax:
    A.scatter(c[:, 1], c[:, 0], s=3, c='blue', linewidths=0)
    for i, (r, k) in enumerate(zip(g['rows'].tolist(), g['cols'].tolist())):
        A.scatter(k, r, s=90, c='yellow', edgecolors='k', linewidths=1.5, zorder=5)
        A.annotate(str(i + 1), (k, r), xytext=(7, -7), textcoords='offset points', color='k', fontsize=13, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.1', fc='yellow', ec='none'), zorder=6)
    A.axis('off')
nolab = None
fsp = f'{D}/{T}_fine_segments.pth'
if os.path.exists(fsp): nolab = 100 * ((torch.load(fsp).numpy() <= 0) & safe).sum() / max(safe.sum(), 1)
ax[0].set_title(f'{T}: the {len(c)} training centres (blue dots), GT points (yellow)', fontsize=11)
ax[1].set_title(f'{n_parts} mask parts. green = visible, usable by the training ({100 * safe.sum() / mask.sum():.1f}% of the mask); red = dropped'
                + ('' if dom is None else '; blue = hidden part (distances only)')
                + f'\n{inside}/{len(c)} centres on green' + ('' if nolab is None else f'; {nolab:.1f}% of the green has no fine segment -> no centres there'), fontsize=9)
os.makedirs(os.path.dirname(a.output_path), exist_ok=True)
fig.tight_layout(); fig.savefig(a.output_path, dpi=90); print('saved', a.output_path, 'centres in safe', inside, '/', len(c))
