"""Amodal mask: how many of pix2gestalt's samples must agree on a pixel (Tal, 2026-10-08: holes left in the middle of the object).
Same merge as analysis/merge_amodal_mask.py, but for several --min_votes side by side; nothing in the run dir is overwritten.
One row per instance: crop | one panel per vote rule (green = visible mask, blue = completed, white inside = still open) | what the
lowest rule adds over the highest (orange).
Usage: python analysis/amodal_votes_preview.py --run_dirs <dirs with _mask.pth, _amodal_input.npz, _amodal_samples.npz> --save_dir X [--votes 2 1]"""
import argparse, json, os
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument('--run_dirs', nargs='+', required=True); ap.add_argument('--save_dir', required=True)
ap.add_argument('--votes', type=int, nargs='+', default=[2, 1])
a = ap.parse_args(); os.makedirs(a.save_dir, exist_ok=True); S8 = np.ones((3, 3))
n, k = len(a.run_dirs), len(a.votes); stats = {}
fig, ax = plt.subplots(n, k + 2, figsize=(4.4 * (k + 2), 4.2 * n), squeeze=False)
for r, D in enumerate(a.run_dirs):
    T = os.path.basename(D.rstrip('/'))
    img = np.asarray(Image.open(f'{D}/{T}.png').convert('RGB'))
    vis = torch.load(f'{D}/{T}_mask.pth').numpy().astype(bool)
    g = np.load(f'{D}/{T}_amodal_input.npz'); smp = np.load(f'{D}/{T}_amodal_samples.npz')['amodal']
    sq, box, size = g['square'], g['box'], g['size']; side = int(sq[2] - sq[0]); votes = np.zeros(vis.shape, np.int32)
    for am in smp:   # 256 -> square crop px -> bbox -> run crop, exactly as merge_amodal_mask.py
        full = np.asarray(Image.fromarray((am * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR)) > 127
        sub = full[box[1] - sq[1]:box[3] - sq[1], box[0] - sq[0]:box[2] - sq[0]]
        votes += np.asarray(Image.fromarray((sub * 255).astype(np.uint8)).resize((int(size[0]), int(size[1])), Image.BILINEAR)) > 127
    ax[r, 0].imshow(img); ax[r, 0].set_title(T, fontsize=10); doms = {}; stats[T] = {}
    for c, v in enumerate(a.votes):
        dom = (votes >= v) | vis
        lab, _ = ndimage.label(dom, structure=S8); cnt = np.bincount(lab[vis]); cnt[0] = 0; main = lab == cnt.argmax()
        holes = ndimage.binary_fill_holes(main) & ~main; doms[v] = main
        s = {'added_pct_of_visible': 100.0 * float((main & ~vis).sum()) / vis.sum(), 'holes_px': int(holes.sum()),
             'visible_connected_pct': 100.0 * float((vis & main).sum()) / vis.sum()}
        stats[T][str(v)] = s
        im = np.full(img.shape, 255, np.uint8); im[main] = (60, 90, 230); im[vis & main] = (160, 215, 160); im[vis & ~main] = (235, 150, 150)
        ax[r, c + 1].imshow(im)
        ax[r, c + 1].set_title(f'>= {v} of {len(smp)} samples: +{s["added_pct_of_visible"]:.1f}% | holes {s["holes_px"]} px | connected {s["visible_connected_pct"]:.1f}%', fontsize=9)
    lo, hi = doms[min(a.votes)], doms[max(a.votes)]
    d = (img.astype(np.float32) * 0.45).astype(np.uint8); d[hi] = (0.5 * img[hi] + 0.5 * np.array((90, 130, 255))).astype(np.uint8); d[lo & ~hi] = (255, 140, 0)
    ax[r, k + 1].imshow(d); ax[r, k + 1].set_title(f'orange = added by ">= {min(a.votes)}" over ">= {max(a.votes)}" ({int((lo & ~hi).sum())} px)', fontsize=9)
    for A in ax[r]: A.axis('off')
    print(T, json.dumps(stats[T]))
fig.tight_layout(); out = f'{a.save_dir}/amodal_votes_preview.png'; fig.savefig(out, dpi=70)
json.dump(stats, open(f'{a.save_dir}/amodal_votes_preview.json', 'w'), indent=1); print('saved', out)
