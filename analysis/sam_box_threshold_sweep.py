"""SAM box mask (box = the whole crop, single output) at several mask thresholds, vs the LVIS polygon (Tal, 2026-10-08:
"SAM takes too much -- make it take fewer pixels"). SAM cuts its per-pixel logit at 0; a higher cut keeps only confident pixels.
One row per instance: crop + GT | polygon | one panel per threshold (title: IoU, precision = share of the SAM mask inside the
polygon, recall = share of the polygon covered). No training.
Usage: python analysis/sam_box_threshold_sweep.py --run_dirs <dir with lvis<id>.png, _mask.pth, _gt.pth> ... --save_dir X [--thresholds 0 2 5 8]"""
import argparse, json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SAM_repo.segment_anything import sam_model_registry, SamPredictor

ap = argparse.ArgumentParser()
ap.add_argument('--run_dirs', nargs='+', required=True); ap.add_argument('--save_dir', required=True)
ap.add_argument('--thresholds', type=float, nargs='+', default=[0, 2, 5, 8])
ap.add_argument('--sam_checkpoint', default='SAM_repo/model_checkpoints/sam_vit_h_4b8939.pth')
a = ap.parse_args(); os.makedirs(a.save_dir, exist_ok=True)
pred = SamPredictor(sam_model_registry['vit_h'](checkpoint=a.sam_checkpoint).to('cuda'))


def overlay(img, m, col):
    out = img.astype(np.float32) * 0.35; out[m] = 0.45 * img[m] + 0.55 * np.array(col); return out.astype(np.uint8)


n, k = len(a.run_dirs), len(a.thresholds); stats = {}
fig, ax = plt.subplots(n, 2 + k, figsize=(3.6 * (2 + k), 3.9 * n), squeeze=False)
for r, D in enumerate(a.run_dirs):
    T = os.path.basename(D.rstrip('/'))
    img = np.asarray(Image.open(f'{D}/{T}.png').convert('RGB')); H, W = img.shape[:2]
    poly = torch.load(f'{D}/{T}_mask.pth').numpy().astype(bool); g = torch.load(f'{D}/{T}_gt.pth')
    rows, cols = np.asarray(g['rows']), np.asarray(g['cols'])
    pred.set_image(img)
    logit = pred.predict(box=np.array([0, 0, W - 1, H - 1], np.float32), multimask_output=False, return_logits=True)[0][0]
    panels = [(img, f'{T}'), (overlay(img, poly, (255, 220, 0)), 'LVIS polygon')]; stats[T] = {}
    for t in a.thresholds:
        m = logit > t; inter = float((m & poly).sum())
        s = {'iou': inter / max(float((m | poly).sum()), 1), 'precision': inter / max(float(m.sum()), 1), 'recall': inter / poly.sum(),
             'gt_in': int(m[rows, cols].sum()), 'n_gt': int(len(rows))}
        stats[T][str(t)] = s
        panels.append((overlay(img, m, (0, 220, 255)), f'thr {t:g}: IoU {s["iou"]:.2f} | prec {s["precision"]:.2f} | rec {s["recall"]:.2f} | GT {s["gt_in"]}/{s["n_gt"]}'))
    for c, (im, title) in enumerate(panels):
        A = ax[r, c]; A.imshow(im); A.set_title(title, fontsize=9); A.axis('off')
        A.scatter(cols, rows, s=45, c='yellow', edgecolors='k', linewidths=1.0, zorder=5)
    print(T, ' '.join(f'thr{t:g}: {stats[T][str(t)]["iou"]:.2f}/{stats[T][str(t)]["precision"]:.2f}/{stats[T][str(t)]["recall"]:.2f}/{stats[T][str(t)]["gt_in"]}' for t in a.thresholds))
fig.tight_layout(); out = f'{a.save_dir}/sam_box_threshold_sweep.png'; fig.savefig(out, dpi=60)
json.dump(stats, open(f'{a.save_dir}/sam_box_threshold_sweep.json', 'w'), indent=1); print('saved', out)
