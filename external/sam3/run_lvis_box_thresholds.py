"""SAM 3 box mask (box = the whole crop, no text) at several mask thresholds vs the LVIS polygon (Tal, 2026-10-08).
SAM cuts its per-pixel logit at 0: a lower cut takes MORE pixels, a higher cut FEWER.
One row per instance: crop + GT | polygon | one panel per threshold (IoU | precision = share of the mask inside the polygon | recall | GT inside).
Usage: HF_HOME=/workspace/.cache/huggingface env/bin/python run_lvis_box_thresholds.py --run_dirs ... --save_dir X [--thresholds -2 -1 0 1 2]"""
import argparse, json, os
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

ap = argparse.ArgumentParser()
ap.add_argument('--run_dirs', nargs='+', required=True); ap.add_argument('--save_dir', required=True)
ap.add_argument('--thresholds', type=float, nargs='+', default=[-2, -1, 0, 1, 2])
a = ap.parse_args(); os.makedirs(a.save_dir, exist_ok=True)
model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)


def overlay(img, m, col):
    out = img.astype(np.float32) * 0.35; out[m] = 0.45 * img[m] + 0.55 * np.array(col); return out.astype(np.uint8)


n, k = len(a.run_dirs), len(a.thresholds); stats = {}
fig, ax = plt.subplots(n, 2 + k, figsize=(3.4 * (2 + k), 3.8 * n), squeeze=False)
for r, D in enumerate(a.run_dirs):
    T = os.path.basename(D.rstrip('/'))
    pil = Image.open(f'{D}/{T}.png').convert('RGB'); img = np.asarray(pil); H, W = img.shape[:2]
    poly = torch.load(f'{D}/{T}_mask.pth', weights_only=False).numpy().astype(bool); g = torch.load(f'{D}/{T}_gt.pth', weights_only=False)
    rows, cols = np.asarray(g['rows']), np.asarray(g['cols'])
    with torch.autocast('cuda', dtype=torch.bfloat16), torch.inference_mode():
        lg, _, _ = model.predict_inst(proc.set_image(pil), point_coords=None, point_labels=None, box=np.array([0, 0, W - 1, H - 1], np.float32)[None],
                                      multimask_output=False, return_logits=True)
    logit = np.asarray(lg, dtype=np.float32).reshape(-1, H, W)[0]
    panels = [(img, T), (overlay(img, poly, (255, 220, 0)), 'LVIS polygon')]; stats[T] = {}
    for t in a.thresholds:
        m = logit > t; inter = float((m & poly).sum())
        s = {'iou': inter / max(float((m | poly).sum()), 1), 'precision': inter / max(float(m.sum()), 1), 'recall': inter / poly.sum(),
             'gt_in': int(m[rows, cols].sum()), 'n_gt': int(len(rows)), 'px': int(m.sum())}
        stats[T][f'{t:g}'] = s
        panels.append((overlay(img, m, (255, 90, 220)), f'thr {t:g}: IoU {s["iou"]:.2f} | prec {s["precision"]:.2f} | rec {s["recall"]:.2f} | GT {s["gt_in"]}/{s["n_gt"]}'))
    for c, (im, title) in enumerate(panels):
        A = ax[r, c]; A.imshow(im); A.set_title(title, fontsize=9); A.axis('off')
        A.scatter(cols, rows, s=40, c='yellow', edgecolors='k', linewidths=1.0, zorder=5)
    print(T, ' '.join(f'{t:g}:{stats[T][f"{t:g}"]["iou"]:.2f}/{stats[T][f"{t:g}"]["precision"]:.2f}/{stats[T][f"{t:g}"]["recall"]:.2f}/{stats[T][f"{t:g}"]["gt_in"]}' for t in a.thresholds), flush=True)
fig.tight_layout(); out = f'{a.save_dir}/sam3_box_threshold_sweep.png'; fig.savefig(out, dpi=55)
json.dump(stats, open(f'{a.save_dir}/sam3_box_threshold_sweep.json', 'w'), indent=1); print('saved', out)
