"""SAM mask from a BOX prompt = the whole crop, vs the LVIS polygon mask (Tal, 2026-10-08: "only the image goes in, from there
the algorithm is on its own" -- can the polygon be dropped?). No training; reads the crop + polygon mask a debug run already wrote.
Per instance (one row): crop + GT | polygon | SAM box, single output | SAM box, best of the 3 multimask outputs | difference.
Saves <save_dir>/lvis<id>_sambox_mask.pth (single output) and one figure.
Usage: python analysis/sam_box_mask_preview.py --run_dirs data/densepose_lvis/runs_debug/lvis56069 ... --save_dir demo/shic/sam_box_mask"""
import argparse, json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SAM_repo.segment_anything import sam_model_registry, SamPredictor

ap = argparse.ArgumentParser()
ap.add_argument('--run_dirs', nargs='+', required=True); ap.add_argument('--save_dir', required=True)
ap.add_argument('--sam_checkpoint', default='SAM_repo/model_checkpoints/sam_vit_h_4b8939.pth')
a = ap.parse_args(); os.makedirs(a.save_dir, exist_ok=True)
pred = SamPredictor(sam_model_registry['vit_h'](checkpoint=a.sam_checkpoint).to('cuda'))


def iou(x, y): return float((x & y).sum()) / max(float((x | y).sum()), 1.0)


def overlay(img, m, col):
    out = img.astype(np.float32) * 0.35; out[m] = 0.45 * img[m] + 0.55 * np.array(col); return out.astype(np.uint8)


n = len(a.run_dirs); fig, ax = plt.subplots(n, 5, figsize=(22, 4.6 * n), squeeze=False); stats = {}
for r, D in enumerate(a.run_dirs):
    T = os.path.basename(D.rstrip('/'))
    img = np.asarray(Image.open(f'{D}/{T}.png').convert('RGB')); H, W = img.shape[:2]
    poly = torch.load(f'{D}/{T}_mask.pth').numpy().astype(bool); g = torch.load(f'{D}/{T}_gt.pth')
    rows, cols = np.asarray(g['rows']), np.asarray(g['cols'])
    pred.set_image(img); box = np.array([0, 0, W - 1, H - 1], np.float32)
    m1, s1, _ = pred.predict(box=box, multimask_output=False)
    m3, s3, _ = pred.predict(box=box, multimask_output=True)
    single, best = m1[0].astype(bool), m3[int(np.argmax(s3))].astype(bool)
    torch.save(torch.from_numpy(single), f'{a.save_dir}/{T}_sambox_mask.pth')
    st = {'polygon_px': int(poly.sum()), 'sambox_px': int(single.sum()), 'iou_single': iou(poly, single), 'iou_best3': iou(poly, best),
          'score_single': float(s1[0]), 'scores_3': [float(x) for x in s3],
          'polygon_only_pct_of_polygon': 100.0 * float((poly & ~single).sum()) / poly.sum(),
          'sam_only_pct_of_polygon': 100.0 * float((single & ~poly).sum()) / poly.sum(),
          'gt_in_polygon': int(poly[rows, cols].sum()), 'gt_in_sambox': int(single[rows, cols].sum()), 'n_gt': int(len(rows))}
    stats[T] = st
    diff = np.full(img.shape, 255, np.uint8); diff[poly & single] = (200, 200, 200); diff[poly & ~single] = (220, 60, 60); diff[single & ~poly] = (50, 90, 230)
    panels = [(img, f'{T}: crop + GT points'), (overlay(img, poly, (255, 220, 0)), f'LVIS polygon ({st["polygon_px"]} px)'),
              (overlay(img, single, (0, 220, 255)), f'SAM box, single output (IoU {st["iou_single"]:.2f}, score {s1[0]:.2f})'),
              (overlay(img, best, (0, 220, 255)), f'SAM box, best of 3 (IoU {st["iou_best3"]:.2f}, score {max(s3):.2f})'),
              (diff, f'grey = both | red = polygon only ({st["polygon_only_pct_of_polygon"]:.1f}%) | blue = SAM only ({st["sam_only_pct_of_polygon"]:.1f}%)\n'
                     f'GT points inside: polygon {st["gt_in_polygon"]}/{st["n_gt"]}, SAM box {st["gt_in_sambox"]}/{st["n_gt"]}')]
    for c, (im, title) in enumerate(panels):
        A = ax[r, c]; A.imshow(im); A.set_title(title, fontsize=10); A.axis('off')
        A.scatter(cols, rows, s=60, c='yellow', edgecolors='k', linewidths=1.2, zorder=5)
    print(T, json.dumps(st))
fig.tight_layout(); out = f'{a.save_dir}/sam_box_vs_polygon.png'; fig.savefig(out, dpi=80)
json.dump(stats, open(f'{a.save_dir}/sam_box_vs_polygon.json', 'w'), indent=1); print('saved', out)
