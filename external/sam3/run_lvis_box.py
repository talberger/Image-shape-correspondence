"""SAM 3 mask from a BOX prompt = the whole crop (its SAM-1-style interactive task, NO text prompt) vs the LVIS polygon and vs SAM 1
(Tal, 2026-10-08: masks only, no training). Runs in /workspace/sam3_repo/env, not the bsb env.
One row per instance: crop + GT | polygon | SAM 1 box | SAM 3 box | SAM 3 vs polygon difference.
Usage: HF_HOME=/workspace/.cache/huggingface env/bin/python run_lvis_box.py --run_dirs <dir with lvis<id>.png,_mask.pth,_gt.pth> ...
       --sam1_dirs <dir with lvis<id>_sambox_mask.pth> ... --save_dir X"""
import argparse, json, os
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

ap = argparse.ArgumentParser()
ap.add_argument('--run_dirs', nargs='+', required=True); ap.add_argument('--sam1_dirs', nargs='+', default=[]); ap.add_argument('--save_dir', required=True)
a = ap.parse_args(); os.makedirs(a.save_dir, exist_ok=True)
model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)


def iou(x, y): return float((x & y).sum()) / max(float((x | y).sum()), 1.0)


def overlay(img, m, col):
    out = img.astype(np.float32) * 0.35; out[m] = 0.45 * img[m] + 0.55 * np.array(col); return out.astype(np.uint8)


n = len(a.run_dirs); fig, ax = plt.subplots(n, 5, figsize=(20, 4.2 * n), squeeze=False); stats = {}
for r, D in enumerate(a.run_dirs):
    T = os.path.basename(D.rstrip('/'))
    pil = Image.open(f'{D}/{T}.png').convert('RGB'); img = np.asarray(pil); H, W = img.shape[:2]
    poly = torch.load(f'{D}/{T}_mask.pth', weights_only=False).numpy().astype(bool); g = torch.load(f'{D}/{T}_gt.pth', weights_only=False)
    rows, cols = np.asarray(g['rows']), np.asarray(g['cols'])
    with torch.autocast('cuda', dtype=torch.bfloat16), torch.inference_mode():
        state = proc.set_image(pil)
        masks, scores, _ = model.predict_inst(state, point_coords=None, point_labels=None, box=np.array([0, 0, W - 1, H - 1], np.float32)[None], multimask_output=False)
    m3 = np.asarray(masks).reshape(-1, H, W)[0] > 0
    torch.save(torch.from_numpy(m3), f'{a.save_dir}/{T}_sam3box_mask.pth')
    m1 = None
    for s in a.sam1_dirs:
        if os.path.exists(f'{s}/{T}_sambox_mask.pth'): m1 = torch.load(f'{s}/{T}_sambox_mask.pth', weights_only=False).numpy().astype(bool)
    inter = float((m3 & poly).sum())
    st = {'iou_sam3': iou(poly, m3), 'iou_sam1': None if m1 is None else iou(poly, m1), 'precision_sam3': inter / max(float(m3.sum()), 1), 'recall_sam3': inter / poly.sum(),
          'score_sam3': float(np.asarray(scores).reshape(-1)[0]), 'gt_in_sam3': int(m3[rows, cols].sum()), 'n_gt': int(len(rows))}
    stats[T] = st
    diff = np.full(img.shape, 255, np.uint8); diff[poly & m3] = (200, 200, 200); diff[poly & ~m3] = (220, 60, 60); diff[m3 & ~poly] = (50, 90, 230)
    panels = [(img, T), (overlay(img, poly, (255, 220, 0)), 'LVIS polygon'),
              (img if m1 is None else overlay(img, m1, (0, 220, 255)), 'SAM 1 box' + ('' if m1 is None else f' (IoU {st["iou_sam1"]:.2f})')),
              (overlay(img, m3, (255, 90, 220)), f'SAM 3 box (IoU {st["iou_sam3"]:.2f}, score {st["score_sam3"]:.2f}, GT {st["gt_in_sam3"]}/{st["n_gt"]})'),
              (diff, 'SAM 3 vs polygon: grey = both | red = polygon only | blue = SAM 3 only')]
    for c, (im, title) in enumerate(panels):
        A = ax[r, c]; A.imshow(im); A.set_title(title, fontsize=10); A.axis('off')
        A.scatter(cols, rows, s=50, c='yellow', edgecolors='k', linewidths=1.0, zorder=5)
    print(T, json.dumps(st), flush=True)
fig.tight_layout(); out = f'{a.save_dir}/sam3_box_vs_polygon.png'; fig.savefig(out, dpi=60)
json.dump(stats, open(f'{a.save_dir}/sam3_box_vs_polygon.json', 'w'), indent=1); print('saved', out)
