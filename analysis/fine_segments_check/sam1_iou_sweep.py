"""SAM 1 fine segments on horse lvis56069 at several pred_iou_thresh values (Tal, 2026-10-21).
Same calls as analysis/build_fine_segments.py (the worker's settings: min_pixels_in_object 50), and for each threshold
both the RAW map (no merge, no fill) and the training recipe (merge 1.5% + fillseg). Writes one PNG per map + a grid."""
import os, sys, time
REPO = os.environ.get('DISC_ROOT', os.path.expanduser('~/Projects/disc')); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'analysis'))
import cv2, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from build_fine_segments import build_label_map, merge_small_segments, fill_unlabelled, load_coarse_mask
from modules.sam import load_img_seg_model
from SAM_repo.segment_anything import SamAutomaticMaskGenerator

SRC = os.path.join(REPO, 'data/densepose_lvis/runs_debug_merge0.015_amodal_fillseg_sam3mask/lvis56069')
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
THRS = [0.90, 0.88, 0.85, 0.80]
image = cv2.cvtColor(cv2.imread(f'{SRC}/lvis56069.png'), cv2.COLOR_BGR2RGB)
mask = load_coarse_mask(f'{SRC}/lvis56069_mask.pth'); n_obj = int(mask.sum())
sam = load_img_seg_model(os.path.join(REPO, 'SAM_repo/model_checkpoints'), 'sam_vit_h_4b8939.pth', 'vit_h', device='cpu')

def overlay(labels):
    rng = np.random.RandomState(0); pal = rng.rand(max(int(labels.max()), 1) + 1, 3); pal[0] = 0
    b = image / 255.0 * 0.35 + pal[labels] * 0.65; b[labels == 0] = image[labels == 0] / 255.0 * 0.3
    return np.clip(b, 0, 1)

def summary(labels):
    ids, cnt = np.unique(labels[labels > 0], return_counts=True)
    order = np.argsort(-cnt)
    return len(ids), [round(100.0 * cnt[k] / n_obj, 1) for k in order], round(100.0 * (labels > 0).sum() / n_obj, 1)

panels = []
for thr in THRS:
    t0 = time.time()
    amg = SamAutomaticMaskGenerator(model=sam, points_per_side=32, pred_iou_thresh=thr, stability_score_thresh=0.92,
                                    crop_n_points_downscale_factor=2, min_mask_region_area=150)
    anns = amg.generate(image)
    raw, _ = build_label_map(anns, mask, 50)
    rec, _ = merge_small_segments(raw, 0.015 * n_obj)
    rec, _ = fill_unlabelled(rec, mask)
    np.save(f'{OUT}/labels_thr{thr:.2f}_raw.npy', raw); np.save(f'{OUT}/labels_thr{thr:.2f}_recipe.npy', rec)
    for name, lab in (('raw', raw), ('recipe', rec)):
        n, sizes, cov = summary(lab)
        print(f'thr {thr:.2f} {name:6s}: {len(anns):3d} SAM masks -> {n:2d} segments, coverage {cov}%, sizes % (largest first) {sizes[:8]}'
              f'{" ..." if n > 8 else ""}  [{time.time() - t0:.0f}s]', flush=True)
        panels.append((f'pred_iou_thresh {thr:.2f} -- {name}\n{n} segments, largest {sizes[0] if sizes else 0}%', overlay(lab)))

fig, axes = plt.subplots(2, len(THRS), figsize=(4.2 * len(THRS), 2 * 6.0))
for j in range(len(THRS)):
    for i in range(2):
        t, im = panels[2 * j + i]; ax = axes[i, j]; ax.imshow(im); ax.set_title(t, fontsize=10); ax.axis('off')
fig.suptitle('horse lvis56069, SAM 1 fine segments: top = raw SAM map, bottom = training recipe (merge 1.5% + fillseg)', fontsize=12)
fig.tight_layout(); fig.savefig(f'{OUT}/sam1_iou_sweep_lvis56069.png', dpi=110); print('saved', f'{OUT}/sam1_iou_sweep_lvis56069.png')
