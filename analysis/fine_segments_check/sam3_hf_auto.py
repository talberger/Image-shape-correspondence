"""SAM 3 automatic masks on horse lvis56069 through the Hugging Face `mask-generation` pipeline (Tal, 2026-10-21) -- the
SAM-1-style point grid, run on SAM 3's interactive (tracker) head. Then OUR build_label_map (min_pixels_in_object 50, as the
worker) -> raw map, + merge 1.5% + fillseg -> recipe map, the same as the SAM 1 sweep. Env: sam3hf (Mac)."""
import os, sys, time
REPO = os.environ.get('DISC_ROOT', os.path.expanduser('~/Projects/disc')); sys.path.insert(0, os.path.join(REPO, 'analysis'))
import numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from transformers import pipeline

SRC = os.path.join(REPO, 'data/densepose_lvis/runs_debug_merge0.015_amodal_fillseg_sam3mask/lvis56069')
OUT = sys.argv[1]; DEV = sys.argv[2] if len(sys.argv) > 2 else 'cpu'; os.makedirs(OUT, exist_ok=True)
pil = Image.open(f'{SRC}/lvis56069.png').convert('RGB'); image = np.asarray(pil)
mask = torch.load(f'{SRC}/lvis56069_mask.pth', map_location='cpu').numpy(); n_obj = int(mask.sum())

# build_fine_segments.py's own post-processing, copied by import would pull in SAM 1 + py3.9 modules; it only needs numpy/scipy,
# so load the three functions from the file's source without running its imports of modules.sam
src = open(os.path.join(REPO, 'analysis/build_fine_segments.py')).read()
ns = {'np': np}
for fn in ('build_label_map', 'merge_small_segments', 'fill_unlabelled'):
    start = src.index('def %s(' % fn); end = src.index('\ndef ', start + 1)
    exec(src[start:end], ns)

gen = pipeline('mask-generation', model='facebook/sam3', device=DEV)
SETTINGS = [('nms 0.9, pred_iou 0.5, stab 0.8', {'crops_nms_thresh': 0.9, 'pred_iou_thresh': 0.5, 'stability_score_thresh': 0.8}),
            ('nms 0.9, pred_iou 0.3, stab 0.7', {'crops_nms_thresh': 0.9, 'pred_iou_thresh': 0.3, 'stability_score_thresh': 0.7})]

def overlay(labels):
    rng = np.random.RandomState(0); pal = rng.rand(max(int(labels.max()), 1) + 1, 3); pal[0] = 0
    b = image / 255.0 * 0.35 + pal[labels] * 0.65; b[labels == 0] = image[labels == 0] / 255.0 * 0.3
    return np.clip(b, 0, 1)

panels = []
for name, kw in SETTINGS:
    t0 = time.time()
    out = gen(pil, points_per_batch=64, **kw)
    masks = [np.asarray(m).astype(bool) for m in out['masks']]
    anns = [{'segmentation': m} for m in masks]
    raw, _ = ns['build_label_map'](anns, mask, 50)
    rec, _ = ns['merge_small_segments'](raw, 0.015 * n_obj); rec, _ = ns['fill_unlabelled'](rec, mask)
    for tag, lab in (('raw', raw), ('recipe', rec)):
        ids, cnt = np.unique(lab[lab > 0], return_counts=True); sizes = sorted((100.0 * cnt / n_obj).round(1).tolist(), reverse=True)
        cov = round(100.0 * (lab > 0).sum() / n_obj, 1)
        print(f'{name:45s} {tag:6s}: {len(masks):3d} SAM 3 masks -> {len(ids):2d} segments, coverage {cov}%, largest {sizes[:6]}  [{time.time() - t0:.0f}s]', flush=True)
        np.save(f'{OUT}/sam3_labels_set{SETTINGS.index((name, kw))}_{tag}.npy', lab)
        panels.append((f'SAM 3, {name}\n{tag}: {len(ids)} segments, largest {sizes[0] if sizes else 0}%', overlay(lab)))

fig, axes = plt.subplots(2, len(SETTINGS), figsize=(4.6 * len(SETTINGS), 2 * 6.2))
for j in range(len(SETTINGS)):
    for i in range(2):
        t, im = panels[2 * j + i]; ax = axes[i, j]; ax.imshow(im); ax.set_title(t, fontsize=9); ax.axis('off')
fig.suptitle('horse lvis56069, SAM 3 automatic masks (HF pipeline): top = raw, bottom = recipe (merge 1.5% + fillseg)', fontsize=11)
fig.tight_layout(); fig.savefig(f'{OUT}/sam3_amg_params2_lvis56069.png', dpi=110); print('saved', f'{OUT}/sam3_amg_params2_lvis56069.png')
