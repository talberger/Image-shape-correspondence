"""Two checks on horse lvis56069 (Tal, 2026-10-21): (1) can SAM 1 produce a HEAD-ONLY mask at all? One point prompt per
location, multimask output (3 masks each) with predicted IoU + stability; (2) the automatic generator's filters that could
drop such a mask: box_nms_thresh x stability_score_thresh, at pred_iou_thresh 0.90 (current)."""
import os, sys, time
REPO = os.environ.get('DISC_ROOT', os.path.expanduser('~/Projects/disc')); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'analysis'))
import cv2, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from build_fine_segments import build_label_map, load_coarse_mask
from modules.sam import load_img_seg_model
from SAM_repo.segment_anything import SamAutomaticMaskGenerator, SamPredictor
from SAM_repo.segment_anything.utils.amg import calculate_stability_score

SRC = os.path.join(REPO, 'data/densepose_lvis/runs_debug_merge0.015_amodal_fillseg_sam3mask/lvis56069')
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
image = cv2.cvtColor(cv2.imread(f'{SRC}/lvis56069.png'), cv2.COLOR_BGR2RGB)
mask = load_coarse_mask(f'{SRC}/lvis56069_mask.pth'); n_obj = int(mask.sum())
sam = load_img_seg_model(os.path.join(REPO, 'SAM_repo/model_checkpoints'), 'sam_vit_h_4b8939.pth', 'vit_h', device='cpu')

# ---- (1) point prompts. (row, col) as Tal reads them off the crop.
POINTS = {'nose (270,150)': (270, 150), 'forehead (70,140)': (70, 140), 'between eyes (190,130)': (190, 130),
          'right body (250,230)': (250, 230), 'left chest (330,40)': (330, 40)}
pred = SamPredictor(sam); pred.set_image(image)
fig, axes = plt.subplots(len(POINTS), 3, figsize=(3 * 3.2, len(POINTS) * 4.4))
for i, (name, (r, c)) in enumerate(POINTS.items()):
    logits, iou, _ = pred.predict(point_coords=np.array([[c, r]]), point_labels=np.array([1]), multimask_output=True, return_logits=True)
    lt = torch.as_tensor(logits)
    stab = calculate_stability_score(lt, sam.mask_threshold, 1.0).numpy()
    for j in range(3):
        m = logits[j] > sam.mask_threshold
        frac = 100.0 * (m & mask).sum() / n_obj
        print(f'{name:24s} mask {j}: {frac:5.1f}% of the horse, predicted IoU {iou[j]:.3f}, stability {stab[j]:.3f}')
        b = image / 255.0 * 0.4; b[m] = image[m] / 255.0 * 0.4 + np.array([1.0, 0.2, 0.6]) * 0.6
        ax = axes[i, j]; ax.imshow(np.clip(b, 0, 1)); ax.plot(c, r, 'c*', ms=14, mec='k')
        ax.set_title(f'{name}\nmask {j}: {frac:.0f}%  IoU {iou[j]:.2f}  stab {stab[j]:.2f}', fontsize=8); ax.axis('off')
fig.tight_layout(); fig.savefig(f'{OUT}/sam1_point_prompts_lvis56069.png', dpi=100); plt.close(fig)

# ---- (2) automatic generator: box NMS x stability, pred_iou 0.90
def overlay(labels):
    rng = np.random.RandomState(0); pal = rng.rand(max(int(labels.max()), 1) + 1, 3); pal[0] = 0
    b = image / 255.0 * 0.35 + pal[labels] * 0.65; b[labels == 0] = image[labels == 0] / 255.0 * 0.3
    return np.clip(b, 0, 1)
runs = [(nms, st) for st in (0.92, 0.85) for nms in (0.70, 0.85, 0.95)]
fig, axes = plt.subplots(2, 3, figsize=(3 * 4.2, 2 * 6.0))
for k, (nms, st) in enumerate(runs):
    t0 = time.time()
    anns = SamAutomaticMaskGenerator(model=sam, points_per_side=32, pred_iou_thresh=0.90, stability_score_thresh=st,
                                     box_nms_thresh=nms, crop_n_points_downscale_factor=2, min_mask_region_area=150).generate(image)
    raw, _ = build_label_map(anns, mask, 50)
    np.save(f'{OUT}/labels_nms{nms:.2f}_stab{st:.2f}_raw.npy', raw)
    ids, cnt = np.unique(raw[raw > 0], return_counts=True); sizes = sorted((100.0 * cnt / n_obj).round(1).tolist(), reverse=True)
    print(f'box_nms {nms:.2f} stability {st:.2f}: {len(anns):3d} SAM masks -> {len(ids):2d} raw segments, largest {sizes[:5]}  [{time.time() - t0:.0f}s]', flush=True)
    ax = axes[k // 3, k % 3]; ax.imshow(overlay(raw)); ax.axis('off')
    ax.set_title(f'box_nms {nms:.2f}, stability {st:.2f}\n{len(ids)} segments, largest {sizes[0]}%', fontsize=10)
fig.suptitle('horse lvis56069, SAM 1 raw fine segments (no merge), pred_iou_thresh 0.90', fontsize=12)
fig.tight_layout(); fig.savefig(f'{OUT}/sam1_nms_stability_lvis56069.png', dpi=110)
print('saved', OUT)
