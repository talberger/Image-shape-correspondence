"""Draw one DensePose-LVIS instance's raw GT labels (point number + GT vertex) on the ORIGINAL image: full image + bbox zoom.
Built 2026-10-07 to check suspected wrong labels (bear 237843). Usage: python analysis/lvis_label_check.py --ann_id 7107 --class_name bear --output_path X.png"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from eval_densepose_lvis import load_instances, get_image

ap = argparse.ArgumentParser()
ap.add_argument('--ann_id', type=int, required=True); ap.add_argument('--class_name', default='bear')
ap.add_argument('--output_path', required=True)
args = ap.parse_args()
anns, imgs = load_instances(args.class_name)
a = next(a for a in anns if a['id'] == args.ann_id)
img = get_image(imgs[a['image_id']]); x, y, w, h = a['bbox']
px = x + np.array(a['dp_x']) / 256 * w; py = y + np.array(a['dp_y']) / 256 * h
print('ref_model', a.get('ref_model'), 'dp_vertex', a['dp_vertex'], 'anns in image', [b['id'] for b in anns if b['image_id'] == a['image_id']])
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(img); ax[0].add_patch(plt.Rectangle((x, y), w, h, fill=False, ec='yellow', lw=2))
ax[1].imshow(img); ax[1].set_xlim(x - 10, x + w + 10); ax[1].set_ylim(y + h + 10, y - 10)
for A in ax:
    for i, (c, r) in enumerate(zip(px, py)):
        A.scatter(c, r, s=60, c='yellow', edgecolors='k')
        A.annotate(f'{i + 1} (v{a["dp_vertex"][i]})', (c, r), xytext=(6, -6), textcoords='offset points', color='yellow', weight='bold', fontsize=11)
    A.axis('off')
ax[0].set_title(f'full image, bbox of ann {args.ann_id}'); ax[1].set_title('bbox zoom, GT points at native resolution')
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
fig.tight_layout(); fig.savefig(args.output_path, dpi=90); print('saved', args.output_path)
