# VLPart part masks (external/vlpart/dump_instances.py -> instances.pth) -> label maps, one per option, next to the current SAM 1 map.
# Each option: keep masks with score >= thr, paint largest-first (build_label_map), merge 1.5% + fillseg (the training recipe).
import argparse, os, sys
import numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from build_fine_segments import build_label_map, merge_small_segments, fill_unlabelled, load_coarse_mask

p = argparse.ArgumentParser()
p.add_argument('--src', required=True); p.add_argument('--ann_id', default='56069')
p.add_argument('--vlpart_dir', required=True); p.add_argument('--save_dir', required=True)
a = p.parse_args()
img = np.array(Image.open(f'{a.src}/lvis{a.ann_id}/lvis{a.ann_id}.png').convert('RGB'))
fg = load_coarse_mask(f'{a.src}/lvis{a.ann_id}/lvis{a.ann_id}_sam3_mask.pth')
os.makedirs(a.save_dir, exist_ok=True)
cur = torch.load(f'{a.src}/lvis{a.ann_id}/lvis{a.ann_id}_fine_segments.pth').numpy()
options = [('partimagenet', 0.3), ('pascal_part', 0.5), ('pascal_part', 0.3)]
panels = [('current SAM 1 map', cur)]
for vocab, thr in options:
    d = torch.load(f'{a.vlpart_dir}/{vocab}/instances.pth')
    anns = [{'segmentation': m.numpy()} for m, s in zip(d['masks'], d['scores']) if s >= thr]
    lab, _ = build_label_map(anns, fg, 50)
    lab, _ = merge_small_segments(lab, 0.015 * int(fg.sum()))
    lab, cnt = fill_unlabelled(lab, fg)
    np.save(f'{a.save_dir}/labels_{vocab}_thr{thr}.npy', lab)
    panels.append((f'VLPart {vocab}, score>={thr}', lab))
fig, axs = plt.subplots(1, len(panels), figsize=(4 * len(panels), 6.4))
for ax, (title, lab) in zip(axs, panels):
    n = int(lab.max()); pal = np.random.RandomState(0).rand(max(n, 1) + 1, 3); pal[0] = 0
    bl = img / 255.0 * 0.35 + pal[lab] * 0.65; bl[lab == 0] = img[lab == 0] / 255.0 * 0.3
    ax.imshow(np.clip(bl, 0, 1)); ax.axis('off')
    fr = sorted([100 * (lab == i).sum() / fg.sum() for i in range(1, n + 1)], reverse=True)
    ax.set_title(f'{title}\n{n} segs: ' + ' / '.join(f'{f:.0f}' for f in fr) + ' %', fontsize=9)
    print(f'{title:35s} {n} segs: ' + ' / '.join(f'{f:.1f}' for f in fr))
fig.tight_layout(); fig.savefig(f'{a.save_dir}/options_lvis{a.ann_id}.png', dpi=110); plt.close(fig)
