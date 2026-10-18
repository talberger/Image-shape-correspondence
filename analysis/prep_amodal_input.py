"""Step 1/3 of the amodal mask (bsb env): the 256x256 input pix2gestalt wants for one DensePose-LVIS instance.
A SQUARE crop around the bbox with a 15% margin, cut from the original image (real context, not padding), + the visible
LVIS mask, + the geometry needed to bring the result back to the run crop that prep_lvis_instance.py writes.
Out: <out_dir>/<tag>_amodal_input.npz. Usage: python analysis/prep_amodal_input.py --ann_id 7107 --class_name bear --out_dir <run dir>"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from eval_densepose_lvis import load_instances, get_image, polygon_mask

ap = argparse.ArgumentParser()
ap.add_argument('--ann_id', type=int, required=True); ap.add_argument('--class_name', default='bear')
ap.add_argument('--out_dir', required=True); ap.add_argument('--long_side', type=int, default=400); ap.add_argument('--margin', type=float, default=1.15)
ap.add_argument('--mask_source', default='polygon', choices=['polygon', 'sam3'])   # sam3: the visible mask is <out_dir>/<tag>_sam3_mask.pth (run-crop grid), no polygon
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
anns, imgs = load_instances(a.class_name)
an = next(x for x in anns if x['id'] == a.ann_id); im = imgs[an['image_id']]
img = np.asarray(get_image(im).convert('RGB')); m = polygon_mask(an['segmentation'], im['height'], im['width']) > 0
x, y, w, h = an['bbox']
box = (round(x), round(y), round(x + w), round(y + h))                       # exactly prep_lvis_instance.py's crop
s = a.long_side / max(box[2] - box[0], box[3] - box[1]); size = (round((box[2] - box[0]) * s), round((box[3] - box[1]) * s))
if a.mask_source == 'sam3':   # bring the run-crop mask back to image pixels, inside the bbox
    import torch
    ms = torch.load(os.path.join(a.out_dir, f'lvis{a.ann_id}_sam3_mask.pth')).numpy().astype(np.uint8) * 255
    x0, y0, x1, y1 = max(box[0], 0), max(box[1], 0), min(box[2], img.shape[1]), min(box[3], img.shape[0])
    full = np.asarray(Image.fromarray(ms).resize((box[2] - box[0], box[3] - box[1]), Image.NEAREST)) > 0
    m = np.zeros(img.shape[:2], bool); m[y0:y1, x0:x1] = full[y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]]
cx, cy, half = x + w / 2, y + h / 2, max(w, h) * a.margin / 2
sq = [int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half)]   # x0, y0, x1, y1 (may leave the image)
pad = max(0, -sq[0], -sq[1], sq[2] - img.shape[1], sq[3] - img.shape[0])
ip = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge'); mp = np.pad(m, pad)
c = ip[sq[1] + pad:sq[3] + pad, sq[0] + pad:sq[2] + pad]; cm = mp[sq[1] + pad:sq[3] + pad, sq[0] + pad:sq[2] + pad]
tag = f'lvis{a.ann_id}'
np.savez(os.path.join(a.out_dir, f'{tag}_amodal_input.npz'),
         image=np.asarray(Image.fromarray(c).resize((256, 256), Image.LANCZOS)),
         mask=np.asarray(Image.fromarray((cm * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST)),
         square=np.array(sq), box=np.array(box), size=np.array(size))
print(f'{tag}: square crop {sq} ({sq[2] - sq[0]} px) -> 256, bbox {box} -> run crop {size}')
