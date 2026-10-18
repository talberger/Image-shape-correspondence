"""Occlusion gap fill of an object mask -- PROTOTYPE + survey (Tal, 2026-10-07). Not wired into training yet.
An occluder (fence, branches) splits the LVIS mask into parts and the 2D geodesic table keeps only the largest one.
fill = (1) closing with the smallest radius r that joins the parts, keeping only added blobs that touch >= 2 parts;
       (2) with that same r: every gap narrower than the occluder (plain closing), and every enclosed hole that touches the fill.
A one-part mask is returned unchanged (r = 0), so un-occluded instances are not affected.
Usage: python analysis/gapfill_preview.py --out_dir demo/shic/occlusion_gapfill [--classes bear horse]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import ndimage
S8 = np.ones((3, 3))

def closing(m, r):
    p = np.pad(m, r + 2); d = ndimage.distance_transform_edt(~p) <= r
    return (ndimage.distance_transform_edt(d) > r)[r + 2:-r - 2, r + 2:-r - 2]

def gap_fill(m, rmax=60, target=0.995, min_part_frac=0.0):
    """-> (filled mask, radius used, fraction of the mask's pixels in the largest connected piece after the fill)."""
    lab, n = ndimage.label(m, structure=S8)
    if n <= 1: return m.copy(), 0, 1.0
    f, frac = m, 0.0
    for r in range(1, rmax + 1):
        add = closing(m, r) & ~m; al, an = ndimage.label(add, structure=S8); keep = np.zeros_like(m)
        for i, sl in enumerate(ndimage.find_objects(al), 1):
            e = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
            blob = ndimage.binary_dilation(al[e] == i, structure=S8)
            if len(np.setdiff1d(np.unique(lab[e][blob]), [0])) >= 2: keep[e] |= (al[e] == i)
        f = m | keep; fl, _ = ndimage.label(f, structure=S8); frac = np.bincount(fl[m]).max() / m.sum()
        if frac >= target: break
    f2 = f | closing(f, r)                                   # every gap narrower than the occluder
    holes = ndimage.binary_fill_holes(f2) & ~f2; hl, hn = ndimage.label(holes, structure=S8)
    grown = ndimage.binary_dilation(f2 & ~m, structure=S8)   # enclosed holes that touch the fill (fence junctions)
    for i in range(1, hn + 1):
        if (grown & (hl == i)).any(): f2 |= (hl == i)
    return f2, r, frac

if __name__ == '__main__':
    import torch
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from PIL import Image
    from eval_densepose_lvis import load_instances, get_image, polygon_mask
    ap = argparse.ArgumentParser(); ap.add_argument('--out_dir', required=True); ap.add_argument('--classes', nargs='+', default=['bear', 'horse'])
    ap.add_argument('--n_show', type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    for cls in a.classes:
        anns, imgs = load_instances(cls); res_dir = {'bear': 'results_twostage', 'horse': 'results_horse_twostage'}[cls]
        rows = []
        for an in anns:
            if not os.path.exists(f'data/densepose_lvis/{res_dir}/lvis{an["id"]}.json'): continue
            im = imgs[an['image_id']]; x, y, w, h = an['bbox']; box = (round(x), round(y), round(x + w), round(y + h))
            s = 400 / max(box[2] - box[0], box[3] - box[1]); size = (round((box[2] - box[0]) * s), round((box[3] - box[1]) * s))
            m = np.array(Image.fromarray(polygon_mask(an['segmentation'], im['height'], im['width'])).crop(box).resize(size, Image.NEAREST)) > 0
            lab, n = ndimage.label(m, structure=S8); big = np.bincount(lab[m]).max() / m.sum()
            f, r, frac = gap_fill(m)
            px = np.clip(np.round((x + np.array(an['dp_x']) / 256 * w - box[0]) * s), 0, size[0] - 1).astype(int)
            py = np.clip(np.round((y + np.array(an['dp_y']) / 256 * h - box[1]) * s), 0, size[1] - 1).astype(int)
            bigmask = lab == np.bincount(lab[m] if m.any() else [0]).argmax()
            rows.append(dict(ann=an['id'], parts=int(n), seen_today=float(big), radius=int(r), connected=float(frac), added=float((f & ~m).sum() / m.sum()),
                             gt_outside=int((~bigmask[py, px]).sum()), n_gt=len(px), m=m, f=f, box=box, size=size, image_id=an['image_id']))
        multi = [q for q in rows if q['seen_today'] < 0.95]
        print(f'{cls}: {len(rows)} instances | more than one part: {sum(q["parts"] > 1 for q in rows)} | training sees < 95% of the mask: {len(multi)} '
              f'| < 80%: {sum(q["seen_today"] < 0.8 for q in rows)} | < 50%: {sum(q["seen_today"] < 0.5 for q in rows)} '
              f'| GT points outside the seen part: {sum(q["gt_outside"] for q in rows)}/{sum(q["n_gt"] for q in rows)} '
              f'| after fill: min connected {min(q["connected"] for q in rows):.3f}, max radius {max(q["radius"] for q in rows)}, max added {max(q["added"] for q in rows):.2f}')
        json.dump([{k: v for k, v in q.items() if k not in ('m', 'f', 'box', 'size')} for q in rows], open(f'{a.out_dir}/survey_{cls}.json', 'w'), indent=1)
        show = sorted(rows, key=lambda q: q['seen_today'])[:a.n_show]
        fig, axes = plt.subplots(3, len(show), figsize=(3.6 * len(show), 11))
        for j, q in enumerate(show):
            img = np.asarray(get_image(imgs[q['image_id']]).crop(q['box']).resize(q['size'], Image.BICUBIC)); m, f = q['m'], q['f']
            lab, _ = ndimage.label(m, structure=S8); big = lab == np.bincount(lab[m]).argmax()
            A = np.full(img.shape, 255, np.uint8); A[m] = (235, 150, 150); A[big] = (160, 215, 160)
            B = np.full(img.shape, 255, np.uint8); B[m] = (160, 215, 160); B[f & ~m] = (60, 90, 230)
            axes[0, j].imshow(img); axes[0, j].set_title(f'{cls} {q["ann"]}', fontsize=11)
            axes[1, j].imshow(A); axes[1, j].set_title(f'today: {q["parts"]} parts, sees {100 * q["seen_today"]:.0f}%', fontsize=10)
            axes[2, j].imshow(B); axes[2, j].set_title(f'filled: {100 * q["connected"]:.0f}% connected, +{100 * q["added"]:.0f}% px', fontsize=10)
            for k in range(3): axes[k, j].axis('off')
        fig.suptitle(f'{cls}: the {len(show)} most fragmented masks. Row 2: green = what the training sees today, red = dropped. Row 3: blue = gap fill.', fontsize=13)
        fig.tight_layout(); fig.savefig(f'{a.out_dir}/gapfill_survey_{cls}.png', dpi=75); print('saved', f'{a.out_dir}/gapfill_survey_{cls}.png')
