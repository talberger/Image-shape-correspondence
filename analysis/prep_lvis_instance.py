"""Cut one DensePose-LVIS instance into this project's per-image inputs.

Crop = the instance bbox (the input that reproduces SHIC's 33.0, see analysis/eval_densepose_lvis.py),
upscaled so the long side is --long_side px (bear3 is 808x750; every pixel-unit knob of the recipe --
2048 centres, 16 neighbours, cycle stride 5 -- was tuned at that scale). The mask is the LVIS polygon
(replaces the SAM click mask). GT points are written in crop-pixel coords for the eval.

Out (in --out_dir): <tag>.png, <tag>_mask.pth (bool [H,W]), <tag>_gt.pth {rows, cols, vertex, ann_id}
"""
import argparse, os
import numpy as np
import torch
from PIL import Image
from eval_densepose_lvis import load_instances, get_image, polygon_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann_id', type=int, required=True)
    ap.add_argument('--class_name', default='bear')
    ap.add_argument('--long_side', type=int, default=400,
                    help='2026-09-20: was 800, which came from bear3/guitar happening to be ~800px '
                         'natively, not from any analysis. No LVIS bear bbox reaches 800 (median long '
                         'side 289, max 612), and every backbone resizes internally anyway (DINO 224, '
                         'SD 512, SAM 1024), so the upscale added no information and only inflated the '
                         'mask pixel count that the 2D geodesic Dijkstra scales with. Kept FIXED rather '
                         'than native so num_center_pixels/min_pixels_in_object/cyc_return_stride keep '
                         'meaning the same thing across instances that differ ~20x in bbox area.')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anns, imgs = load_instances(args.class_name)
    a = next(a for a in anns if a['id'] == args.ann_id)
    img = get_image(imgs[a['image_id']])
    W0, H0 = img.size
    x, y, w, h = a['bbox']
    box = (round(x), round(y), round(x + w), round(y + h))
    s = args.long_side / max(box[2] - box[0], box[3] - box[1])
    size = (round((box[2] - box[0]) * s), round((box[3] - box[1]) * s))
    crop = img.crop(box).resize(size, Image.BICUBIC)
    m = Image.fromarray(polygon_mask(a['segmentation'], H0, W0)).crop(box).resize(size, Image.NEAREST)
    mask = torch.from_numpy(np.array(m) > 0)

    px = x + np.array(a['dp_x']) / 256.0 * w
    py = y + np.array(a['dp_y']) / 256.0 * h
    cols = np.clip(np.round((px - box[0]) * s), 0, size[0] - 1).astype(int)
    rows = np.clip(np.round((py - box[1]) * s), 0, size[1] - 1).astype(int)

    tag = f'lvis{args.ann_id}'
    crop.save(os.path.join(args.out_dir, f'{tag}.png'))
    torch.save(mask, os.path.join(args.out_dir, f'{tag}_mask.pth'))
    torch.save({'rows': torch.tensor(rows), 'cols': torch.tensor(cols), 'vertex': torch.tensor(a['dp_vertex']),
                'ann_id': args.ann_id, 'in_mask': mask[rows, cols]}, os.path.join(args.out_dir, f'{tag}_gt.pth'))
    print(f'{tag}: bbox {w:.0f}x{h:.0f} -> crop {size[0]}x{size[1]} (x{s:.2f}), mask {int(mask.sum())} px, '
          f'{len(rows)} GT points, {int(mask[rows, cols].sum())} inside the mask')


if __name__ == '__main__':
    main()
