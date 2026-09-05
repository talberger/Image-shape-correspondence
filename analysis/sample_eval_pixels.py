"""
Build a hand-labeled pixel<->vertex evaluation set: pick N pixels spread across the whole
masked region of the real photo (Farthest Point Sampling in pixel space, not uniform random --
uniform random clusters on the body and undersamples thin regions like the neck/headstock),
number them 0..N-1 in SPATIAL (serpentine) order -- consecutive numbers sit next to each other
on the photo, so finding "the next point to label" is a glance rather than a hunt across the
whole image -- and save both
a machine-readable label file and a numbered visualization image. Tal then manually marks each
corresponding vertex in MeshLab (the "Point Picking" / "PP" tool), typing the SAME number as
each picked point's name -- so the label file and the MeshLab .pp file can later be joined by
name, robust to picking out of order or re-picking a mistake, rather than relying on click
order alone.

Run:
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/sample_eval_pixels.py \
  --mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --num_pixels 100 --seed 0 \
  --out_dir ./demo/guitar/eval_labels

Numbering note (2026-08-25): `--numbering serpentine` is the default and produces spatially
ordered labels. The original 2026-08-24 run of this script used `--numbering fps` (labels in raw
Farthest-Point-Sampling order, which is spatially scattered by construction) -- pass that flag to
reproduce it. WHICH of the two was used is recorded in the output .pth's `numbering` field, and
the .pth/.csv also carry `fps_rank` (the FPS-order index each point had), so a label file written
against one numbering can always be remapped to the other. The SELECTED PIXELS ARE IDENTICAL
either way at the same --seed; only the numbers drawn on them change.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def farthest_point_sampling(points, n_samples, seed):
    """Greedy FPS: start from a seeded-random point, repeatedly add whichever
    remaining point is farthest (max of the min-distance-to-any-selected-point).
    Gives even geometric coverage of `points`, unlike uniform random which is
    biased toward wherever `points` is denser (the guitar body, in this project)."""
    rng = np.random.RandomState(seed)
    n = points.shape[0]
    selected = np.zeros(n_samples, dtype=np.int64)
    selected[0] = rng.randint(n)
    min_dist = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for i in range(1, n_samples):
        selected[i] = np.argmax(min_dist)
        new_dist = np.sum((points - points[selected[i]]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, new_dist)
    return selected


def serpentine_order(pixel_row, pixel_col, band_height):
    """Order points so consecutive numbers are spatial neighbours: slice the photo into
    horizontal bands of `band_height` rows, walk the bands top-to-bottom, and alternate the
    within-band direction (left-to-right, then right-to-left, ...). The alternation is what
    keeps the LAST point of one band next to the FIRST point of the next -- a plain
    left-to-right scanline would jump the full width of the object at every band boundary."""
    band = (pixel_row // band_height).astype(np.int64)
    # negate the column on odd bands so a single ascending sort walks them backwards
    signed_col = np.where(band % 2 == 0, pixel_col, -pixel_col).astype(np.int64)
    # np.lexsort takes the PRIMARY key last
    return np.lexsort((pixel_row.astype(np.int64), signed_col, band))


def draw_numbered_labels(img_path, pixel_row, pixel_col, label_id, output_path,
                          radius=6, font_size=22, margin=4):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', font_size)
    w, h = img.size
    for i in range(len(pixel_row)):
        r, c = int(pixel_row[i]), int(pixel_col[i])
        draw.ellipse([c - radius, r - radius, c + radius, r + radius],
                     fill=(255, 0, 0), outline=(0, 0, 0), width=2)
        text = str(int(label_id[i]))
        tb = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        # flip to whichever side of the dot actually fits inside the image, so
        # numbers near the photo's edges (common once points reach the body's
        # outline) never get clipped
        tx = c + radius + 2 if c + radius + 2 + text_w + margin <= w else c - radius - 2 - text_w
        ty = r - radius - text_h if r - radius - text_h - margin >= 0 else r + radius + 2
        tx = min(max(tx, margin), w - text_w - margin)
        ty = min(max(ty, margin), h - text_h - margin)
        # white halo behind the number so it stays legible over any photo color
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                draw.text((tx + dx, ty + dy), text, font=font, fill=(255, 255, 255))
        draw.text((tx, ty), text, font=font, fill=(200, 0, 0))
    img.save(output_path)


def main(args):
    mask = torch.load(args.mask_path, map_location='cpu')
    mask_row, mask_col = torch.where(mask)
    n_masked = mask_row.shape[0]
    if not (1 <= args.num_pixels <= n_masked):
        raise ValueError(f'--num_pixels must be in [1, {n_masked}]')

    points = np.stack([mask_row.numpy(), mask_col.numpy()], axis=1).astype(np.float64)
    selected_idx = farthest_point_sampling(points, args.num_pixels, args.seed)

    pixel_row = mask_row[selected_idx]
    pixel_col = mask_col[selected_idx]

    # FPS picks points in a spatially scattered order by construction (each new point is the
    # farthest one from everything picked so far), which makes hand-labeling painful: point i+1
    # is typically on the opposite side of the object from point i. Renumber into spatial order
    # instead. This does NOT change WHICH pixels were selected -- only the number each carries.
    if args.numbering == 'serpentine':
        order = serpentine_order(pixel_row.numpy(), pixel_col.numpy(), args.band_height)
    elif args.numbering == 'fps':
        order = np.arange(args.num_pixels)
    else:
        raise ValueError(f'unknown --numbering {args.numbering}')
    fps_rank = order  # point with new label i had FPS-order index fps_rank[i]
    selected_idx = selected_idx[order]
    pixel_row = pixel_row[order]
    pixel_col = pixel_col[order]

    # 0-based (0..num_pixels-1), matching MeshLab's own default Point Picking
    # point names ("0", "1", ...) so Tal doesn't have to rename anything there
    # if just picking in order -- naming to the same number is still recommended
    # for out-of-order/mistake robustness (see this script's own docstring).
    label_id = np.arange(0, args.num_pixels)

    os.makedirs(args.out_dir, exist_ok=True)
    labels_path = os.path.join(args.out_dir, 'pixel_labels.pth')
    torch.save({
        'label_id': torch.from_numpy(label_id),      # [N], 0-based, matches the numbers drawn on the image
        'row': pixel_row,                              # [N], pixel row in the ORIGINAL (unpadded) photo
        'col': pixel_col,                               # [N], pixel col in the ORIGINAL (unpadded) photo
        'mask_index': torch.from_numpy(selected_idx),  # [N], index into torch.where(mask) order --
                                                          # same convention as this project's other
                                                          # explicit-pixel-index scripts (e.g. tracked_9_pixels)
        'fps_rank': torch.from_numpy(fps_rank),        # [N], the FPS-order index of each point --
                                                          # lets a file numbered under one --numbering
                                                          # be remapped to the other
        'mask_path': args.mask_path,
        'img_path': args.img_path,
        'sampling': f'fps_seed{args.seed}',
        'numbering': args.numbering,
        'band_height': args.band_height if args.numbering == 'serpentine' else None,
    }, labels_path)
    print(f'saved {args.num_pixels} labels to {labels_path}')

    viz_path = os.path.join(args.out_dir, 'pixel_labels_numbered.png')
    draw_numbered_labels(args.img_path, pixel_row.numpy(), pixel_col.numpy(), label_id, viz_path)
    print(f'saved numbered visualization to {viz_path}')

    csv_path = os.path.join(args.out_dir, 'pixel_labels.csv')
    with open(csv_path, 'w') as f:
        f.write('label_id,row,col,mask_index,fps_rank\n')
        for i in range(args.num_pixels):
            f.write(f'{label_id[i]},{int(pixel_row[i])},{int(pixel_col[i])},'
                     f'{int(selected_idx[i])},{int(fps_rank[i])}\n')
    print(f'saved human-readable csv to {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=str,
                         default='./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth')
    parser.add_argument('--img_path', type=str, default='./images/guitar_01.png')
    parser.add_argument('--num_pixels', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='./demo/guitar/eval_labels')
    parser.add_argument('--numbering', type=str, default='serpentine', choices=['serpentine', 'fps'],
                         help='serpentine: consecutive numbers are spatial neighbours (default, '
                               'much easier to hand-label). fps: raw Farthest-Point-Sampling order, '
                               'as the original 2026-08-24 run used.')
    parser.add_argument('--band_height', type=int, default=50,
                         help='serpentine band height in pixels; ~one row of FPS points per band')
    main(parser.parse_args())
