"""
Pull the hand-labeled evaluation pixels AWAY FROM THE OBJECT'S OUTLINE, keeping every
label's NUMBER (Tal, 2026-09-04).

Why: the eval set was built with Farthest Point Sampling over the mask, and FPS by
construction favours extremes -- 47 of the 100 labels sit within 8 px of the silhouette,
many of them exactly ON it (distance-to-edge = 1.0). A pixel on the outline is a bad
correspondence target: its DINO/SD feature is a mix of guitar and background, and the
vertex it should map to is itself on a silhouette the renderer may or may not show.

What this does NOT do: delete anything. Tal has to edit the mesh side by hand to match, so
the numbering has to survive -- an edge pixel is MOVED INWARD instead of dropped, and keeps
its label_id. `orig_row`/`orig_col`/`shift_px` are saved alongside so any move can be
inspected or undone.

How the new position is chosen: on the mask's Euclidean distance transform, take the
CLOSEST pixel to the original that is at least --min_edge_dist from the edge (minimal
displacement, so the point stays in the same part of the guitar). If the object is locally
too thin for that -- the neck, the headstock -- take the most interior pixel available
within --max_shift instead, i.e. the local ridge.

Run:
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/inset_eval_pixels.py \
  --min_edge_dist 20 --out_dir ./demo/guitar/eval_labels/v2_inset
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

from sample_eval_pixels import draw_numbered_labels


def inset_point(dist_map, r0, c0, min_edge_dist, max_shift, taken, min_separation):
    """Nearest pixel to (r0,c0) with distance-to-edge >= min_edge_dist, searched inside a
    +-max_shift window, and at least `min_separation` from every already-placed label in
    `taken`. Returns (row, col, reached_target).

    The separation constraint is what makes an aggressive --min_edge_dist usable at all: on
    the neck and the headstock the object is only ~70-90 px wide, so "go 30 px inside" alone
    sends six different labels to the SAME centerline pixel. Two labels on one pixel cannot
    be two evaluation points, and Tal has to pick a distinct vertex for each number.

    Falls back, in order: drop the edge target and keep separation (locally too thin), then
    drop separation too (nothing else available) -- both reported by the caller."""
    H, W = dist_map.shape
    r_lo, r_hi = max(0, r0 - max_shift), min(H, r0 + max_shift + 1)
    c_lo, c_hi = max(0, c0 - max_shift), min(W, c0 + max_shift + 1)
    sub = dist_map[r_lo:r_hi, c_lo:c_hi]
    rr, cc = np.mgrid[r_lo:r_hi, c_lo:c_hi]
    disp2 = (rr - r0) ** 2 + (cc - c0) ** 2
    in_range = disp2 <= max_shift ** 2

    free = in_range.copy()
    for (tr, tc) in taken:
        free &= ((rr - tr) ** 2 + (cc - tc) ** 2) >= min_separation ** 2

    deep = sub >= min_edge_dist
    for candidate, reached in ((free & deep, True), (free, False), (in_range & deep, True),
                               (in_range, False)):
        if candidate.any():
            if reached:
                # among all pixels deep enough, the one that moves the label least
                k = np.argmin(np.where(candidate, disp2, np.inf))
            else:
                # locally too thin: the most interior pixel available, nearest one on ties
                best = sub.max(initial=-1, where=candidate)
                tie = candidate & (sub >= best - 1e-9)
                k = np.argmin(np.where(tie, disp2, np.inf))
            return int(rr.flat[k]), int(cc.flat[k]), bool(reached)
    raise RuntimeError(f'no candidate pixel for label at ({r0},{c0})')


def draw_move_arrows(img_path, r0, c0, r1, c1, label_id, moved, output_path,
                     radius=6, font_size=22):
    """Before/after overlay: hollow grey dot at the old position, red dot at the new one,
    a yellow arrow between them. Unmoved labels are drawn as plain red dots."""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', font_size)
    w, h = img.size
    for i in range(len(label_id)):
        if moved[i]:
            draw.ellipse([c0[i] - radius, r0[i] - radius, c0[i] + radius, r0[i] + radius],
                         outline=(120, 120, 120), width=2)
            draw.line([c0[i], r0[i], c1[i], r1[i]], fill=(255, 220, 0), width=3)
        colour = (255, 0, 0) if moved[i] else (0, 170, 255)
        draw.ellipse([c1[i] - radius, r1[i] - radius, c1[i] + radius, r1[i] + radius],
                     fill=colour, outline=(0, 0, 0), width=2)
        text = str(int(label_id[i]))
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tx = min(max(c1[i] + radius + 2, 4), w - tw - 4)
        ty = min(max(r1[i] - radius - th, 4), h - th - 4)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                draw.text((tx + dx, ty + dy), text, font=font, fill=(255, 255, 255))
        draw.text((tx, ty), text, font=font, fill=(200, 0, 0))
    img.save(output_path)


def draw_mesh_relabel_map(img_path, r1, c1, label_id, moved, output_path,
                          radius=7, font_size=22):
    """Tal's working figure for the MESH side, and nothing else -- no run reads this.

    An unmoved label still points at the same pixel, so the vertex Tal already picked for it in
    MeshLab is still correct. Only the moved ones need a new vertex. Colouring by that split
    turns "re-pick 100 points" into "re-pick these 57", so this figure marks moved labels in RED
    and leaves the untouched ones small and GREEN."""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', font_size)
    w, h = img.size
    for i in range(len(label_id)):
        if moved[i]:
            rad, fill, text_col = radius, (255, 0, 0), (210, 0, 0)
        else:
            rad, fill, text_col = radius - 3, (0, 200, 60), (0, 120, 40)
        draw.ellipse([c1[i] - rad, r1[i] - rad, c1[i] + rad, r1[i] + rad],
                     fill=fill, outline=(0, 0, 0), width=2)
        text = str(int(label_id[i]))
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tx = min(max(c1[i] + rad + 2, 4), w - tw - 4)
        ty = min(max(r1[i] - rad - th, 4), h - th - 4)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                draw.text((tx + dx, ty + dy), text, font=font, fill=(255, 255, 255))
        draw.text((tx, ty), text, font=font, fill=text_col)

    legend = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
    lines = [((255, 0, 0), f'RED ({int(moved.sum())}): moved -- re-pick this vertex'),
             ((0, 200, 60), f'GREEN ({int((~moved).sum())}): unchanged -- keep the vertex')]
    pad, dot, gap = 12, 18, 34
    bw = max(draw.textlength(t, font=legend) for _, t in lines) + dot + 3 * pad
    bh = 2 * pad + gap * len(lines)
    x0, y0 = w - bw - 14, 14
    draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    for k, (col, text) in enumerate(lines):
        y = y0 + pad + gap * k
        draw.ellipse([x0 + pad, y + 2, x0 + pad + dot, y + 2 + dot], fill=col, outline=(0, 0, 0), width=2)
        draw.text((x0 + pad + dot + pad, y), text, font=legend, fill=(0, 0, 0))
    img.save(output_path)


def main(args):
    labels = torch.load(args.labels_path, map_location='cpu')
    mask = torch.load(args.mask_path, map_location='cpu').numpy()
    dist_map = distance_transform_edt(mask)

    label_id = labels['label_id'].numpy()
    r0 = labels['row'].numpy()
    c0 = labels['col'].numpy()
    n = len(label_id)

    r1 = np.zeros(n, dtype=np.int64)
    c1 = np.zeros(n, dtype=np.int64)
    reached = np.zeros(n, dtype=bool)
    # place the most constrained labels first -- the ones with the least interior room nearby
    # (neck, headstock, horns). Placing a roomy body point first would let it sit wherever and
    # then block the one point that had only one legal spot.
    overrides = {}
    for item in filter(None, (x.strip() for x in args.label_overrides.split(','))):
        k, v = item.split(':')
        overrides[int(k)] = float(v)
    unknown = set(overrides) - set(int(x) for x in label_id)
    if unknown:
        raise ValueError(f'--label_overrides names labels that do not exist: {sorted(unknown)}')
    target = np.full(n, float(args.min_edge_dist))
    is_override = np.zeros(n, dtype=bool)
    for i in range(n):
        if int(label_id[i]) in overrides:
            target[i] = overrides[int(label_id[i])]
            is_override[i] = True
    if overrides:
        print('per-label overrides: '
              + ', '.join(f'{k} -> {v:g}px' for k, v in sorted(overrides.items())))

    room = np.array([dist_map[max(0, int(r0[i]) - args.max_shift):int(r0[i]) + args.max_shift + 1,
                              max(0, int(c0[i]) - args.max_shift):int(c0[i]) + args.max_shift + 1].max()
                     for i in range(n)])
    taken = []
    order = np.lexsort((room, ~is_override))  # overridden labels first, then least room first
    for i in order:
        r1[i], c1[i], reached[i] = inset_point(dist_map, int(r0[i]), int(c0[i]),
                                                target[i], args.max_shift,
                                                taken, args.min_separation)
        taken.append((r1[i], c1[i]))

    d_before = dist_map[r0, c0]
    d_after = dist_map[r1, c1]
    shift = np.hypot(r1 - r0, c1 - c0)
    moved = shift > 0

    # the new position must land in the same connected mask -- a cheap sanity check that the
    # search never jumped a background gap (it cannot, EDT is 0 outside the mask, but assert it)
    assert mask[r1, c1].all(), 'a moved label landed off-mask'

    print(f'{int(moved.sum())} of {n} labels moved  '
          f'(target >= {args.min_edge_dist} px from the edge, max shift {args.max_shift} px)')
    print(f'  edge distance  before: min {d_before.min():.1f}  median {np.median(d_before):.1f}')
    print(f'  edge distance  after : min {d_after.min():.1f}  median {np.median(d_after):.1f}')
    print(f'  shift          median {np.median(shift[moved]):.1f} px  max {shift.max():.1f} px'
          if moved.any() else '  nothing moved')
    sep = np.hypot(r1[:, None] - r1[None, :], c1[:, None] - c1[None, :]).astype(float)
    np.fill_diagonal(sep, np.inf)
    print(f'  label separation  min {sep.min():.1f} px  (original set: 41.7 px)')
    if is_override.any():
        print('  overridden labels after placement: '
              + ', '.join(f'{int(label_id[i])} d={d_after[i]:.1f} (shift {shift[i]:.0f}px)'
                          for i in np.where(is_override)[0]))
    if (~reached).any():
        print('  LOCALLY TOO THIN for the target, took the local ridge instead: '
              + ', '.join(f'{int(label_id[i])} (d={d_after[i]:.1f})' for i in np.where(~reached)[0]))

    os.makedirs(args.out_dir, exist_ok=True)
    out = dict(labels)
    out['row'] = torch.from_numpy(r1)
    out['col'] = torch.from_numpy(c1)
    out['orig_row'] = torch.from_numpy(r0)
    out['orig_col'] = torch.from_numpy(c0)
    out['shift_px'] = torch.from_numpy(shift)
    out['edge_dist_before'] = torch.from_numpy(d_before)
    out['edge_dist_after'] = torch.from_numpy(d_after)
    out['reached_target'] = torch.from_numpy(reached)
    out['inset_min_edge_dist'] = args.min_edge_dist
    out['inset_max_shift'] = args.max_shift
    out['inset_min_separation'] = args.min_separation
    out['inset_target'] = torch.from_numpy(target)          # [N], per-label, = min_edge_dist unless overridden
    out['inset_label_overrides'] = args.label_overrides
    out['inset_source_labels'] = args.labels_path
    # mask_index refers to torch.where(mask) order and no longer matches the moved pixel
    flat_idx = np.flatnonzero(mask.ravel())
    lookup = -np.ones(mask.size, dtype=np.int64)
    lookup[flat_idx] = np.arange(flat_idx.size)
    out['mask_index'] = torch.from_numpy(lookup[r1 * mask.shape[1] + c1])
    labels_out = os.path.join(args.out_dir, 'pixel_labels.pth')
    torch.save(out, labels_out)
    print(f'saved {labels_out}')

    csv_path = os.path.join(args.out_dir, 'pixel_labels.csv')
    with open(csv_path, 'w') as f:
        f.write('label_id,row,col,orig_row,orig_col,shift_px,edge_dist_before,edge_dist_after,'
                'reached_target,mask_index,fps_rank\n')
        for i in range(n):
            f.write(f'{int(label_id[i])},{r1[i]},{c1[i]},{r0[i]},{c0[i]},{shift[i]:.1f},'
                    f'{d_before[i]:.1f},{d_after[i]:.1f},{int(reached[i])},'
                    f"{int(out['mask_index'][i])},{int(labels['fps_rank'][i])}\n")
    print(f'saved {csv_path}')

    numbered = os.path.join(args.out_dir, 'pixel_labels_numbered.png')
    draw_numbered_labels(args.img_path, r1, c1, label_id, numbered)
    print(f'saved {numbered}')

    arrows = os.path.join(args.out_dir, 'pixel_labels_moved_arrows.png')
    draw_move_arrows(args.img_path, r0, c0, r1, c1, label_id, moved, arrows)
    print(f'saved {arrows}')

    relabel = os.path.join(args.out_dir, 'MESH_RELABEL_moved_in_red.png')
    draw_mesh_relabel_map(args.img_path, r1, c1, label_id, moved, relabel)
    print(f'saved {relabel}')

    moved_txt = os.path.join(args.out_dir, 'MESH_RELABEL_moved_label_ids.txt')
    with open(moved_txt, 'w') as f:
        f.write(f'# labels whose pixel MOVED -- these {int(moved.sum())} vertices need re-picking\n')
        f.write(', '.join(str(int(x)) for x in sorted(label_id[moved])) + '\n')
        f.write(f'# labels UNCHANGED -- keep the vertex already picked ({int((~moved).sum())})\n')
        f.write(', '.join(str(int(x)) for x in sorted(label_id[~moved])) + '\n')
    print(f'saved {moved_txt}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--labels_path', type=str, default='./demo/guitar/eval_labels/pixel_labels.pth')
    p.add_argument('--mask_path', type=str,
                   default='./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth')
    p.add_argument('--img_path', type=str, default='./images/guitar_01.png')
    p.add_argument('--min_edge_dist', type=float, default=20.0,
                   help='how far inside the silhouette every label must end up, in photo pixels')
    p.add_argument('--max_shift', type=int, default=60,
                   help='hard cap on how far a label may travel from its original position')
    p.add_argument('--min_separation', type=float, default=25.0,
                   help='no two labels may end up closer than this, in photo pixels. Without it '
                        'an aggressive --min_edge_dist merges labels onto one centerline pixel '
                        'wherever the object is thin (the original set has 41.7 px min separation)')
    p.add_argument('--label_overrides', type=str, default='',
                   help='per-label edge-distance targets that override --min_edge_dist, as '
                        '"label:target,label:target" (e.g. "2:30,3:40"). Tal reviews the figure and '
                        'names the points that still sit too close to the outline; this is how those '
                        'get pushed further without disturbing the other 96. Overridden labels are '
                        'placed FIRST, so the separation constraint yields to them rather than to '
                        'whichever neighbour happened to be placed earlier.')
    p.add_argument('--out_dir', type=str, default='./demo/guitar/eval_labels/v2_inset')
    main(p.parse_args())
