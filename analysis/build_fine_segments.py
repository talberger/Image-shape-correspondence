"""Build a part-level (fine) segmentation label map for an image.

Why this exists
---------------
The phase-2 mapping loss needs to know, for a given pixel, which other pixels
belong to the *same part* of the object -- its Euclidean neighbours are drawn
only from within that part. Nothing in the repo produced such a map:

  - The `*_sam_mask_2.pth` files on disk are COARSE whole-object masks. (Loaded
    and measured: three of them, from three different clicks, are (1080,810)
    bool with ~23.5% coverage and pairwise IoU 0.955-0.981 -- three copies of
    the whole guitar.) They are what the analysis scripts consume via --mask_path.
  - BSB *does* compute a genuine part-level mask on every demo run
    (best_segmentation_buddies.py:178 segment_image() -> test_seg_click with
    --test_mask_idx 0), but it is used only as an IoU reference and persisted
    only as a matplotlib figure -- so it is not recoverable as an exact mask.
    Only prepare_data() (line 219) ever torch.saves a mask, and that is the
    coarse one.

The machinery for a real map is already here though: SamAutomaticMaskGenerator
is vendored and imported at encoder.py:21 and configured at encoder.py:94-101
with a 32x32 point grid, but its output is only ever drawn into PNG overlays.
This script reuses that configuration and keeps the labels.

Output: an int32 [H, W] label map (0 = background/unassigned, 1..N = parts),
restricted to the coarse object mask, plus a color PNG and per-segment pixel
counts for eyeballing. Segments are painted largest-first so smaller parts end
up on top -- the same ordering show_anns uses (modules/utils.py:125).

Run:
cd $DISC_ROOT && python \
  analysis/build_fine_segments.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SAM_repo.segment_anything import SamAutomaticMaskGenerator
from modules.sam import load_img_seg_model
from modules.utils import get_img_name


def load_coarse_mask(mask_path):
    """Same contract as analysis/pixel_coloring.py's load_mask."""
    mask = torch.load(mask_path, map_location='cpu')
    if mask.dtype != torch.bool:
        raise ValueError('expected a bool mask, got %s' % mask.dtype)
    if mask.dim() != 2:
        raise ValueError('expected mask of shape [H, W], got %s' % (tuple(mask.shape),))
    return mask.numpy()


def build_label_map(anns, coarse_mask, min_pixels_in_object):
    """Paint SAM's masks into one int32 label map, largest first.

    Painting largest-first means a smaller part drawn later overwrites the
    bigger region containing it, which is what we want: the finest available
    part wins. Anything outside the coarse object mask is dropped, and segments
    left with too few pixels inside the object are discarded (a segment smaller
    than the loss's neighbour count is useless for it).
    """
    height, width = coarse_mask.shape
    labels = np.zeros((height, width), dtype=np.int32)

    # Clip to the object first, THEN sort, so ordering reflects how much of each
    # segment actually lands on the object rather than its full-image area.
    clipped = []
    for ann in anns:
        seg = ann['segmentation'] & coarse_mask
        n = int(seg.sum())
        if n >= min_pixels_in_object:
            clipped.append((n, seg))
    clipped.sort(key=lambda t: t[0], reverse=True)

    for i, (_, seg) in enumerate(clipped):
        labels[seg] = i + 1

    # Painting overwrites, so a segment can be fully covered by later ones.
    # Renumber to keep labels contiguous and drop the ones that vanished.
    final = np.zeros_like(labels)
    counts = []
    next_label = 1
    for old_label in range(1, len(clipped) + 1):
        m = labels == old_label
        n = int(m.sum())
        if n >= min_pixels_in_object:
            final[m] = next_label
            counts.append(n)
            next_label += 1
    return final, counts


def merge_small_segments(labels, min_pixels):
    """Segments with fewer than `min_pixels` pixels get no label of their own (Tal, 2026-10-06).

    Each one, smallest first, is absorbed by the neighbouring segment it shares the longest border with
    (4-neighbourhood) among the segments that are NOT small; one touching none of those goes to the largest. Labels are then
    renumbered 1..K by size, largest first. Unassigned pixels (label 0) are left as they are.
    """
    labels = labels.copy()
    while True:
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        small = [(n, i) for i, n in zip(ids.tolist(), counts.tolist()) if n < min_pixels]
        if not small or len(ids) == 1:
            break
        _, i = min(small)
        m = labels == i
        border = np.zeros_like(m)
        border[1:] |= m[:-1]; border[:-1] |= m[1:]; border[:, 1:] |= m[:, :-1]; border[:, :-1] |= m[:, 1:]
        nb = labels[border & ~m]
        big = ids[counts >= min_pixels]              # absorb only into a segment that keeps its own label
        nb = nb[np.isin(nb, big)]
        if nb.size:
            target = int(np.bincount(nb).argmax())
        else:
            target = int(ids[np.argmax(np.where(ids == i, -1, counts))])
        labels[m] = target
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    order = np.argsort(-counts, kind='stable')
    final = np.zeros_like(labels)
    for new, k in enumerate(order):
        final[labels == ids[k]] = new + 1
    return final, [int(counts[k]) for k in order]


def fill_unlabelled(labels, coarse_mask):
    """Every object pixel SAM left without a segment joins the nearest labelled pixel's segment (Tal, 2026-10-07, TRIAL).
    Why: centres are drawn only from labelled pixels, so unlabelled parts of the animal never train -- 28.9% of bear 7107's mask.
    Nearest = Euclidean, in the image. Returns (labels, counts) with the same numbering."""
    from scipy import ndimage
    todo = coarse_mask & (labels == 0)
    if not todo.any() or not (labels > 0).any():
        return labels, [int((labels == i).sum()) for i in range(1, int(labels.max()) + 1)]
    _, (ri, ci) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    out = labels.copy(); out[todo] = labels[ri[todo], ci[todo]]
    return out, [int((out == i).sum()) for i in range(1, int(out.max()) + 1)]


def save_label_png(labels, image, out_path, alpha=0.65):
    """Color-coded overlay of the label map on the source image."""
    n_labels = int(labels.max())
    rng = np.random.RandomState(0)
    palette = rng.rand(n_labels + 1, 3)
    palette[0] = 0.0  # unassigned stays black

    colored = palette[labels]
    blended = image / 255.0 * (1 - alpha) + colored * alpha
    blended[labels == 0] = image[labels == 0] / 255.0 * 0.3  # dim the background

    fig, ax = plt.subplots(figsize=(10, 13))
    ax.imshow(np.clip(blended, 0, 1))
    ax.set_title('%d fine segments' % n_labels)
    ax.axis('off')
    fig.savefig(out_path, bbox_inches='tight', dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./images/guitar_01.png')
    parser.add_argument('--mask_path', type=str, required=True,
                        help='coarse whole-object mask; segments are clipped to it')
    parser.add_argument('--save_dir', type=str, default='./demo/guitar/phase2/')
    parser.add_argument('--sam_dir', type=str, default='./SAM_repo/model_checkpoints/')
    parser.add_argument('--sam_fname', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_model_type', type=str, default='vit_h')
    # Defaults copied from encoder.py:94-101 so this matches the configuration
    # already chosen in this repo rather than inventing a new one.
    parser.add_argument('--points_per_side', type=int, default=32)
    parser.add_argument('--pred_iou_thresh', type=float, default=0.90)
    parser.add_argument('--stability_score_thresh', type=float, default=0.92)
    parser.add_argument('--crop_n_points_downscale_factor', type=int, default=2)
    parser.add_argument('--min_mask_region_area', type=int, default=150)
    parser.add_argument('--min_pixels_in_object', type=int, default=200,
                        help='drop segments with fewer than this many pixels inside the '
                             'coarse mask; they cannot support the loss neighbourhoods')
    parser.add_argument('--fill_unlabelled', action='store_true',
                        help='object pixels with no SAM segment join the nearest segment (Tal, 2026-10-07, trial). Off = every run before')
    parser.add_argument('--merge_below_frac', type=float, default=0.0,
                        help='segments smaller than this share of the object mask (e.g. 0.007) get no label of '
                             'their own: each is absorbed by the neighbouring segment with the longest shared '
                             'border (Tal, 2026-10-06). 0 = off, the behaviour of every run before that date')
    args = parser.parse_args()

    image = cv2.imread(args.img_path)
    if image is None:
        raise FileNotFoundError('could not read image: %s' % args.img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    coarse_mask = load_coarse_mask(args.mask_path)
    if coarse_mask.shape != image.shape[:2]:
        raise ValueError('mask %s does not match image %s'
                         % (coarse_mask.shape, image.shape[:2]))
    print('image %s, coarse mask covers %d px (%.1f%%)'
          % (image.shape[:2], coarse_mask.sum(),
             100.0 * coarse_mask.sum() / coarse_mask.size))

    sam = load_img_seg_model(args.sam_dir, args.sam_fname, args.sam_model_type)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
    )
    print('running SAM automatic mask generation (%d point grid)...'
          % (args.points_per_side ** 2))
    anns = mask_generator.generate(image)
    print('SAM produced %d raw masks' % len(anns))

    labels, counts = build_label_map(anns, coarse_mask, args.min_pixels_in_object)

    if args.merge_below_frac > 0:
        n_before = len(counts)
        labels, counts = merge_small_segments(labels, args.merge_below_frac * int(coarse_mask.sum()))
        print('--merge_below_frac %g: %d segments -> %d' % (args.merge_below_frac, n_before, len(counts)))
    if args.fill_unlabelled:
        n_un = int((coarse_mask & (labels == 0)).sum())
        labels, counts = fill_unlabelled(labels, coarse_mask)
        print('--fill_unlabelled: %d unlabelled object pixels (%.1f%%) joined their nearest segment'
              % (n_un, 100.0 * n_un / coarse_mask.sum()))

    covered = int((labels > 0).sum())
    in_object = int(coarse_mask.sum())
    print('\n%d fine segments kept, covering %d/%d object pixels (%.1f%%)'
          % (len(counts), covered, in_object, 100.0 * covered / max(in_object, 1)))
    for i, n in enumerate(counts):
        print('  segment %2d : %7d px (%.1f%% of object)'
              % (i + 1, n, 100.0 * n / in_object))
    if covered < in_object:
        print('  unassigned : %7d px (%.1f%% of object) -- pixels SAM left uncovered;'
              % (in_object - covered, 100.0 * (in_object - covered) / in_object))
        print('               center pixels sampled there are skipped by the loss.')

    img_name = get_img_name(args.img_path)
    os.makedirs(args.save_dir, exist_ok=True)
    out_pth = os.path.join(args.save_dir, '%s_fine_segments.pth' % img_name)
    out_png = os.path.join(args.save_dir, '%s_fine_segments.png' % img_name)

    torch.save(torch.from_numpy(labels), out_pth)
    save_label_png(labels, image, out_png)
    print('\nsaved %s' % out_pth)
    print('saved %s' % out_png)
