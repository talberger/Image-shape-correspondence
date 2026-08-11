"""Unified relational (distance-proportional) loss for phase-2 encoder training.

`mapping_loss.py` trains on two separate signals -- an attractive mapping_loss
and a repulsive hinge repulsion_loss, combined as `lambda_map * map_loss +
lambda_repel * repel_loss` -- which needed a hand-derived `repel_margin` to
reconcile pixel-space (px) and mesh-space (normalized, bbox diagonal ~1.89)
distances, since a hinge has no notion of scale on its own.

This module replaces that split with one continuous loss: pixels close
together in the photo should map to points close together on the mesh, in
direct proportion, and pixels far apart in the photo should map to points far
apart, also in proportion -- not just "closer than margin" / "farther than
margin". Formalized as the distance-wise loss from Relational Knowledge
Distillation (Park et al., CVPR 2019): normalize pairwise distances in each
space by that space's own mean pairwise distance (removing the need for any
hand-picked scale constant like `repel_margin`), then penalize the difference
between the two normalized distances with a Huber loss, continuous in both
directions rather than a one-sided hinge.

COLLAPSED THE FIRST TIME IT RAN (300 steps, full detail in PROJECT_NOTES.md
2026-08-10): vertex coverage fell to 0.27% -- worse than mapping_loss's own
original collapse (1.68%) -- and the renders showed all 7 seed-152 pixels'
mapped points, including headstock/neck ones, converging onto one spot.
ROOT CAUSE: `mu_3D` (the 3D-space normalizer) was being recomputed from the
LIVE, currently-training encoder's own output every step. Normalizing by a
mean makes the loss invariant to uniformly rescaling every distance by the
same factor -- shrink every pairwise 3D distance together, in proportion, and
every `norm_d_3D` comes out identical, so the loss cannot see or resist that
move. Because the reference was the model's own output rather than anything
external, that free direction had no floor: ordinary gradient noise
accumulated a slow drift down it over ~250 steps (mu_3D 0.41 -> 0.11), which
then ran into the mesh's fixed 4036-vertex granularity and became an outright
collapse (mu_3D -> 0.003, loss rising, not falling, in the last 45 steps).
FIX: `mu_3D_reference` is now measured ONCE, from the untouched phase-1
encoder, before any relational-loss training happens (`measure_mu_3D_reference`
in train_relational_loss.py), and used as a fixed constant for the whole run
-- the same role `repel_margin` played before normalization replaced it,
just measured empirically instead of hand-picked. `relational_loss` still
computes the CURRENT step's actual mean 3D distance too (`mu_3D_live`,
returned for logging only) -- watching that number is exactly what would
have caught the collapse earlier, so it stays in the step log even though it
no longer feeds the loss. NOT YET RE-RUN as of this writing -- see
PROJECT_NOTES.md for what to do next.

`mapping_loss.py` is not modified by this module, and is not imported from it
except where explicitly noted -- this is a separate experiment, run from
train_relational_loss.py, so the existing mapping+repulsion runs stay exactly
reproducible.

Pair sampling: unlike `build_adjacent_neighbors`'s fixed 8-connected ring
(always ~1.2px apart) or far ring (always ~stride px apart), this loss needs a
SPREAD of pixel-space distances per center to have any proportional relation
to learn -- a pair set clustered at one or two near-constant distances only
ever expresses "near" vs. "far" as two buckets, same as the split loss it
replaces. `sample_in_segment_neighbors`/`sample_out_segment_neighbors` below
draw randomly instead, and are resampled fresh every training step (unlike
`sample_center_pixels`'s centers, fixed for the whole run) so the network sees
many different pixel-distance pairings over training rather than overfitting
to one fixed random draw.
"""
import collections

import numpy as np
import torch
import torch.nn.functional as F


SegmentIndex = collections.namedtuple(
    'SegmentIndex', ['mask_coords', 'mask_segids', 'segment_pixel_coords', 'segment_pixel_count'])


def build_segment_index(fine_segments, coarse_mask):
    """Reverse index from segment id -> its member pixel coordinates, plus the
    flat coordinate/segment-id list of every on-object pixel. Built once,
    before the training loop -- everything downstream of it is a per-step
    random draw against these fixed tables.

    Neither `mapping_loss.py` nor `build_fine_segments.py` keeps a reverse
    index like this (every existing caller computes membership ad hoc, over
    just the sampled center set); this loss needs one over the WHOLE image,
    since neighbours are drawn from the whole segment/mask, not the center set.

    Args:
        fine_segments int32 ndarray [H, W], values 0 (background) or 1..N
        coarse_mask   bool  ndarray [H, W]

    Returns a SegmentIndex:
        mask_coords          int64 [M, 2]        every on-object (row, col)
        mask_segids           int64 [M]           fine_segments at each of those
        segment_pixel_coords int64 [N+1, max_size, 2]  per-segment pixel lists,
            padded to the largest segment's size. Row 0 is unused (segment ids
            start at 1). Padding slots repeat that segment's own first pixel --
            never sampled in practice, since every draw is bounded by
            `segment_pixel_count`, so the pad value only has to be *a* valid
            pixel of that segment, not a sentinel.
        segment_pixel_count   int64 [N+1]         true pixel count per segment
    """
    fine_segments = np.asarray(fine_segments)
    coarse_mask = np.asarray(coarse_mask)

    mask_rows, mask_cols = np.nonzero(coarse_mask)
    mask_coords = torch.from_numpy(np.stack([mask_rows, mask_cols], axis=1).astype(np.int64))
    mask_segids = torch.from_numpy(fine_segments[mask_rows, mask_cols].astype(np.int64))

    num_segments = int(fine_segments.max())
    counts = np.bincount(fine_segments[fine_segments > 0], minlength=num_segments + 1)
    max_size = int(counts.max())

    segment_pixel_coords = torch.zeros(num_segments + 1, max_size, 2, dtype=torch.long)
    for s in range(1, num_segments + 1):
        rows, cols = np.nonzero(fine_segments == s)
        n = len(rows)
        segment_pixel_coords[s, :n, 0] = torch.from_numpy(rows.astype(np.int64))
        segment_pixel_coords[s, :n, 1] = torch.from_numpy(cols.astype(np.int64))
        if n < max_size:
            segment_pixel_coords[s, n:, 0] = int(rows[0])
            segment_pixel_coords[s, n:, 1] = int(cols[0])

    return SegmentIndex(mask_coords, mask_segids, segment_pixel_coords,
                        torch.from_numpy(counts.astype(np.int64)))


def sample_in_segment_neighbors(coords, segment_ids, seg_idx, k=8, max_retries=10):
    """For each center pixel, k random pixels from its own fine segment,
    without replacement, excluding the center itself. Resampled fresh on
    every call -- unlike `build_adjacent_neighbors`, nothing here is cached
    across steps.

    Without replacement, no danger of running out: `build_fine_segments.py`
    drops any segment under `min_pixels_in_object` (default 200) after
    painting, so every real segment has far more than k=8 members -- on this
    mesh the smallest of the 16 segments has 330 pixels, the largest 136596.

    THIS FUNCTION USED TO gather each center's entire same-segment pixel list
    (`seg_idx.segment_pixel_coords[segment_ids]`, shape [n, max_size, 2]) and
    rank it by a random priority per slot -- correct, but on this mesh
    max_size=136596, so that gather (plus the matching priority/mask tensors)
    allocated multiple GB of CPU tensors PER TRAINING STEP just to pick 8
    pixels out of them. Measured: ~8.6s/step (vs. train_mapping_loss.py's
    near-instant steps, whose near/far rings are computed once, outside the
    loop, not resampled). Fixed below by sampling INDICES directly in
    [0, count) -- O(n*k), independent of segment size -- and resolving the
    rare self-match/duplicate collision by rejection sampling instead of
    ranking the whole pool.

    Args:
        coords        int64 [n, 2]  center pixel (row, col), CPU
        segment_ids   int64 [n]     each center's fine-segment id, CPU
        seg_idx       SegmentIndex  from build_segment_index
        k             int           neighbours to draw per center
        max_retries   int           vectorized rejection-sampling rounds for
                                     self-matches / within-draw duplicates

    Returns int64 [n, k, 2] (row, col), CPU.
    """
    if coords.device.type != 'cpu':
        raise ValueError('coords must stay on CPU (got %s)' % coords.device)

    n = coords.shape[0]
    count = seg_idx.segment_pixel_count[segment_ids]            # [n]
    seg_rows = segment_ids.unsqueeze(1).expand(-1, k)            # [n, k]

    def draw(idx):
        cand = seg_idx.segment_pixel_coords[seg_rows, idx]                    # [n, k, 2]
        is_self = (cand == coords.unsqueeze(1)).all(-1)                      # [n, k]
        dup = (idx.unsqueeze(2) == idx.unsqueeze(1)) & ~torch.eye(k, dtype=torch.bool)
        return cand, is_self | dup.any(-1)

    idx = (torch.rand(n, k) * count.unsqueeze(1)).long().clamp(max=count.unsqueeze(1) - 1)
    cand, bad = draw(idx)
    for _ in range(max_retries):
        if not bad.any():
            break
        redraw = (torch.rand(n, k) * count.unsqueeze(1)).long().clamp(max=count.unsqueeze(1) - 1)
        idx = torch.where(bad, redraw, idx)
        cand, bad = draw(idx)

    return cand


def sample_out_segment_neighbors(coords, segment_ids, seg_idx, k=8, max_retries=10):
    """For each center pixel, k random pixels NOT in its fine segment (any
    other segment, still on the object), with replacement, resampled fresh on
    every call.

    Segment-EXCLUSIVE, deliberately stricter than `build_adjacent_neighbors`'s
    `require_same_segment=False`, which only checks "on the object at all"
    (segment-agnostic) -- this checks "on the object AND in a different
    segment than the center."

    Implemented as rejection sampling: draw uniformly from the whole
    on-object pixel pool, redraw any pick that lands back in the center's own
    segment, for up to `max_retries` vectorized rounds. The rejection rate per
    round is low (one segment is normally a small fraction of the whole
    mask), so residual same-segment draws after `max_retries` should be rare
    to nonexistent; any that remain are fixed up with a plain per-entry loop
    (expected to touch ~0 entries in practice, so a Python loop here is fine).

    Args:
        coords        int64 [n, 2]  center pixel (row, col), CPU
        segment_ids   int64 [n]     each center's fine-segment id, CPU
        seg_idx       SegmentIndex  from build_segment_index
        k             int           neighbours to draw per center
        max_retries   int           vectorized rejection-sampling rounds

    Returns int64 [n, k, 2] (row, col), CPU.
    """
    if coords.device.type != 'cpu':
        raise ValueError('coords must stay on CPU (got %s)' % coords.device)

    n = coords.shape[0]
    num_mask_pixels = seg_idx.mask_coords.shape[0]

    idx = torch.randint(0, num_mask_pixels, (n, k))
    bad = seg_idx.mask_segids[idx] == segment_ids.unsqueeze(1)

    for _ in range(max_retries):
        if not bad.any():
            break
        redraw = torch.randint(0, num_mask_pixels, (n, k))
        idx = torch.where(bad, redraw, idx)
        bad = seg_idx.mask_segids[idx] == segment_ids.unsqueeze(1)

    if bad.any():
        for i, j in torch.nonzero(bad, as_tuple=False).tolist():
            s = segment_ids[i].item()
            while True:
                candidate = torch.randint(0, num_mask_pixels, (1,)).item()
                if seg_idx.mask_segids[candidate].item() != s:
                    idx[i, j] = candidate
                    break

    return seg_idx.mask_coords[idx]


def relational_loss(center_points, neighbor_points, center_coords, neighbor_coords,
                    mu_3D_reference=None, huber_delta=1.0, eps=1e-12):
    """Penalize pixel-space distance and mesh-space distance for not being
    proportional, in either direction -- close-in-image-but-far-on-mesh and
    far-in-image-but-close-on-mesh are both penalized, continuously, replacing
    `mapping_loss`'s pure attraction and `repulsion_loss`'s margin hinge with
    one term.

    Normalizing each space's distances by a mean pairwise distance is what
    makes pixel-space (px) and mesh-space (normalized, bbox diagonal ~1.89)
    comparable without a hand-picked constant like `repulsion_loss`'s
    `repel_margin`: after dividing, both distributions have mean 1, unitless.

    `mu_3D_reference` picks which of two normalizers `d_3D` is divided by --
    both are real, deliberately-supported modes, not a fixed-vs-legacy split
    (2026-08-11: Tal wants an A/B test between them, isolating which change
    actually fixes the 2026-08-10 collapse):

      - FIXED (a float): the value measured once before training, from the
        untouched phase-1 encoder (train_relational_loss.py's
        `measure_mu_3D_reference`), used as a constant for the whole run.
        This is the 2026-08-10 end-of-day fix.
      - LIVE (`None`, the default): recomputed from THIS STEP's own output
        every call (`mu_3D_live`, detached) -- the ORIGINAL, pre-fix
        behavior. Known to collapse when neighbours are ALSO resampled every
        step (full postmortem in PROJECT_NOTES.md, 2026-08-10): detaching
        only blocks *direct* gradient credit for shrinking the reference, it
        does not stop the model drifting there anyway, because normalizing
        by a mean makes the loss invariant to every pairwise 3D distance
        shrinking together in proportion -- that move is invisible to a loss
        that only ever compares ratios to their OWN batch's average.
        Tal's 2026-08-11 hypothesis: it may have been the resampled-every-
        step neighbours, not the live normalizer per se, that let training
        drift into that blind spot -- worth testing LIVE normalization
        together with FIXED (not resampled) neighbours, since a fixed batch
        re-measured every step is a much narrower, more consistent target to
        drift than a fresh random draw each time. See
        train_relational_loss.py's `--fixed_neighbors`/`--live_mu_3d` flags.

    `mu_2D` stays a live per-step mean regardless of mode (no equivalent
    problem: pixel coordinates are fixed data, not something the training
    loop can drift -- measured stable at ~310px across the entire 300-step
    run that collapsed).

    `d_3D` reuses `repulsion_loss`'s `sqrt(x + eps)` stabilization: it sits on
    the gradient path (center_points/neighbor_points come from the live
    encoder), and plain sqrt's gradient is unbounded as the squared distance
    approaches zero -- the exact NaN-at-step-1 failure `repulsion_loss`'s own
    docstring documents. `d_2D` carries no gradient (coordinates are fixed
    pixel positions) but gets the same eps for symmetry/robustness.

    Args:
        center_points    [n, 3]     y_hat of each center pixel
        neighbor_points  [n, k, 3]  y_hat of each center's k sampled neighbours
        center_coords    [n, 2]     pixel (row, col) of each center
        neighbor_coords  [n, k, 2]  pixel (row, col) of each neighbour
        mu_3D_reference  float or None  FIXED normalizer if a float; LIVE
                                     (this step's own mean) if None -- see above
        huber_delta      float      Huber transition point on the normalized-
                                     distance residual; 1.0 is RKD's own fixed
                                     choice, not just a default to tune away
                                     from without reason
        eps              float      floor inside both sqrt()s

    Returns (loss, mu_2D, mu_3D_live): the scalar loss, the live per-step
    pixel-space mean (float, for logging), and the live per-step MESH-space
    mean (float, for logging always, and for the loss itself when
    `mu_3D_reference` is None). Watching `mu_3D_live` drift away from a FIXED
    `mu_3D_reference` (when one is given) is exactly the signal that exposed
    the original collapse, so it stays in the step log either way.
    """
    device = center_points.device
    center_coords = center_coords.to(device)
    neighbor_coords = neighbor_coords.to(device)

    diff_2d = center_coords.float().unsqueeze(1) - neighbor_coords.float()
    d_2D = (diff_2d.pow(2).sum(-1) + eps).sqrt()                 # [n, k]

    diff_3d = center_points.unsqueeze(1) - neighbor_points
    d_3D = (diff_3d.pow(2).sum(-1) + eps).sqrt()                 # [n, k], on the grad path

    mu_2D = d_2D.mean().detach()
    mu_3D_live = d_3D.mean().detach()

    norm_d_2D = d_2D / mu_2D.clamp(min=1e-8)
    if mu_3D_reference is None:
        norm_d_3D = d_3D / mu_3D_live.clamp(min=1e-8)
    else:
        norm_d_3D = d_3D / max(mu_3D_reference, 1e-8)

    loss = F.huber_loss(norm_d_3D, norm_d_2D, delta=huber_delta, reduction='mean')
    return loss, mu_2D.item(), mu_3D_live.item()
