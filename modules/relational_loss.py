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
                    mu_3D_reference=None, mu_2D_reference=None, distance_weight=None,
                    mu_3d_scale=1.0, huber_delta=1.0, eps=1e-12):
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

    `mu_2D_reference` is the same idea applied to `d_2D`'s divisor (`None` =
    LIVE per-step mean, a float = FIXED constant). Unlike `mu_3D_reference`,
    `d_2D`'s live mean was never a *drift* risk on its own (pixel coordinates
    are fixed data, not something the training loop can move -- measured
    stable at ~310px across the entire 300-step run that collapsed). The
    reason to fix it anyway (2026-08-11, Tal's proposal): both `mu_2D` and
    `mu_3D_live`/`mu_3D_reference` as computed above are a MEAN OVER THE
    SAMPLED PAIRS THAT STEP -- and center pixels are drawn uniformly over the
    object's pixels (`sample_center_pixels`), so a segment's share of the
    2048 centers is proportional to its pixel count. On this mesh that's
    wildly uneven (measured directly: body segment 66.5% -> ~1362 expected
    centers, neck 16.7% -> ~342, headstock 4.5% -> ~92 out of 2048) -- so
    even the "fixed" `mu_3D_reference` mode is really a fixed value that was
    itself measured mostly from body/neck pairs, and a small segment like the
    headstock has to satisfy a ratio calibrated for a different part of the
    object entirely. A per-step or per-run MEAN can't fix this, no matter how
    it's stabilized, because the imbalance is baked into which pairs get
    sampled in the first place.

    `mu_2D_reference`/`mu_3D_reference` given together as fixed BOUNDING-BOX
    DIAGONALS (`train_relational_loss.py`'s `--bbox_norm`: pixel-space bbox
    diagonal of the coarse mask, mesh-space bbox diagonal of `vertices`) sidestep
    that failure mode entirely -- neither is a statistic over sampled pairs at
    all, just a fixed property of the whole object's geometry, computed once,
    identical regardless of which pixels happen to get drawn as centers or
    neighbours that step. This is the standard convention for reconciling
    image-space and shape-space scale in the literature (e.g. NOCS's
    unit-cube canonicalization, CVPR 2019) -- a fixed geometric normalizer,
    not a live or sampled-mean one.

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
        mu_2D_reference  float or None  same, for d_2D's divisor
        distance_weight  None, 'far', or 'near'  reweight the per-pair Huber
                                     residual by norm_d_2D before averaging,
                                     instead of every sampled pair counting
                                     equally (None, the default, unchanged
                                     behavior). 'far': weight = norm_d_2D --
                                     pairs distant in the image get more say
                                     in the loss. 'near': weight =
                                     1/norm_d_2D (floored) -- the opposite.
                                     Both use norm_d_2D specifically (not
                                     norm_d_3D) because it is the one side of
                                     the pair that is genuinely fixed data --
                                     "near" and "far" are properties of the
                                     photo, not something the loss should get
                                     to redefine by moving d_3D around.
        mu_3d_scale      float      divides norm_d_3D by this AFTER normalization
                                     (1.0, the default, is a no-op). Mathematically
                                     identical to multiplying mu_3D_reference (or
                                     mu_3D_live) by this same value -- it does not
                                     add a new kind of normalization, it rescales
                                     whichever one is already in use. The point:
                                     Tal's 2026-08-12 hypothesis that the model is
                                     converging its mapped points too close together
                                     (the headstock-drift family of problems). A
                                     scale > 1 shrinks norm_d_3D for a GIVEN model
                                     output, which raises the effective target --
                                     to still satisfy norm_d_3D/mu_3d_scale ~=
                                     norm_d_2D, the model has to produce LARGER raw
                                     3D distances between mapped points than it
                                     would without this scale, i.e. push them
                                     further apart. Kept as an explicit separate
                                     knob (not folded into mu_3D_reference itself)
                                     so --bbox_norm's reference stays an honest,
                                     literal bounding-box diagonal, sweepable
                                     independently of this multiplier.
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

    if mu_2D_reference is None:
        norm_d_2D = d_2D / mu_2D.clamp(min=1e-8)
    else:
        norm_d_2D = d_2D / max(mu_2D_reference, 1e-8)
    if mu_3D_reference is None:
        norm_d_3D = d_3D / mu_3D_live.clamp(min=1e-8)
    else:
        norm_d_3D = d_3D / max(mu_3D_reference, 1e-8)
    if mu_3d_scale != 1.0:
        norm_d_3D = norm_d_3D / mu_3d_scale

    if distance_weight is None:
        loss = F.huber_loss(norm_d_3D, norm_d_2D, delta=huber_delta, reduction='mean')
    else:
        per_pair = F.huber_loss(norm_d_3D, norm_d_2D, delta=huber_delta, reduction='none')
        if distance_weight == 'far':
            weight = norm_d_2D.detach()
        elif distance_weight == 'near':
            # Floored, not a raw reciprocal: an unfloored 1/norm_d_2D lets a
            # single near-zero-distance pair (a real possibility -- in- and
            # out-of-segment neighbours can land a few pixels apart) blow up
            # to dominate the whole batch. 0.1 caps the max weight at 10x,
            # the mirror image of 'far' pairs naturally topping out under
            # ~4x (the sampled pool's own max distance / mu_2D).
            weight = 1.0 / norm_d_2D.detach().clamp(min=0.1)
        else:
            raise ValueError("distance_weight must be None, 'far', or 'near', got %r"
                             % (distance_weight,))
        loss = (weight * per_pair).sum() / weight.sum().clamp(min=1e-8)

    return loss, mu_2D.item(), mu_3D_live.item()


# ---------------------------------------------------------------------------
# Geodesic variant (2026-08-13): both d_2D and d_3D become geodesic distances
# instead of Euclidean -- 2D constrained to travel through the mask (any
# segment, not just the center's own), 3D constrained to the mesh surface.
# Full design discussion (why a naive Euclidean->geodesic swap silently kills
# the gradient, the SHIC-inspired fix, the heat-method undershoot/mask-
# disconnection bugs found while precomputing the tables) is in
# PROJECT_NOTES.md, 2026-08-13. Summary of the key point:
#
# `relational_loss()` above computes d_3D directly between two already-
# collapsed points (center_points/neighbor_points, each a softmax-weighted
# AVERAGE of top-K vertex positions) -- valid for Euclidean distance, which is
# a smooth closed-form function of any two points in R^3, on- or off-surface.
# Geodesic distance has no such closed-form: it is a shortest-path quantity
# defined only for points ON the surface, and the collapsed points generally
# are not on it. Hard-snapping each collapsed point to its nearest vertex and
# looking up a precomputed table would "work" numerically, but the snap is a
# discrete/non-differentiable operation -- d_3D would carry ZERO gradient
# back to the encoder, silently (no NaN, no collapse signal, just a dead loss
# term).
#
# Fix: never collapse to a single point at all for the geodesic side. Keep
# each pixel's own top-K vertex indices AND softmax weights (from
# soft_map_to_vertices(..., return_topk=True)) and compute an EXPECTED
# geodesic distance -- a weighted sum over all K*K (center-candidate,
# neighbour-candidate) vertex pairs, looked up in a table that is fixed data
# (computed once, before training, analysis/build_geodesic_tables.py). Every
# individual geo_table entry is a constant; only the weights multiplying them
# are differentiable (softmax over similarity, same as always) -- exactly the
# role vertex_positions plays in the Euclidean path above, and exactly the
# mechanism SHIC's own L_dist uses (gradient through p(x|u,I), not through
# the geodesic values themselves).
# ---------------------------------------------------------------------------


def expected_geodesic_distance_3d(center_topk_idx, center_weights, neighbor_topk_idx,
                                  neighbor_weights, geo_table_3d):
    """Weighted-expectation geodesic distance between a center pixel and each of its k
    neighbours, without ever collapsing either side to a single (generally off-surface,
    non-differentiable-to-snap) point -- see this module's geodesic-variant docstring above.

    d_3D_geo[i, j] = sum_a sum_b center_weights[i, a] * neighbor_weights[i, j, b]
                          * geo_table_3d[center_topk_idx[i, a], neighbor_topk_idx[i, j, b]]

    Args:
        center_topk_idx    long  [n, K]     each center's top-K feature-nearest vertex ids
        center_weights     float [n, K]     softmax weights over those K (from soft_map_to_vertices)
        neighbor_topk_idx  long  [n, k, K]  each neighbour's own top-K vertex ids
        neighbor_weights   float [n, k, K]  softmax weights over those K
        geo_table_3d       float [V, V]     precomputed vertex-to-vertex geodesic distance
                                             (analysis/build_geodesic_tables.py) -- CONSTANT,
                                             no gradient needed/expected through this argument

    Returns [n, k] float, on the gradient path via center_weights/neighbor_weights only.
    """
    n, K = center_topk_idx.shape
    k = neighbor_topk_idx.shape[1]

    # Gather every (a, b) vertex-pair's table entry -- [n, k, K, K] -- via ordinary fancy
    # indexing (both index tensors broadcast to the same shape, elementwise gather). K=10 by
    # convention (matches --k_vertices_per_pixel everywhere else in this project), so this is
    # n*k*K*K = 2048*16*10*10 ~= 3.3M elements (~13MB float32) -- not the "entire V rows"
    # allocation a naive gather-then-select would produce.
    center_idx_exp = center_topk_idx.view(n, 1, K, 1).expand(n, k, K, K)
    neighbor_idx_exp = neighbor_topk_idx.view(n, k, 1, K).expand(n, k, K, K)
    gathered = geo_table_3d[center_idx_exp, neighbor_idx_exp]                 # [n, k, K, K]

    weight_outer = center_weights.view(n, 1, K, 1) * neighbor_weights.view(n, k, 1, K)  # [n, k, K, K]
    return (gathered * weight_outer).sum(dim=(-1, -2))                        # [n, k]


def geodesic_distance_2d_lookup(neighbor_coords, geo_table_2d_center_rows, pixel_to_node):
    """Direct table lookup, no computation, no gradient -- pixel coordinates are fixed data
    (same reasoning `relational_loss()`'s d_2D already relies on: "d_2D carries no gradient,
    coordinates are fixed pixel positions"), so unlike the 3D side there is no differentiability
    problem to solve here at all, only a lookup.

    Args:
        neighbor_coords          long [n, k, 2]  (row, col) of each neighbour, CPU
        geo_table_2d_center_rows float [n, M]    each of THIS BATCH's centers' own row already
                                                  selected out of the full [num_centers, M] table
                                                  (see rkd_spatial_dist_loss -- centers are
                                                  fixed for the whole run, so this is the same
                                                  n rows every call, but selecting it here keeps
                                                  this function a pure lookup, not knowing about
                                                  center identity/ordering itself)
        pixel_to_node             long [H, W]    (row, col) -> compact column index into the
                                                  table, -1 for off-mask pixels (from
                                                  analysis/build_geodesic_tables.py)

    Returns [n, k] float.
    """
    node_idx = pixel_to_node[neighbor_coords[..., 0], neighbor_coords[..., 1]]  # [n, k]
    if (node_idx < 0).any():
        raise ValueError('a sampled neighbour pixel has no entry in the geodesic-safe mask '
                         '(pixel_to_node == -1) -- sample_out_segment_neighbors/'
                         'sample_in_segment_neighbors must be drawing against the SAME '
                         'geodesic_safe_mask the 2D table was built from, not the raw coarse '
                         'mask, or a disconnected speck pixel can get sampled again')
    n, k = node_idx.shape
    return geo_table_2d_center_rows.gather(1, node_idx)


def optimal_mu_3d_scale(norm_d_3D, norm_d_2D, group_ids=None, n_groups=None, min_pairs=32,
                        eps=1e-12, pair_mask=None, fit_space='abs'):
    """Closed-form `mu_3d_scale` (MU-LEARNED direction, 2026-09-14 -- see PROJECT_NOTES.md).

    The loss compares norm_d_3D / s against norm_d_2D, so for a fixed mapping the best s is a
    one-dimensional least-squares fit, not something an optimizer has to discover:
        minimise sum (b/s - a)^2 over s,  with a = norm_d_2D, b = norm_d_3D
        substitute t = 1/s  ->  t* = sum(a*b) / sum(b*b)  ->  s* = sum(b*b) / sum(a*b)
    It is solved exactly at EVERY step, so the features never spend gradient on the global zoom.
    Returned DETACHED: s is a fitted constant for this step, not a parameter to backprop through.

    group_ids (long, broadcastable to the pair shape) + n_groups fit ONE s PER GROUP -- the
    per-segment variant. A group with fewer than `min_pairs` pairs falls back to the global fit
    rather than to a noisy one of its own.

    NOTE this fit is exact for the MSE branch of the Huber. With huber_delta=1.0 and both
    distances divided by a geodesic diameter the residuals are ~1e-2 at most, so the loss is
    plain MSE in practice (measured 2026-09-09, `--dump_residuals`) and the fit is the true
    optimum. If delta is ever lowered into the residual range, this becomes an approximation.
    """
    a = norm_d_2D.detach().flatten()
    b = norm_d_3D.detach().flatten()
    if pair_mask is not None:
        # FIX 1 (Tal, 2026-09-14): a segment's scale must be fitted from pairs that lie INSIDE it.
        # 8 of every centre's 16 neighbours are drawn out-of-segment on purpose, so without this
        # mask half of each segment's pairs are shared neck<->body pairs and every segment ends up
        # measuring the same global structure -- which is exactly what the first per-segment run
        # produced (all 17 scales inside 0.724-0.738).
        m = pair_mask.detach().flatten().bool()
        a, b = a[m], b[m]
        if group_ids is not None:
            group_ids = group_ids.detach().flatten()[m]

    def _fit(aa, bb):
        if fit_space == 'log':   # FIX 3, opt-in: relative error, so a long pair and a short pair
            return torch.exp((bb.clamp_min(eps).log() - aa.clamp_min(eps).log()).mean())
        return (bb * bb).sum() / (aa * bb).sum().clamp_min(eps)

    s_global = _fit(a, b).clamp(0.05, 20.0)
    if group_ids is None:
        return s_global

    g = group_ids.detach().flatten().long()
    if g.numel() != a.numel():
        raise ValueError('group_ids must have one entry per PAIR (got %d for %d pairs)'
                         % (g.numel(), a.numel()))
    ones = torch.ones_like(a)
    cnt = torch.zeros(n_groups, device=a.device, dtype=a.dtype).index_add_(0, g, ones)
    if fit_space == 'log':
        r = (b.clamp_min(eps).log() - a.clamp_min(eps).log())
        acc = torch.zeros(n_groups, device=a.device, dtype=a.dtype).index_add_(0, g, r)
        s = torch.exp(acc / cnt.clamp_min(1.0)).clamp(0.05, 20.0)
    else:
        num = torch.zeros(n_groups, device=a.device, dtype=a.dtype).index_add_(0, g, b * b)
        den = torch.zeros(n_groups, device=a.device, dtype=a.dtype).index_add_(0, g, a * b)
        s = (num / den.clamp_min(eps)).clamp(0.05, 20.0)
    return torch.where(cnt >= min_pairs, s, s_global.expand_as(s))


def per_pair_mu_scale(s_per_group, center_groups, neighbor_groups):
    """FIX 2 (Tal, 2026-09-14): which scale a pair uses once each segment has its own.

    An in-segment pair uses its segment's scale. A CROSS-segment pair spans two scales, and
    charging it to the centre's segment alone is arbitrary -- the same pair would get a different
    answer if read from the other end. It gets the GEOMETRIC mean instead, because a scale is
    multiplicative: half and double must cancel to 1, which only sqrt(s_i*s_j) does.
    """
    sc = s_per_group[center_groups.long()]
    sn = s_per_group[neighbor_groups.long()]
    same = center_groups.long() == neighbor_groups.long()
    return torch.where(same, sc, (sc * sn).clamp_min(1e-12).sqrt())


def rkd_spatial_angle3_loss(center_points, neighbor_points, center_coords, neighbor_coords,
                       min_ray_px=14.0, huber_delta=1.0, eps=1e-12):
    """rkd_spatial_angle3 (Tal, 2026-09-14/15; tagged ANGLE-SPATIAL until then): RKD's angle term in the SPATIAL domain -- the spatial twin
    of rkd_feat_angle3_loss(). For every centre and every ordered pair of distinct neighbours
    (a, b), the angle AT THE CENTRE between the rays to a and to b must match on both sides:
        image : rays between PIXEL coordinates                     (frozen, no grad)
        mesh  : rays between the SOFT-MAPPED 3D points Σ_k w_k·v_k (on the grad path via w)
    **EUCLIDEAN angles on both sides, Tal's call 2026-09-14** -- image-plane rays vs chordal 3D rays,
    NOT geodesic/tangent-space angles, unlike L_spatial's geodesic distances. Any figure or note
    about this term must say "Euclidean angles".

    Scale-invariant by construction (a cosine), so it needs no normaliser and does not interact
    with mu_3d_scale. Known limit: the photo is a PROJECTION, so image angles equal surface angles
    only where the surface faces the camera; expect it to pull the wrong way across depth changes.

    min_ray_px: a ray shorter than this (default 14 = one DINO patch) makes the angle noise --
    both features are nearly identical inside one patch, which is also what makes the smallest
    segments' pairs unreliable. Any triplet with a short ray on the IMAGE side is dropped.

        center_points   [n, 3]      soft-mapped 3D point of each centre
        neighbor_points [n, k, 3]   soft-mapped 3D point of each neighbour
        center_coords   [n, 2]      (row, col) of each centre   (any device/dtype)
        neighbor_coords [n, k, 2]   (row, col) of each neighbour

    Returns (loss, mean_abs_cos_gap, valid_fraction). Row/col order does not matter: swapping two
    axes is a reflection, and a reflection preserves every cosine.
    """
    dev = center_points.device
    cc = center_coords.to(dev).float()
    nc = neighbor_coords.to(dev).float()
    ray_img = nc - cc.unsqueeze(1)                                            # [n, k, 2]
    ray_msh = neighbor_points - center_points.unsqueeze(1)                    # [n, k, 3]
    long_enough = ray_img.norm(dim=-1) >= min_ray_px                          # [n, k]

    e_img = F.normalize(ray_img, dim=-1, eps=eps)
    e_msh = F.normalize(ray_msh, dim=-1, eps=eps)
    cos_img = torch.bmm(e_img, e_img.transpose(1, 2))                         # [n, k, k]
    cos_msh = torch.bmm(e_msh, e_msh.transpose(1, 2))

    k = ray_img.shape[1]
    off_diag = ~torch.eye(k, dtype=torch.bool, device=dev)
    valid = off_diag.unsqueeze(0) & long_enough.unsqueeze(2) & long_enough.unsqueeze(1)  # [n, k, k]
    if not valid.any():
        zero = center_points.sum() * 0.0
        return zero, float('nan'), 0.0
    loss = F.huber_loss(cos_msh[valid], cos_img[valid], delta=huber_delta, reduction='mean')
    gap = (cos_msh[valid] - cos_img[valid]).abs().mean().item()
    return loss, gap, valid.float().sum().item() / (valid.shape[0] * k * (k - 1))


def rkd_spatial_dist_loss(center_topk_idx, center_weights, neighbor_topk_idx, neighbor_weights,
                             neighbor_coords, geo_table_3d, geo_table_2d_center_rows, pixel_to_node,
                             mu_3D_reference, mu_2D_reference, mu_3d_scale=1.0, huber_delta=1.0,
                             return_residuals=False, mu_groups=None, mu_n_groups=None,
                             return_mu_scale=False, mu_neighbor_groups=None, mu_fit_space='abs'):
    """Geodesic counterpart of relational_loss() above -- same Huber-on-normalized-ratios
    structure, same mu_3d_scale/huber_delta roles, but both distances are geodesic (see this
    module's geodesic-variant docstring) instead of Euclidean, and normalization is against
    geodesic diameters (max pairwise geodesic distance) rather than Euclidean bbox diagonals --
    Tal's explicit call, matching SHIC's own convention of normalizing by the max geodesic
    distance on the mesh, given the distances themselves are now geodesic too. Unlike
    relational_loss(), `mu_3D_reference`/`mu_2D_reference` are REQUIRED here (no LIVE/None mode):
    this variant was only ever designed and run with the fixed-geodesic-diameter normalization
    Tal chose, so there is no verified LIVE behavior to silently fall back to.

    Args:
        center_topk_idx, center_weights, neighbor_topk_idx, neighbor_weights: see
            expected_geodesic_distance_3d
        neighbor_coords            long [n, k, 2]  see geodesic_distance_2d_lookup
        geo_table_3d                float [V, V]
        geo_table_2d_center_rows    float [n, M]    see geodesic_distance_2d_lookup
        pixel_to_node               long [H, W]
        mu_3D_reference, mu_2D_reference  float, required -- geodesic diameters
        mu_3d_scale, huber_delta   see relational_loss()

    Returns (loss, mu_2D_live, mu_3D_live): the scalar loss, plus this step's own actual mean
    geodesic distances in each space (logging only, matching relational_loss()'s mu_3D_live --
    watching these drift is exactly the signal that exposed the original Euclidean-mode collapse).
    """
    if mu_3D_reference is None or mu_2D_reference is None:
        raise ValueError('rkd_spatial_dist_loss requires both mu_3D_reference and '
                         'mu_2D_reference (fixed geodesic diameters) -- there is no LIVE mode '
                         'here, unlike relational_loss()')

    d_3D = expected_geodesic_distance_3d(center_topk_idx, center_weights,
                                         neighbor_topk_idx, neighbor_weights, geo_table_3d)
    # 2D lookup stays on CPU throughout (neighbor_coords/geo_table_2d_center_rows/pixel_to_node
    # are all CPU -- the table is 1.69GB, no reason to duplicate it onto the GPU just to index
    # it, same reasoning relational_loss()'s own d_2D CPU-first handling already follows) --
    # only the small [n, k] RESULT crosses to the encoder's device, to combine with d_3D below.
    d_2D = geodesic_distance_2d_lookup(
        neighbor_coords, geo_table_2d_center_rows, pixel_to_node).to(center_weights.device)

    mu_2D_live = d_2D.mean().detach()
    mu_3D_live = d_3D.mean().detach()

    norm_d_2D = d_2D / max(mu_2D_reference, 1e-8)
    norm_d_3D = d_3D / max(mu_3D_reference, 1e-8)
    # mu_3d_scale (2026-09-14) is now one of three things, see PROJECT_NOTES.md `MU-LEARNED`:
    #   float          FIXED, the original behaviour (1.0 = no-op)
    #   'optimal'      CLOSED FORM, refitted every step by optimal_mu_3d_scale(), detached
    #   Tensor         LEARNED, an nn.Parameter-derived scale the caller backprops through;
    #                  scalar, or per-GROUP with mu_groups giving each pair's group
    # mu_groups may be given per CENTRE ([n]) -- every pair of a centre belongs to that centre's
    # segment -- so expand it to one entry per PAIR before any grouped arithmetic.
    pair_groups = mu_groups
    if pair_groups is not None and pair_groups.dim() == 1:
        pair_groups = pair_groups.to(norm_d_3D.device).unsqueeze(1).expand_as(norm_d_3D)
    # mu_neighbor_groups [n, k]: each NEIGHBOUR pixel's segment, so an in-segment pair can be told
    # from a cross-segment one (fixes 1 and 2, 2026-09-14)
    nbr_groups = mu_neighbor_groups.to(norm_d_3D.device) if mu_neighbor_groups is not None else None
    in_segment = None if nbr_groups is None else (pair_groups.long() == nbr_groups.long())

    mu_scale_used = mu_3d_scale
    if isinstance(mu_3d_scale, str):
        if mu_3d_scale != 'optimal':
            raise ValueError("mu_3d_scale as a string must be 'optimal', got %r" % mu_3d_scale)
        mu_scale_used = optimal_mu_3d_scale(norm_d_3D, norm_d_2D,
                                            group_ids=pair_groups, n_groups=mu_n_groups,
                                            pair_mask=in_segment, fit_space=mu_fit_space)
    if torch.is_tensor(mu_scale_used):
        s = mu_scale_used.to(norm_d_3D.device)
        if s.numel() == 1:                       # one global scale
            s = s.reshape(())
        elif s.dim() == 1:                       # one per GROUP -> expand to one per PAIR
            if mu_groups is None:
                raise ValueError('a per-group mu_3d_scale needs mu_groups')
            if nbr_groups is not None:           # FIX 2: cross-segment pairs get the geometric mean
                s = per_pair_mu_scale(s, pair_groups.to(s.device), nbr_groups)
            else:
                s = s[mu_groups.to(s.device).long()]
            if s.dim() == 1:                     # mu_groups was per CENTRE: [n] -> [n, 1], broadcasts
                s = s.unsqueeze(1)
        if s.dim() > 1 and s.shape not in (norm_d_3D.shape, (norm_d_3D.shape[0], 1)):
            raise ValueError('mu_3d_scale expanded to %s, expected %s or [n, 1]'
                             % (tuple(s.shape), tuple(norm_d_3D.shape)))
        norm_d_3D = norm_d_3D / s
    elif mu_scale_used != 1.0:
        norm_d_3D = norm_d_3D / mu_scale_used

    loss = F.huber_loss(norm_d_3D, norm_d_2D, delta=huber_delta, reduction='mean')

    if return_mu_scale:
        return loss, mu_2D_live.item(), mu_3D_live.item(), mu_scale_used

    if return_residuals:
        # Diagnostic path only (--dump_residuals): the per-pair signed residual the Huber sees.
        # delta is a transition point ON THIS quantity, so its distribution is what decides
        # whether delta is doing anything at all -- see train_relational_loss.py's dump block.
        return loss, mu_2D_live.item(), mu_3D_live.item(), (norm_d_3D - norm_d_2D).detach()

    return loss, mu_2D_live.item(), mu_3D_live.item()


# CANONICAL NAMES (Tal, 2026-09-15): rkd_spatial_dist (the geodesic distance term, "L_spatial" /
# "relational" in older notes) and rkd_spatial_angle3. Old names stay importable.
relational_loss_geodesic = rkd_spatial_dist_loss
spatial_angle_loss = rkd_spatial_angle3_loss
