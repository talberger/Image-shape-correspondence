"""DPC-style mapping loss for image->shape correspondence (phase-2 training).

The idea in one sentence: pixels that are close together in the photo should map
to nearby points on the mesh surface.

Adapted from DPC (arXiv 2110.08636), with source = image pixels and target =
mesh vertices. The paper's equations, and what each becomes here:

  Eq. 1  s_ij = cos(pixel i's DINO feature, vertex j's encoder feature)
  Eq. 2  w_ij = softmax over the k_cc vertices nearest i *in feature space*
                (plain softmax, no temperature)
  Eq. 3  y_i_hat = sum_j w_ij * y_j     <- weighted average of vertex POSITIONS
  Eq. 9  L = mean over (i, l) of ||y_i_hat - y_l_hat||^2, where l ranges over
                the 8 pixels immediately surrounding i in the image

The two neighbourhoods live in different spaces and behave differently:

  feature-space (k_vertices_per_pixel) -> must be recomputed every step, because
      the vertex features change as the encoder trains.
  pixel-space   (the 8-pixel ring)     -> computed once, because pixel
      coordinates never change.

Selecting neighbours in feature space but averaging their *positions* is what
makes y_i_hat a soft point on the surface rather than a hard nearest vertex.

Naming note: DPC's k_cc is opaque, so this module uses `k_vertices_per_pixel`.

HISTORY -- read this before changing `build_adjacent_neighbors`. Until
2026-08-02 the pixel neighbours were the k=7 nearest *among the sampled center
pixels* (`build_pixel_neighbors`, kept below for the analysis/measure_* scripts),
which put them a mean of 11.4 px apart instead of the intended ~1.2 px. That is
not what the loss was meant to express, and it invalidates the phase-2 results
recorded in FINE_TUNING_PLAN.md. The justification given at the time -- "a pixel
that wasn't sampled has no y_hat, so it cannot be a neighbour" -- is wrong:
y_hat is a function of a pixel's DINO feature, and features exist for every
pixel in the image. Nothing stops us computing y_hat for the neighbours too, and
`build_adjacent_neighbors` + the [n*(1+k)] batching in train_mapping_loss.py is
exactly that.
"""
import numpy as np
import torch
import torch.nn.functional as F


def sample_center_pixels(coarse_mask, fine_segments, num_pixels, seed=0):
    """Pick a fixed set of center pixels from the object, once.

    Fixed rather than resampled per step: with a fixed set the loss curve moves
    only because the encoder improved, which is the only training signal we have
    here (one image, one shape, no validation split). It also matches DPC, whose
    point clouds are downsampled once at data-prep time.

    Only pixels that carry a fine-segment label are eligible -- a pixel with no
    segment has no same-part neighbours and so cannot contribute to the loss.

    Returns (coords, normalized_coords, segment_ids):
        coords            int64  [n, 2]  (row, col) in image pixels
        normalized_coords float  [n, 2]  coords scaled by the object bounding-box
                                         diagonal, so distances are ~unit scale
                                         like DPC's point clouds
        segment_ids       int64  [n]
    """
    if coarse_mask.shape != fine_segments.shape:
        raise ValueError('mask %s and segments %s disagree'
                         % (tuple(coarse_mask.shape), tuple(fine_segments.shape)))

    eligible = coarse_mask & (fine_segments > 0)
    rows, cols = np.nonzero(eligible)
    if len(rows) < num_pixels:
        raise ValueError('asked for %d center pixels but only %d are eligible'
                         % (num_pixels, len(rows)))

    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(rows), size=num_pixels, replace=False)
    coords = np.stack([rows[chosen], cols[chosen]], axis=1)

    # Scale by the object's bbox diagonal, not the image's: it keeps distances
    # comparable across images and puts them in the same ~unit range DPC's
    # alpha=8 was calibrated for. In raw pixel units, exp(-d^2/8) underflows to
    # zero at any realistic spacing and the loss loses its gradient entirely.
    obj_rows, obj_cols = np.nonzero(coarse_mask)
    height = obj_rows.max() - obj_rows.min() + 1
    width = obj_cols.max() - obj_cols.min() + 1
    diagonal = float(np.hypot(height, width))

    return (torch.from_numpy(coords).long(),
            torch.from_numpy(coords.astype(np.float32) / diagonal),
            torch.from_numpy(fine_segments[coords[:, 0], coords[:, 1]].astype(np.int64)))


def sample_grid_center_pixels(coarse_mask, fine_segments, stride):
    """Deterministic, evenly-spaced CANDIDATE POOL of center pixels: every eligible
    (on-object, fine-segment-labeled) pixel at (row, col) with row % stride == 0 and
    col % stride == 0 -- as opposed to sample_center_pixels's single random draw of
    exactly num_pixels centers, fixed for the whole run.

    Built for --resample_centers (phase2/train_relational_loss.py, 2026-08-18): a
    per-step random SUBSET of this pool becomes that step's centers, so the network
    sees a different center set every step instead of the same fixed 2048 forever.
    The pool itself must stay identical between analysis/build_geodesic_tables.py's
    --center_mode grid run (which needs a source for every candidate pixel) and
    training (which needs the exact same candidates to index into that table) -- both
    call this same function with the same (mask, fine_segments, stride), no seed
    involved since the grid is a deterministic function of its inputs.

    Returns (coords, normalized_coords, segment_ids), same shapes/convention as
    sample_center_pixels -- just typically far more of them (e.g. stride=3 on this
    mesh's mask gives ~23.6k candidates vs. sample_center_pixels's usual 2048).
    """
    if coarse_mask.shape != fine_segments.shape:
        raise ValueError('mask %s and segments %s disagree'
                         % (tuple(coarse_mask.shape), tuple(fine_segments.shape)))
    if stride < 1:
        raise ValueError('stride must be >= 1, got %r' % (stride,))

    eligible = coarse_mask & (fine_segments > 0)
    grid_mask = np.zeros_like(eligible)
    grid_mask[::stride, ::stride] = True
    eligible_grid = eligible & grid_mask

    rows, cols = np.nonzero(eligible_grid)
    if len(rows) == 0:
        raise ValueError('stride=%d leaves zero eligible candidate centers' % stride)
    coords = np.stack([rows, cols], axis=1)

    obj_rows, obj_cols = np.nonzero(coarse_mask)
    height = obj_rows.max() - obj_rows.min() + 1
    width = obj_cols.max() - obj_cols.min() + 1
    diagonal = float(np.hypot(height, width))

    return (torch.from_numpy(coords).long(),
            torch.from_numpy(coords.astype(np.float32) / diagonal),
            torch.from_numpy(fine_segments[coords[:, 0], coords[:, 1]].astype(np.int64)))


def build_pixel_neighbors(coords, segment_ids, k):
    """LEGACY -- not used by training. For each center pixel, its k nearest
    *center* pixels in the same segment.

    Superseded by `build_adjacent_neighbors`: drawing neighbours from the center
    set puts them ~11.4 px apart, which is not the intended neighbourhood (see
    the module docstring). Kept because analysis/measure_neighbor_stats.py and
    analysis/measure_neighbor_features.py import it to characterise what the old
    phase-2 run actually trained on.

    Computed once -- pixel coordinates do not change during training.

    Restricting to one segment is what keeps the loss from pulling together
    pixels that merely happen to be adjacent across a part boundary, e.g. where
    the neck meets the body.

    Segments smaller than k+1 members yield fewer than k neighbours; those slots
    are marked invalid rather than padded with far-away pixels, so a small
    segment contributes only the pairs it genuinely has.

    Returns (neighbor_idx, neighbor_valid):
        neighbor_idx   int64 [n, k]  indices into the center set
        neighbor_valid bool  [n, k]
    """
    num_pixels = coords.shape[0]
    device = coords.device
    neighbor_idx = torch.zeros(num_pixels, k, dtype=torch.long, device=device)
    neighbor_valid = torch.zeros(num_pixels, k, dtype=torch.bool, device=device)
    coords_f = coords.float()

    for segment in torch.unique(segment_ids):
        members = torch.nonzero(segment_ids == segment, as_tuple=True)[0]
        if len(members) < 2:
            continue  # nothing to pair with

        distances = torch.cdist(coords_f[members], coords_f[members])
        distances.fill_diagonal_(float('inf'))  # a pixel is not its own neighbour

        available = min(k, len(members) - 1)
        _, local_idx = torch.topk(distances, available, dim=1, largest=False)

        neighbor_idx[members, :available] = members[local_idx]
        neighbor_valid[members, :available] = True

    return neighbor_idx, neighbor_valid


# The 8-connected ring, in reading order. Fixed and documented because the
# column order of `neighbor_valid` is only interpretable against it.
NEIGHBOR_OFFSETS = ((-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 1),
                    (1, -1), (1, 0), (1, 1))


def build_adjacent_neighbors(coords, segment_ids, fine_segments, stride=1,
                              require_same_segment=True):
    """For each center pixel, the 8 pixels surrounding it in the image.

    This is the intended pixel neighbourhood: `stride=1` gives the immediately
    adjacent pixels, a mean of (4 + 4*sqrt(2))/8 = 1.2071 px away. `stride`
    scales the ring outward -- e.g. stride=100 is the same 8 directions, just
    100px away instead of 1 -- and is how the repulsive far ring reuses this
    same function (`require_same_segment=False`, large stride) instead of
    duplicating it.

    Unlike `build_pixel_neighbors` these neighbours are arbitrary image pixels,
    not members of the center set, so the caller must compute a y_hat for them
    too -- see the [n*(1+k)] batching in train_mapping_loss.py.

    `require_same_segment` (default True, the near-ring / attractive case): a
    neighbour is valid only if it lies in the *same fine segment* as its
    center, which keeps the loss from pulling together pixels that merely happen
    to be adjacent across a part boundary (e.g. where the neck meets the body).
    Since build_fine_segments.py ANDs every SAM mask with the coarse object mask
    before painting, `fine_segments > 0` is a strict subset of that mask, so the
    same-segment test alone already guarantees the neighbour is on the object.

    `require_same_segment=False` (the far-ring / repulsive case): segment
    membership isn't relevant for "is this pair far apart" -- a far neighbour
    in a different segment, or the same large segment, is equally valid. A
    neighbour is then valid as long as it's on the object at all
    (`fine_segments > 0`, per the same coarse-mask-subset relationship above).

    Everything stays on CPU: the coordinates are used to index the 3.6 GB pixel
    feature file, and torch rejects a mix of CPU and CUDA index tensors.

    Returns (neighbor_coords, neighbor_valid):
        neighbor_coords int64 [n, 8, 2]  (row, col), clamped into the image
        neighbor_valid  bool  [n, 8]
    """
    if coords.device.type != 'cpu':
        raise ValueError('coords must stay on CPU -- they index the pixel '
                         'feature file (got %s)' % coords.device)

    segments = torch.as_tensor(np.asarray(fine_segments)).long()
    height, width = segments.shape

    offsets = torch.tensor(NEIGHBOR_OFFSETS, dtype=torch.long) * stride  # [8, 2]
    unclamped = coords.unsqueeze(1) + offsets.unsqueeze(0)               # [n, 8, 2]

    # Out-of-bounds must be decided on the UNCLAMPED coordinates. Clamping first
    # would let an off-image slot snap to an edge pixel that happens to share the
    # center's segment, and be counted as an adjacent neighbour it is nowhere
    # near. Unreachable at stride=1 for this image, but not in general.
    out_of_bounds = ((unclamped[..., 0] < 0) | (unclamped[..., 0] >= height) |
                     (unclamped[..., 1] < 0) | (unclamped[..., 1] >= width))

    neighbor_coords = torch.stack(
        [unclamped[..., 0].clamp(0, height - 1),
         unclamped[..., 1].clamp(0, width - 1)], dim=-1)

    segment_at_neighbor = segments[neighbor_coords[..., 0], neighbor_coords[..., 1]]
    if require_same_segment:
        on_object = segment_at_neighbor == segment_ids.unsqueeze(1)
    else:
        on_object = segment_at_neighbor > 0
    neighbor_valid = ~out_of_bounds & on_object

    return neighbor_coords, neighbor_valid


def soft_map_to_vertices(pixel_features, vertex_features, vertex_positions, k, return_topk=False,
                          temperature=1.0):
    """DPC Eq. 1-3: map each pixel to a soft point on the mesh surface.

    Differentiable end to end -- no detach anywhere, so gradients reach the
    encoder through `vertex_features`. `topk` selection itself is not
    differentiable (the *choice* of which k vertices gets no gradient), which is
    fine and is what DPC does: the gradient flows through the selected
    similarity values, not through the selection.

    Args:
        pixel_features   [n, F]  frozen DINO features, L2-normalized and detached
        vertex_features  [V, F]  live encoder output
        vertex_positions [V, 3]
        k                how many vertices each pixel is mapped onto
        return_topk      if True, ALSO return (topk_idx, weights) -- see below.
                          Default False: unchanged return value/behavior, for every
                          existing caller (byte-identical, no new allocation on that path).
        temperature      softmax temperature T: weights = softmax(topk_similarity / T).
                          Default 1.0 == the original plain softmax, byte-identical for
                          every existing caller (the divide is skipped entirely at 1.0).
                          Added 2026-08-25 because the plain softmax was MEASURED to be
                          very nearly a no-op: cosine similarities within a pixel's top-k
                          differ by only ~0.03 (median, over the 100-point eval set), and
                          softmax concentrates as exp(difference), so it needs a gap of
                          O(1) to do anything. Result was effective-k 9.999 out of 10
                          across all 212,549 masked pixels -- i.e. a near-uniform average
                          over the k candidates, with no ability to express confidence.
                          T~0.01-0.02 is the range where the weights actually concentrate
                          on this project's similarity scale.

    Returns [n, 3] soft-mapped points (return_topk=False, the original behavior), or
    (points [n, 3], topk_idx [n, k] long, weights [n, k] float) if return_topk=True --
    for callers that need the underlying per-vertex weights themselves rather than
    the already-collapsed position (e.g. modules/relational_loss.py's geodesic distance,
    which needs a weighted expectation over PAIRS of actual vertices, not the distance
    between two already-averaged, generally off-surface points -- see that module's
    docstring for why the collapsed point isn't usable for a geodesic distance).
    """
    # One matmul. Training calls this with n = 2048 centers + 2048*8 neighbours
    # = 18432 rows, so at V~4036 the similarity is ~297 MB -- the dominant
    # allocation, but still far from needing the chunking that the full-image
    # case in analysis/pixel_to_vertex_nn.py does. It does not survive into the
    # backward pass: topk saves only its indices, softmax saves its [n, k]
    # output, and the matmul saves its two (small) inputs. Measured peak was
    # 757 MB for the whole forward+backward.
    similarity = pixel_features @ F.normalize(vertex_features, dim=-1).t()

    topk_similarity, topk_idx = torch.topk(similarity, k, dim=1)
    # `is not 1.0`-style guard by value: at the default the divide is skipped so the
    # existing training/analysis path is bit-for-bit what it was before temperature existed
    if temperature != 1.0:
        topk_similarity = topk_similarity / temperature
    weights = F.softmax(topk_similarity, dim=1)

    # Average POSITIONS, not features: the loss measures distance on the
    # surface, which is only meaningful in coordinate space.
    neighbor_positions = vertex_positions[topk_idx]           # [n, k, 3]
    points = (weights.unsqueeze(-1) * neighbor_positions).sum(dim=1)
    if return_topk:
        return points, topk_idx, weights
    return points


def mapping_loss(center_points, neighbor_points, neighbor_valid):
    """DPC Eq. 9: penalize nearby pixels whose soft-mapped points drift apart.

    Args:
        center_points   [n, 3]     y_hat of each center pixel
        neighbor_points [n, k, 3]  y_hat of each center's surrounding pixels
        neighbor_valid  [n, k]

    Returns a scalar. Invalid pairs contribute to neither the numerator nor the
    denominator. The denominator is global rather than per-row on purpose: a
    center with no valid neighbours at all then drops out cleanly instead of
    producing 0/0. (At stride=1 on the guitar, 2 of 2048 centers are in that
    state.)

    DPC's Gaussian weight v_il = exp(-||x_i - x_l||^2 / alpha) is gone, not
    forgotten. On normalized coordinates (divided by the ~1231 px bbox diagonal)
    a 1 px offset gives d^2/alpha = 8.2e-8, so exp(-d^2/alpha) is 1-2 ulp from
    1.0 in fp32 and is *bit-identical* for the 4 axial and 4 diagonal
    neighbours. At this scale the Gaussian is exactly uniform; keeping the flag
    would only suggest a choice that does not exist.

    WHAT THIS ACTUALLY OPTIMIZES, measured on the phase-1 encoder at stride=1:
    the median pair distance is 5.9e-06 and 31% of pairs already resolve to the
    *identical* ordered top-10 vertex set, i.e. the smoothness objective is
    ~99.99% satisfied before training starts, and 8% of pairs carry 90% of the
    loss. The residual gradient therefore lives almost entirely in the
    discontinuity of the topk in `soft_map_to_vertices` -- in practice this
    penalizes top-k boundary flips between adjacent pixels rather than
    smoothness per se. That is expected (adjacent pixels share a DINO patch by
    construction), but it is what any result here has to be read against.
    """
    squared_distance = (center_points.unsqueeze(1) - neighbor_points).pow(2).sum(-1)
    valid = neighbor_valid.float()
    return (squared_distance * valid).sum() / valid.sum().clamp(min=1.0)


def repulsion_loss(center_points, far_points, far_valid, margin):
    """Repulsive complement to `mapping_loss()`: penalize a pixel and a *far*
    pixel (see `build_adjacent_neighbors(..., require_same_segment=False)`)
    whose soft-mapped points end up closer than `margin` on the mesh surface.

    `L_map` alone is purely attractive -- nothing in it ever says "keep these
    apart" -- which is the documented mechanism behind the coverage collapse
    (every pixel's global minimum is one point, regardless of neighbours).
    This is the missing half: same shape of call as `mapping_loss()` (center
    vs. its ring, validity mask, global denominator), just the near ring
    swapped for the far one and squared-distance swapped for a hinge.

    Flat penalty, not scaled by how far apart the pair is in the image (that
    is a possible refinement, not implemented here): every far pair inside
    the margin is penalized equally.

    Args:
        center_points [n, 3]     y_hat of each center pixel (same tensor
                                  `mapping_loss` is called with)
        far_points    [n, 8, 3]  y_hat of each center's far-ring neighbours
        far_valid     [n, 8]     from build_adjacent_neighbors(require_same_segment=False)
        margin        scalar     pairs closer than this are penalized

    Returns a scalar. Same "global denominator, invalid pairs contribute to
    neither sum" convention as `mapping_loss()`, for the same reason (a
    center near the object's edge may have few or no valid far neighbours,
    per the coverage table in the plan this was sized against).

    `mapping_loss()` gets away with plain squared distance (no sqrt) because
    it never needs actual distance, only its gradient toward zero. This loss
    needs true (non-squared) distance to compare against `margin`, and
    sqrt(x)'s gradient is 1/(2*sqrt(x)) -- unbounded as x -> 0. Any far pair
    that starts (or lands, mid-training) at ~zero distance would then blow up
    to a NaN gradient on the very next backward pass. Confirmed by the first
    smoke test run: grad norm was already NaN at step 1. Fixed with a small
    epsilon inside the sqrt, the standard fix for Euclidean-distance losses
    (same trick torch's own TripletMarginLoss uses internally).
    """
    eps = 1e-12
    squared_distance = (center_points.unsqueeze(1) - far_points).pow(2).sum(-1)
    distance = (squared_distance + eps).sqrt()
    penalty = F.relu(margin - distance).pow(2)
    valid = far_valid.float()
    return (penalty * valid).sum() / valid.sum().clamp(min=1.0)


def feature_drift(vertex_features, reference_features):
    """Mean cosine similarity against the phase-1 features. 1.0 = untouched.

    Phase 2 fine-tunes a checkpoint on a single fixed objective with no data
    diversity, so it can wander much further than the step count suggests. This
    turns "change the weights somewhat, but not a lot" into a number to steer by,
    and lets the run be judged after the fact from its checkpoint sweep.
    """
    with torch.no_grad():
        return F.cosine_similarity(vertex_features, reference_features, dim=-1).mean().item()
