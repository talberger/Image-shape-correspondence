"""DPC-style mapping loss for image->shape correspondence (phase-2 training).

The idea in one sentence: pixels that are close together in the photo should map
to nearby points on the mesh surface.

Adapted from DPC (arXiv 2110.08636), with source = image pixels and target =
mesh vertices. The paper's equations, and what each becomes here:

  Eq. 1  s_ij = cos(pixel i's DINO feature, vertex j's encoder feature)
  Eq. 2  w_ij = softmax over the k_cc vertices nearest i *in feature space*
                (plain softmax, no temperature)
  Eq. 3  y_i_hat = sum_j w_ij * y_j     <- weighted average of vertex POSITIONS
  Eq. 9  L = mean over (i, l) of v_il * ||y_i_hat - y_l_hat||^2,
                where l ranges over i's k_m nearest neighbours *in 2D pixel
                space*, and v_il = exp(-||x_i - x_l||^2 / alpha)

The two neighbourhoods live in different spaces and behave differently:

  feature-space (k_vertices_per_pixel) -> must be recomputed every step, because
      the vertex features change as the encoder trains.
  pixel-space   (k_pixel_neighbors)    -> computed once, because pixel
      coordinates never change.

Selecting neighbours in feature space but averaging their *positions* is what
makes y_i_hat a soft point on the surface rather than a hard nearest vertex.

Naming note: DPC's k_cc / k_m are opaque, so this module uses
`k_vertices_per_pixel` and `k_pixel_neighbors` instead.
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


def build_pixel_neighbors(coords, segment_ids, k):
    """For each center pixel, its k nearest center pixels in the same segment.

    Computed once -- pixel coordinates do not change during training.

    Neighbours are drawn from the center set itself, matching DPC (its Eq. 9
    neighbourhood N_X lives inside the same sampled point cloud). Restricting to
    one segment is what keeps the loss from pulling together pixels that merely
    happen to be adjacent across a part boundary, e.g. where the neck meets the
    body.

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


def soft_map_to_vertices(pixel_features, vertex_features, vertex_positions, k):
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

    Returns [n, 3] soft-mapped points.
    """
    # One matmul: at n~2048 and V~4036 this is ~33 MB, so unlike the full-image
    # case in analysis/pixel_to_vertex_nn.py no chunking is needed.
    similarity = pixel_features @ F.normalize(vertex_features, dim=-1).t()

    topk_similarity, topk_idx = torch.topk(similarity, k, dim=1)
    weights = F.softmax(topk_similarity, dim=1)  # plain softmax, no temperature

    # Average POSITIONS, not features: the loss measures distance on the
    # surface, which is only meaningful in coordinate space.
    neighbor_positions = vertex_positions[topk_idx]           # [n, k, 3]
    return (weights.unsqueeze(-1) * neighbor_positions).sum(dim=1)


def mapping_loss(soft_points, neighbor_idx, neighbor_valid,
                 normalized_coords=None, weighting='uniform', alpha=8.0):
    """DPC Eq. 9: penalize nearby pixels whose soft-mapped points drift apart.

    `weighting`:
      'uniform'  -- v_il = 1. The default, and effectively what DPC does: at
                    their scale alpha=8 gives v_il ~ 0.9999, so their Gaussian
                    is very nearly uniform anyway. With a small
                    k_pixel_neighbors the k nearest pixels are all at similar
                    distance, which makes the Gaussian close to degenerate.
      'gaussian' -- v_il = exp(-||x_i - x_l||^2 / alpha) on the normalized
                    coordinates. Worth enabling once k_pixel_neighbors is large
                    enough for the distances to actually vary.

    Returns a scalar. Pairs marked invalid contribute nothing to either the
    numerator or the denominator.
    """
    neighbor_points = soft_points[neighbor_idx]                       # [n, k, 3]
    squared_distance = (soft_points.unsqueeze(1) - neighbor_points).pow(2).sum(-1)

    if weighting == 'uniform':
        weight = torch.ones_like(squared_distance)
    elif weighting == 'gaussian':
        if normalized_coords is None:
            raise ValueError("weighting='gaussian' needs normalized_coords")
        pixel_offset = normalized_coords.unsqueeze(1) - normalized_coords[neighbor_idx]
        weight = torch.exp(-pixel_offset.pow(2).sum(-1) / alpha)
    else:
        raise ValueError('unknown weighting: %s' % weighting)

    valid = neighbor_valid.float()
    return (weight * squared_distance * valid).sum() / valid.sum().clamp(min=1.0)


def feature_drift(vertex_features, reference_features):
    """Mean cosine similarity against the phase-1 features. 1.0 = untouched.

    Phase 2 fine-tunes a checkpoint on a single fixed objective with no data
    diversity, so it can wander much further than the step count suggests. This
    turns "change the weights somewhat, but not a lot" into a number to steer by,
    and lets the run be judged after the fact from its checkpoint sweep.
    """
    with torch.no_grad():
        return F.cosine_similarity(vertex_features, reference_features, dim=-1).mean().item()
