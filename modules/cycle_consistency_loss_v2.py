"""
Cycle-consistency loss v2 (2026-09-10): SHIC's L_cyc with CSE's temperature, meant to ride ON TOP of
the ZS anchor (modules/anchor_loss.py). The two older versions (cycle_consistency_loss.py hard-snap,
cycle_consistency_loss_soft.py) are kept untouched for reproducibility.

What changed vs cycle_consistency_loss_soft.py, and where each change comes from:

  1. pixel -> vertex over ALL V vertices, not the top-k  (SHIC Eq. 3, CSE cycle_pix2shape.py)
     A vertex outside a pixel's top-k gets no gradient, so a wrong top-k can never be corrected.
  2. a TEMPERATURE on BOTH legs  (CSE: TEMPERATURE_PIXEL_TO_VERTEX = TEMPERATURE_VERTEX_TO_PIX = 0.05)
     Our features are L2-normalized, so similarities are cosines whose top candidates differ by ~0.03.
     softmax at T=1 turns that into nearly equal weights (measured: q_ZS over 3886 vertices at T=1 has
     3868 effective vertices), the round trip spreads over the whole object whatever the map is, and
     the loss carries almost no signal. SHIC does not need an explicit T because its 16-d embeddings
     are UNnormalized -- their norm is a learned sharpness. CSE normalizes and therefore divides by T.
  3. vertex -> pixel is a softmax over the SAME sampled pixel set as the forward leg -- exactly what both
     references do (SHIC: "we randomly select 100 foreground points from each image"; CSE: 100 sampled
     foreground pixels, softmax over them). Here that set is the run's 2048 centers, whose pairwise 2D
     geodesic distances are already tabulated.

Kept from ours: 2D GEODESIC distance normalized by the mask's geodesic diameter (SHIC uses Euclidean
pixel distance, CSE squared Euclidean), and SHIC's aggregation, the expected round-trip distance
    L_cyc = mean_u sum_v d(u, v) p(v|u),   p(v|u) = sum_x p(v|x) p(x|u).
"""
import torch
import torch.nn.functional as F


def cycle_consistency_loss_v2(center_pixel_features, vertex_features, center_geo_dist, temperature):
    """
    Args:
        center_pixel_features  [n, F]  frozen photo features of the centers, already L2-normalized
        vertex_features        [V, F]  live encoder output
        center_geo_dist        [n, n]  2D geodesic distance between centers / 2D geodesic diameter
        temperature            T for both legs

    Returns (loss, loss value, num_referenced_vertices) -- same signature as the older versions, so the
    training loop's logging is unchanged. num_referenced_vertices = distinct argmax vertices, diagnostic only.
    """
    sim = center_pixel_features @ F.normalize(vertex_features, dim=-1).t()    # [n, V]
    p_x_given_u = F.softmax(sim / temperature, dim=1)                          # [n, V]  pixel -> vertex
    p_v_given_x = F.softmax(sim.t() / temperature, dim=1)                      # [V, n]  vertex -> pixel
    p_v_given_u = p_x_given_u @ p_v_given_x                                    # [n, n]  round trip

    loss = (p_v_given_u * center_geo_dist).sum(dim=1).mean()

    with torch.no_grad():
        num_referenced_vertices = int(torch.unique(sim.argmax(dim=1)).numel())
    return loss, loss.detach().item(), num_referenced_vertices


def _return_leg_expected_distance(vertex_chunk, all_pixel_features, center_to_all_geo_dist, temperature):
    """For a chunk of vertices x: E_{v ~ p(v|x)}[d(u, v)] for every center u.  Returns [n, c].
    p(v|x) is a softmax over ALL mask pixels -- the [c, M] matrices live only inside this function,
    which is gradient-checkpointed, so they are never stored for the whole mesh at once."""
    p_v_given_x = F.softmax(vertex_chunk @ all_pixel_features.t() / temperature, dim=1)   # [c, M]
    return center_to_all_geo_dist @ p_v_given_x.t()                                       # [n, c]


def cycle_consistency_loss_v2_allpix(center_pixel_features, vertex_features, all_pixel_features,
                                    center_to_all_geo_dist, temperature, chunk_size=256):
    """
    SHIC's L_cyc AS WRITTEN (arXiv 2407.18907 Sec 3.3: sum over u in U_I and v in U_I, U_I = the object
    mask): the return leg lands on ANY mask pixel, not only on the sampled centers. The start pixels u
    stay the run's 2048 centers, so d(u, v) is a direct lookup in the 2D geodesic table, whose rows are
    the centers and whose columns are ALL mask pixels -- no snapping, no new table. (Tal, 2026-09-10.)

    Identical math to cycle_consistency_loss_v2 with the return set enlarged:
        L = mean_u sum_x p(x|u) E_{v ~ p(v|x)}[d(u, v)]
    (swapping the sums avoids ever forming the [n, M] round-trip matrix.)

    Args:
        center_pixel_features   [n, F]  L2-normalized
        vertex_features         [V, F]  live encoder output
        all_pixel_features      [M, F]  L2-normalized, ordered by 2D-table column (pixel_to_node)
        center_to_all_geo_dist  [n, M]  the 2D geodesic table / 2D geodesic diameter
        temperature             T for both legs
        chunk_size              vertices per checkpointed chunk (memory ~ chunk_size * M * 4 B * ~3)
    """
    from torch.utils.checkpoint import checkpoint
    vf = F.normalize(vertex_features, dim=-1)
    sim = center_pixel_features @ vf.t()                                            # [n, V]
    p_x_given_u = F.softmax(sim / temperature, dim=1)                                # [n, V]
    expected = torch.cat([
        # use_reentrant=False: the reentrant form raises under torch.autograd.grad, which the
        # --anchor_balance_from calibration uses (found 2026-09-10); .backward() works with both.
        checkpoint(_return_leg_expected_distance, vf[s:s + chunk_size], all_pixel_features,
                   center_to_all_geo_dist, temperature, use_reentrant=False)
        for s in range(0, vf.shape[0], chunk_size)], dim=1)                          # [n, V]
    loss = (p_x_given_u * expected).sum(dim=1).mean()

    with torch.no_grad():
        num_referenced_vertices = int(torch.unique(sim.argmax(dim=1)).numel())
    return loss, loss.detach().item(), num_referenced_vertices
