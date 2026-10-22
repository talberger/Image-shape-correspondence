"""Relational Knowledge Distillation losses in FEATURE space (Park et al., CVPR 2019,
arXiv:1904.05068), added 2026-08-28 at Tal's request as two additive terms on top of the existing
phase-2 relational loss.

WHAT ALREADY EXISTED, AND HOW THIS DIFFERS
------------------------------------------
`modules/relational_loss.py` is already RKD-D, but applied to GEOMETRY: teacher = the 2D pixel
distance between a center pixel and its neighbour (fixed coordinates, no gradient), student = the
3D mesh distance between the two soft-mapped points. Each side divided by its own FIXED mu, Huber
between them.

This module adds the same two RKD potentials applied to FEATURES instead of positions:
  * RKD-D (distance): the distance between two IMAGE features should match the distance between
    the two corresponding MESH features.
  * RKD-A (angle):    the angle at a center, formed with two of its neighbours, should match
    between image-feature space and mesh-feature space.
Teacher is the frozen photo feature field (no gradient, never changes). Student is the encoder's
mesh features, reached through the same soft mapping the rest of phase-2 uses.

THE PAPER'S EQUATIONS, AND WHAT WE CHANGED
------------------------------------------
  psi_D(t_i, t_j) = (1/mu) * ||t_i - t_j||_2,  mu = mean pairwise distance over the batch
  L_RKD-D         = sum l_delta( psi_D(teacher pair), psi_D(student pair) )
  psi_A(t_i,t_j,t_k) = <e_ij, e_kj>,  e_ij = (t_i - t_j)/||t_i - t_j||   (cosine of angle AT j)
  L_RKD-A         = sum l_delta( psi_A(teacher triplet), psi_A(student triplet) )
  l_delta         = Huber

LIVE mu WAS BROKEN UNTIL 2026-08-29 -- it detached the student normalizer and collapsed the
model to a point. Fixed below; see the comment in rkd_feat_dist_loss. Any result recorded
against `--feat_mu_mode live` before that date tested the bug, not the paper.

ONE DELIBERATE DEPARTURE: the paper recomputes mu from the live batch every step. We default to a
FIXED mu measured once from the phase-1 encoder, matching what `relational_loss()` already does and
for the same reason -- a loss that only ever compares distances to its own batch's mean cannot see
every distance shrinking together in proportion, which is exactly how the original relational loss
collapsed (postmortem in PROJECT_NOTES.md, 2026-08-10). `mu_*_reference=None` restores the paper's
live behaviour; Tal explicitly wants to try that later, so it is a flag, not a rewrite.

RKD-A needs NO normalizer at all -- a cosine is already scale-free. That is the paper's own
argument for why the angle-wise potential is the stronger of the two.

ON THE CIRCULARITY CONCERN (raised, considered, accepted as a risk to measure)
-----------------------------------------------------------------------------
`soft_feature()` reaches the mesh feature of a pixel through soft_map_to_vertices' weights, and
those weights are themselves computed from image<->mesh feature similarity. So in principle the
feature-distance loss could be satisfied by sharpening the weights rather than by improving the
features. Tal's judgement: the image side is frozen, so the teacher relation cannot move to meet
the student, which limits how far that can go. Recorded here as a thing to watch, not a thing
proven safe -- if beta-only runs improve the loss without improving any eval metric, look here first.
"""
import torch
import torch.nn.functional as F


def soft_feature(vertex_features, topk_idx, weights):
    """The mesh-space FEATURE of a pixel: the same soft mapping the rest of phase-2 uses, applied
    to features instead of to positions.

    This is the feature analogue of soft_map_to_vertices' returned point. Differentiable through
    both `vertex_features` and `weights`, exactly like the position path.

        vertex_features [V, F]     live encoder output
        topk_idx        [n, k]     from soft_map_to_vertices(..., return_topk=True)
        weights         [n, k]     softmax weights, same call
    Returns [n, F].

    MEMORY (2026-09-04). The obvious form, `(weights.unsqueeze(-1) * gathered).sum(dim=1)`,
    materializes an [n, k, F] tensor TWICE -- once for the gather, once for the product. At the
    standard recipe (n = 2048 centers + 2048*16 neighbours = 34816, k = 10, F = 2304) that is
    2.99 GiB EACH, which OOMs a 16 GB GPU (it fit on the 24 GB cards earlier pods had). The bmm
    below keeps the gather but drops the product temporary, halving the peak. Same arithmetic, so
    results are unchanged up to float-summation order.
    """
    gathered = vertex_features[topk_idx]                    # [n, k, F]
    return torch.bmm(weights.unsqueeze(1), gathered).squeeze(1)


def hard_feature(vertex_features, topk_idx, weights):
    """The k=1 alternative to soft_feature(): the mesh feature of a pixel is the feature of its
    SINGLE best-matching vertex, with no mixing at all.

    Added 2026-09-01 (Tal: "let's try it, nothing to lose"). Motivation, from the 2026-08-29
    circularity test: `soft_feature` reaches the mesh through a softmax-weighted average, and the
    gradient can be reduced by re-mixing the weights instead of by moving features. Detaching the
    weights blocks that, but leaves the student as an average over k vertices with FROZEN
    coefficients -- so every pixel's gradient is still smeared across k vertices, diluted, and
    conflicting between the thousands of pixels that share them.

    Selecting the argmax removes both problems at once: the assignment is a discrete index, so
    there is no continuous "sharpening" knob for the loss to exploit, and the whole of a pixel's
    gradient lands on ONE vertex's feature. Note the assignment is still self-supplied (it comes
    from the current features), so a wrong match can reinforce itself -- judge this variant on the
    eval metrics, not on the loss curve.

        vertex_features [V, F]   live encoder output
        topk_idx        [n, k]   from soft_map_to_vertices(..., return_topk=True)
        weights         [n, k]   softmax weights from the same call; only their argmax is used
    Returns [n, F], differentiable through `vertex_features` only.
    """
    best = weights.argmax(dim=1, keepdim=True)              # [n, 1], no gradient (an index)
    return vertex_features[topk_idx.gather(1, best).squeeze(1)]


def _pair_distances(center_feats, neighbor_feats, eps):
    """||center - neighbour|| for every (center, neighbour) pair.

        center_feats   [n, F]
        neighbor_feats [n, k, F]
    Returns [n, k]. Uses the same sqrt(x + eps) stabilization as relational_loss'/repulsion_loss'
    distance, for the same reason: a pair can legitimately land at distance ~0 and the raw sqrt has
    an infinite gradient there.
    """
    diff = center_feats.unsqueeze(1) - neighbor_feats
    return (diff.pow(2).sum(-1) + eps).sqrt()


def rkd_feat_dist_loss(img_center, img_neighbors, msh_center, msh_neighbors,
                              mu_img_reference=None, mu_msh_reference=None,
                              huber_delta=1.0, eps=1e-12):
    """rkd_feat_dist (RKD-D, 2 points) on features. Teacher = image features (frozen), student = mesh features.

        img_center    [n, F]      frozen photo feature at each center pixel
        img_neighbors [n, k, F]   frozen photo feature at each neighbour pixel
        msh_center    [n, F]      soft-mapped mesh feature at each center pixel
        msh_neighbors [n, k, F]   soft-mapped mesh feature at each neighbour pixel
        mu_img_reference  float or None   FIXED normalizer; None = live batch mean (the paper's)
        mu_msh_reference  float or None   same, mesh side

    Returns (loss, mu_img_live, mu_msh_live) -- the two live means are returned for logging
    ALWAYS, even when fixed references are in use, because watching the live mesh mean drift away
    from its fixed reference is precisely the signal that exposed the 2026-08-10 collapse.
    """
    d_img = _pair_distances(img_center, img_neighbors, eps)          # [n, k], no grad (frozen)
    d_msh = _pair_distances(msh_center, msh_neighbors, eps)          # [n, k], on the grad path

    # BUGFIX 2026-08-29. Both live means used to be .detach()ed, and the mesh (STUDENT) one being
    # detached COLLAPSED the model: mu_msh_live 0.740 -> 0.000 in 100 steps, all 4036 vertices onto
    # one point. Mechanism: the loss VALUE is scale-invariant (shrink every d_msh and numerator and
    # denominator move together), but with mu frozen inside the step the GRADIENT sees only "shrink
    # d_msh, loss drops" -- and next step mu is recomputed smaller. A runaway to zero.
    # Verified against the official RKD repo (lenscloth/RKD, metric/loss.py, RkdDistance.forward):
    # the TEACHER's normalizer is under `with torch.no_grad()`, the STUDENT's `mean_d` is NOT --
    # it is differentiable. With the gradient flowing through mu the invariance holds in the
    # gradient too: a uniform shrink gives gradient ZERO, so there is nothing to run away.
    # So: teacher (image) side stays detached -- it is frozen anyway and matches the paper;
    # student (mesh) side must NOT be, when it is actually being used as the normalizer.
    mu_img_live = d_img.mean().detach()
    mu_msh_live_grad = d_msh.mean()                 # differentiable -- the paper's student mu
    mu_msh_live = mu_msh_live_grad.detach()         # detached copy, for logging/returning only

    norm_img = d_img / (mu_img_live.clamp(min=1e-8) if mu_img_reference is None
                        else max(mu_img_reference, 1e-8))
    norm_msh = d_msh / (mu_msh_live_grad.clamp(min=1e-8) if mu_msh_reference is None
                        else max(mu_msh_reference, 1e-8))

    loss = F.huber_loss(norm_msh, norm_img, delta=huber_delta, reduction='mean')
    return loss, float(mu_img_live), float(mu_msh_live)


def rkd_feat_angle3_loss(img_center, img_neighbors, msh_center, msh_neighbors,
                           huber_delta=1.0, eps=1e-12):
    """rkd_feat_angle3 (RKD-A, 3 points) on features. The triplet is (neighbour_a, CENTER, neighbour_b) and the angle is the one
    AT THE CENTER -- which is what the existing center/neighbour sampling already gives us for free,
    with no new sampling machinery.

    No normalizer: psi_A is a cosine, already scale-free. This is the paper's stated reason for
    preferring the angle-wise potential -- it transfers relational structure without being tied to
    either domain's distance scale, and our two domains have genuinely different scales (image
    features carry norm ~0.707 after the 0.5/0.5 fusion, mesh features are unit-norm).

    Every ordered pair of distinct neighbours (a, b) contributes one angle. The self-pairs a == b
    are excluded -- their cosine is identically 1 on both sides, so they would add a constant zero
    residual and dilute the mean by k out of k^2 terms.

    Returns (loss, mean_abs_cos_gap) -- the second is a plain diagnostic: the average |teacher
    cosine - student cosine| BEFORE the Huber, i.e. how far apart the two angular structures
    actually are, in units a person can read.
    """
    e_img = F.normalize(img_neighbors - img_center.unsqueeze(1), dim=-1, eps=eps)   # [n, k, F]
    e_msh = F.normalize(msh_neighbors - msh_center.unsqueeze(1), dim=-1, eps=eps)   # [n, k, F]

    cos_img = torch.bmm(e_img, e_img.transpose(1, 2))    # [n, k, k], no grad (frozen)
    cos_msh = torch.bmm(e_msh, e_msh.transpose(1, 2))    # [n, k, k], on the grad path

    k = e_img.shape[1]
    off_diag = ~torch.eye(k, dtype=torch.bool, device=e_img.device)
    cos_img = cos_img[:, off_diag]                       # [n, k*(k-1)]
    cos_msh = cos_msh[:, off_diag]

    loss = F.huber_loss(cos_msh, cos_img, delta=huber_delta, reduction='mean')
    return loss, float((cos_msh - cos_img).abs().mean().detach())


def rkd_feat_angle2_loss(img_center, img_neighbors, msh_center, msh_neighbors,
                                    huber_delta=1.0, eps=1e-12):
    """rkd_feat_angle2: PAIRWISE (2-point) angle on features -- Tal's variant, 2026-09-04. NOT the paper's RKD-A.

    THE DIFFERENCE FROM rkd_feat_angle3_loss(), which matters and is easy to blur:
      * RKD-A (the paper, above) is 3-POINT: it builds the difference vectors e_a = n_a - center
        and e_b = n_b - center, and matches <e_a, e_b> -- the angle AT the center subtended by two
        NEIGHBOURS. Every ordered neighbour pair contributes, so the term is O(k^2) per center.
      * THIS one is 2-POINT: the relation is cos(center, neighbour) directly, one per
        (center, neighbour) pair, O(k). It is the same 2-point structure the DISTANCE term already
        uses -- center vs each neighbour -- with a cosine substituted for the distance.

    WHY TAL WANTS IT (2026-09-04). The distance term needs `mu` on both sides, and the whole
    2026-08-29 -> 2026-08-31 normalizer thread (live vs fixed, mean/max/p95) failed to beat baseline
    with it. A cosine is scale-invariant by construction, so `mu` DISAPPEARS ENTIRELY -- nothing to
    measure, freeze, or get wrong -- while keeping the pairwise structure the 3-point angle gives up.
    Note the 3-point term at gamma=0.2 collapsed a run (AUC 54.0/48.9/47.8 vs 76.2/75.7/73.2), which
    is the reason to start this one an order of magnitude lower.

    INTENDED USE: as an ADDITIVE term alongside the geometric relational loss, never instead of it
    (Tal: "זה יהיה בנוסף והם ירוצו שניהם יחד ממושקלים"). The known risk is the same one that sank the
    soft DISTANCE student -- with `F` free, an angle relation can be satisfied by reshaping `F` for
    any `w`, without pinning the correspondence. The spatial term running alongside is what is meant
    to hold the map in place. `--feat_detach_features` is the ablation that tests exactly this.

        img_center    [n, F]      frozen photo feature at each center pixel
        img_neighbors [n, k, F]   frozen photo feature at each neighbour pixel
        msh_center    [n, F]      mesh feature at each center pixel (soft_feature/hard_feature)
        msh_neighbors [n, k, F]   mesh feature at each neighbour pixel

    Returns (loss, mean_abs_cos_gap) -- the diagnostic is the average |teacher cos - student cos|
    BEFORE the Huber, in readable units, same convention as rkd_feat_angle3_loss.
    """
    cos_img = F.cosine_similarity(img_center.unsqueeze(1), img_neighbors,
                                  dim=-1, eps=eps)          # [n, k], no grad (frozen teacher)
    cos_msh = F.cosine_similarity(msh_center.unsqueeze(1), msh_neighbors,
                                  dim=-1, eps=eps)          # [n, k], on the grad path

    loss = F.huber_loss(cos_msh, cos_img, delta=huber_delta, reduction='mean')
    return loss, float((cos_msh - cos_img).abs().mean().detach())


def measure_feature_mu_references(img_center, img_neighbors, msh_center, msh_neighbors, eps=1e-12,
                                  stat='mean'):
    """The two FIXED normalizers rkd_feat_dist_loss() needs, measured from ONE draw.

    Callers should average this over several independent neighbour draws, exactly as
    train_relational_loss.py's measure_mu_3D_reference already does -- any single draw of
    2048*16 pairs is itself a sample with variance.

    `stat` picks WHICH statistic of the pair-distance distribution is used as the normalizer:
    'mean' (the paper's, and the default), 'max' (each space divided by its own largest pair
    distance), or 'pNN' for a percentile, e.g. 'p95'. Added 2026-08-31 at Tal's request -- the mean
    is pulled by the bulk of near pairs, and a spread-based normalizer may match the two spaces
    better. NOTE: max/p95 are much LARGER than the mean, so the normalized distances and hence the
    loss shrink -- beta has to be rescaled to keep the same balance against the spatial term.

    Returns (mu_img, mu_msh) as plain floats.
    """
    def reduce(d):
        if stat == 'mean':
            return float(d.mean())
        if stat == 'max':
            return float(d.max())
        if stat.startswith('p'):           # e.g. 'p95', 'p99'
            q = float(stat[1:]) / 100.0
            return float(torch.quantile(d.flatten().float(), q))
        raise ValueError(f'unknown mu stat {stat!r}')

    with torch.no_grad():
        return (reduce(_pair_distances(img_center, img_neighbors, eps)),
                reduce(_pair_distances(msh_center, msh_neighbors, eps)))


XSEG_FEAT_MODES = ('table', 'wonly', 'Fonly', 'full')


def xseg_feat_loss(img_center, img_neighbors, center_topk_idx, center_weights,
                   neighbor_topk_idx, neighbor_weights, vertex_features, frozen_vertex_unit,
                   mode, huber_delta=1.0, eps=1e-12):
    """Feature-RKD for CROSS-SEGMENT pairs only (--xseg_feat_mode, Tal 2026-10-19).

    Why: across two segments the 2D geodesic is the wrong teacher when parts touch in the image
    (horse 56069: nose and belly are close in the photo, far on the mesh). Those pairs get this
    term INSTEAD of rkd_spatial_dist; in-segment pairs keep rkd_spatial_dist (the caller splits them).

    teacher  t_ij = cos(f_i, f_j)   frozen photo features (DINO+SD), no gradient
    student, by mode -- s_i = sum_a w_ia V[v_a], with V and w as below:
      'table'  V = frozen_vertex_unit (the phase-1 features, unit norm, CONSTANT), w live.
               student = <s_i, s_j> = sum_a sum_b w_ia w_jb C[v_a, v_b], C = cos of the frozen vertex
               features -- a fixed table exactly like D3D in rkd_spatial_dist, so the gradient reaches
               the encoder ONLY through the match w. The VxV table is never built: the double sum is
               bilinear, so it equals the dot product of the two weighted means.
      'wonly'  V = vertex_features.detach() (the LIVE features, detached), w live; student = cos(s_i, s_j).
               Same gradient path as 'table' (w only), but the values drift with training.
      'Fonly'  V = vertex_features (live), w DETACHED; student = cos(s_i, s_j). The gradient reaches the
               encoder only by reshaping F under a fixed match -- the circular path on its own.
      'full'   V = vertex_features (live), w live, nothing detached (Tal 2026-10-22); student = cos(s_i, s_j).
               Both paths at once, the circular one included -- accepted knowingly.

        img_center         [n, D]        photo feature at each centre
        img_neighbors      [n, k, D]     photo feature at each CROSS-segment neighbour
        center_topk_idx    long [n, K]   centre's top-K vertices; center_weights [n, K] their softmax
        neighbor_topk_idx  long [n, k, K]; neighbor_weights [n, k, K]
        vertex_features    [V, D]        live encoder output
        frozen_vertex_unit [V, D] or None  needed for 'table' only
    Returns (loss, mean |student - teacher|, mean teacher, mean student) -- the last three are floats.
    """
    if mode not in XSEG_FEAT_MODES:
        raise ValueError('xseg_feat_loss: mode must be one of %s, got %r' % (XSEG_FEAT_MODES, mode))
    n, k = neighbor_topk_idx.shape[:2]
    K = center_topk_idx.shape[1]
    with torch.no_grad():
        teacher = F.cosine_similarity(img_center.float().unsqueeze(1), img_neighbors.float(),
                                      dim=-1, eps=eps)                              # [n, k]
    idx = torch.cat([center_topk_idx, neighbor_topk_idx.reshape(n * k, K)], dim=0)
    w = torch.cat([center_weights, neighbor_weights.reshape(n * k, K)], dim=0)
    if mode == 'table':
        if frozen_vertex_unit is None:
            raise ValueError("xseg_feat_loss: mode 'table' needs frozen_vertex_unit")
        s = soft_feature(frozen_vertex_unit, idx, w)
    elif mode == 'wonly':
        s = soft_feature(vertex_features.detach(), idx, w)
    elif mode == 'Fonly':
        s = soft_feature(vertex_features, idx, w.detach())
    else:
        s = soft_feature(vertex_features, idx, w)
    s_c, s_n = s[:n], s[n:].view(n, k, -1)
    if mode == 'table':
        student = (s_c.unsqueeze(1) * s_n).sum(-1)                                  # [n, k]
    else:
        student = F.cosine_similarity(s_c.unsqueeze(1), s_n, dim=-1, eps=eps)
    loss = F.huber_loss(student, teacher.to(student.dtype), delta=huber_delta, reduction='mean')
    with torch.no_grad():
        return (loss, float((student - teacher).abs().mean()),
                float(teacher.mean()), float(student.mean()))


# CANONICAL NAMES (Tal, 2026-09-15): rkd_feat_dist / rkd_feat_angle2 / rkd_feat_angle3. The old names stay
# importable so older code and archived notes still resolve.
feature_rkd_distance_loss = rkd_feat_dist_loss
feature_rkd_pairwise_angle_loss = rkd_feat_angle2_loss
feature_rkd_angle_loss = rkd_feat_angle3_loss
