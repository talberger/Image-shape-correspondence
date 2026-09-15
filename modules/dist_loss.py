"""
SHIC L_dist (2026-09-12): the METRIC term the anchor is blind to.

    L_dist = (1/n) sum_u  sum_x  p(x|u) * d_geo(x, x~_u) / D
    x~_u   = argmax_x q_ZS(x|u)        the frozen phase-1 (ZS) pick, one vertex per center pixel
    p(x|u) = softmax_x( <f_u, E(x)> / T_dist )   live encoder, ALL V vertices
    d_geo  = geo_table_3d, D = the 3D geodesic diameter, so the loss is a fraction of the diameter
             (same units convention as L_spatial, L_cyc and eval)

WHY THIS IS NOT A DUPLICATE OF THE ANCHOR. Both terms reference the same x~, but they read
different things out of p:

    L_anchor = -sum_x q_ZS(x) log p(x)   depends on p only through the entries q likes.
                                          A vertex with q ~ 0 gets ZERO pull whether it sits
                                          1mm from x~ or half a guitar away. METRIC-BLIND.
    L_dist   =  sum_x p(x) d(x, x~)      depends only on WHERE the mass sits, weighted by
                                          distance. Indifferent to how peaked p is. Its
                                          gradient is SIGNED by distance:

    d L_dist / d logit_x = p(x) * ( d(x, x~) - E_p[d] ) / T
        farther than average -> pushed down;  closer than average -> pushed UP.

So the anchor controls the PEAK and is blind to the tail; L_dist controls the TAIL and is blind
to the peak. The measured symptom L_dist targets: hub10 rises to 20-22% under the anchor (ZS:
9.8%) -- mass leaking to far-away popular vertices that the CE never sees.

TWO PROPERTIES THAT DIFFER FROM anchor_loss.py, both deliberate:

1. ITS GRADIENT IS NOT ZERO AT PHASE-1. The soft anchor's is (p == q there), which is why its
   lambda has to be calibrated off a step-25 checkpoint. L_dist is linear in p and its gradient
   at p = q_ZS is nonzero, so gamma CAN be calibrated at step 1. It is a DRIVING force, not a
   restoring one.

2. IT IS LINEAR IN p, so its unique minimizer over the simplex is the delta at x~ -- the same
   fixed point as SHIC's hard L_pseudo, reached by a metric-aware path. That is the "more grip
   on the anchor" this term is for, and also its risk: pushed too hard it collapses diversity
   (the lambda=3.5e-6 anchor arm already dropped argmax coverage to 5.8%). Hence a small gamma,
   calibrated as a FRACTION of the weighted anchor's gradient norm, the way cyc_weight is.

   This is also exactly why the SOFT-label variant stays rejected (2026-09-10): replacing the
   hard x~ with q_ZS makes the objective sum_x sum_y q(y) p(x) d(x,y), whose minimizer is the
   geodesic MEDIAN of q, not q itself.

T_dist IS A SEPARATE FLAG FROM T_anchor, AND ITS DEFAULT IS 1. The gradient above is alive only
where p(x) > 0. At T = 0.01 p is ~75 effective vertices (measured 2026-09-10), so L_dist would
only re-rank inside the anchor's own blob -- nearly a duplicate of it. At T = 1 the mass is
spread over all 3886, which is where the far leak actually lives and where this term earns its
place. T_anchor = 0.01 was chosen to sharpen the TARGET q; that reasoning does not transfer to
the distribution L_dist integrates over.
"""
import torch
import torch.nn.functional as F


def dist_loss(pixel_features, vertex_features, teacher_vertex_features, geo_table_3d,
              geo_diameter, temperature, target_idx=None, tail_frac=0.2, soft_target_temperature=None):
    """
    Args:
        pixel_features           [n, F] frozen photo features, already L2-normalized
        vertex_features          [V, F] live encoder output (gradient flows here)
        teacher_vertex_features  [V, F] frozen phase-1 pred_f -- only its ARGMAX is used, so no
                                  temperature is needed on the teacher side (argmax is scale-free)
        geo_table_3d             [V, V] precomputed vertex-to-vertex geodesic distance, CONSTANT
        geo_diameter             float, 3D geodesic diameter (geo_diag_3d)
        temperature              T_dist for p. Separate from T_anchor on purpose -- see docstring.
        target_idx               the 4th return value of a previous call (skips the teacher matmul).
                                  None = compute.
        soft_target_temperature  2026-09-13, None = hard x~ (unchanged). Otherwise SOFT target:
                                  d(x) = E_{y~q_ZS}[d_geo(x, y)] / D with q_ZS at this T. Its
                                  minimizer is the geodesic median of q, which sits 1% of D from
                                  x~ at T=0.01 but 3.8% at T=0.05 and 7.4% at T=0.1
                                  (analysis/soft_dist_median_check.py) -- so it is only a
                                  different term from the hard one at T >= ~0.05.
        tail_frac                logging only: report the p-mass sitting farther than this
                                  fraction of the diameter from x~ (0.2 = the 2026-09-10
                                  uniform-noise marker)

    Returns (loss, tail_mass, top1_agreement, target_idx):
        loss            scalar, differentiable, in units of the geodesic diameter
        tail_mass       float, mean over pixels of sum_{d > tail_frac} p(x) -- the leak this term
                        exists to drain
        top1_agreement  float, fraction of pixels whose student argmax equals x~
        target_idx      (x~ long [n], d [n, V]) for reuse across calls
    """
    logits = pixel_features @ F.normalize(vertex_features, dim=-1).t() / temperature
    p = F.softmax(logits, dim=1)

    if target_idx is None:
        with torch.no_grad():
            t_logits = pixel_features @ F.normalize(teacher_vertex_features, dim=-1).t()
            idx = t_logits.argmax(dim=1)
            if soft_target_temperature is None:
                # [n, V] constant: each pixel's row of geodesic distances FROM its own x~, as a
                # fraction of the diameter. Only p carries gradient.
                d = geo_table_3d[idx] / geo_diameter
            else:
                q = F.softmax(t_logits / soft_target_temperature, dim=1)
                d = (q @ geo_table_3d) / geo_diameter   # table is symmetric
        target_idx = (idx, d)
    idx, d = target_idx
    target_idx_out = target_idx
    target_idx = idx

    loss = (p * d).sum(dim=1).mean()

    with torch.no_grad():
        tail_mass = (p * (d > tail_frac)).sum(dim=1).mean().item()
        agree = (logits.argmax(dim=1) == target_idx).float().mean().item()

    return loss, tail_mass, agree, target_idx_out
