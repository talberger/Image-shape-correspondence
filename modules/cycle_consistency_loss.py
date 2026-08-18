"""Cycle-consistency loss for the relational-loss encoder training (phase2/train_relational_loss.py).

Tal measured (2026-08-14, analysis/vertex_mapping_stats.py on encoder_dino_relational_geodesic):
only 603/4036 vertices (14.9%) receive any pixel mapping at all, and the top 1% of covered
vertices absorb 10.4% of all pixels -- severe many-to-one collapse. This loss penalizes that
directly, styled after two things discussed together:

  1. modules/best_segmentation_buddies.py's find_bsb -- a mutual pixel<->vertex nearest-neighbor
     ("best buddies") reciprocity check, this project's own baseline's namesake mechanism. That
     check is hard/boolean and inference-only (never used in training); this is a differentiable,
     trainable version of the same idea.
  2. SHIC's L_cyc (arXiv 2407.18907): p(v|u) = sum_xk p(v|xk)*p(xk|u), loss = sum_u sum_v
     ||u-v||*p(v|u) -- a round-trip pixel -> vertex -> pixel probability, penalized by how far
     the round trip lands from where it started. SHIC itself subsamples 100 points per image
     for this (their own words: "we randomly select 100 foreground points from each image"),
     confirming the batch-restriction below is not just our own tractability compromise.

Two deliberate departures from SHIC, both Tal's calls:
  - Distance weighting is GEODESIC (mask-constrained), not Euclidean -- matches why the main
    relational_loss_geodesic already switched off Euclidean this session (see
    modules/relational_loss.py's geodesic-variant docstring). Reuses the same 2D geodesic table
    (analysis/build_geodesic_tables.py) already built for that purpose.
  - The round trip is restricted to this run's FIXED --num_center_pixels center pixels for BOTH
    legs, not the full ~205,783-pixel mask (SHIC's own full-image U_I). Necessary for
    tractability (the full mask would need an intractable [205783, 205783] pairwise structure
    every step) and for stability (the centers are the one pixel set proven stable across a
    whole run -- everything else, including neighbours, is resampled every step; comparing a
    round trip against a moving target would repeat the exact live-vs-fixed collapse risk
    relational_loss.py's own docstring documents for mu_3D_reference).

Forward leg (pixel -> vertex): uses this project's own SOFT-TO-HARD mechanism -- the same
soft_map_to_vertices(...) + nearest-real-vertex snap as analysis/color_mesh_from_soft_mapping.py
-- not a plain cosine-similarity argmax, because it needs to target the exact quantity
vertex_mapping_stats.py measures (the hard-snapped vertex a pixel actually ends up assigned to),
not a different proxy (raw top-1 feature match) the model doesn't actually use for its real
correspondence. The hard-snap index is only ever used to GATHER rows out of vertex_features
(differentiable in the gathered values, not in which index got picked -- same convention as
topk_idx everywhere else in this codebase), so it costs nothing in gradient fidelity.

Backward leg (vertex -> pixel): deliberately NOT a mirrored soft-to-hard pixel-position
function -- no such thing exists in this codebase, and building one would actively break
differentiability rather than just be extra work. A hypothetical soft 2D landing point
(weighted average of candidate pixel positions) would sit at an arbitrary continuous location
the geodesic table has no entry for; computing a distance for it would require hard-snapping to
the nearest real pixel first, which reintroduces exactly the non-differentiable-selection trap
modules/relational_loss.py's expected_geodesic_distance_3d was already built to avoid (loss
becomes a function of WHICH pixel got snapped to, not smoothly of the soft point). The fix here
mirrors that one exactly, one dimension over: never collapse to a point, keep the full softmax
distribution over the candidate pool and take a weighted expectation of the (fixed,
precomputed) geodesic distances directly.

KNOWN BLIND SPOT, worth watching when evaluating a run: this loss only penalizes RECIPROCITY
(many-to-one collapse), not geometric CORRECTNESS. It would be equally satisfied by every
center pixel mapping to an arbitrary but mutually-distinct vertex, regardless of whether that
vertex is anywhere near the right part of the mesh -- that check is still entirely
relational_loss's job. Watch mu_3D_live / renders, not just vertex coverage %, when sweeping
--cyc_weight; coverage improving alongside visibly worse correspondence means the weight is
too high relative to relational_loss's own signal.

WEIGHTING (2026-08-14, Tal's explicit request, checked directly against the paper's full text,
Section 3.6/4.1): SHIC's own total loss is L = alpha*L_pseudo + beta*L_cyc + gamma*L_dist +
delta*L_eq (+ zeta*L_syn), with alpha=0.1, beta=0.002, gamma=0.002, delta=0.001, zeta=0.1 -- and
their own stated principle is "we used values for the loss hyper-parameters that make the losses
roughly of similar scale," not a universal ratio. Their literal numbers don't transfer here:
their L_cyc/L_dist happen to already share a raw scale in their pipeline (hence beta==gamma),
ours did not -- this loss's round-trip distance was raw geodesic PIXELS (0-1169.7px on this
mesh/mask) against relational_loss's own already-normalized ([0,~1], Huber-delta=1.0) residual,
off by ~3 orders of magnitude. Applied their PRINCIPLE instead of their numbers: normalized
center_geo_dist by the same geodesic diameter relational_loss_geodesic already divides by (see
this function's own docstring below) so this loss's raw scale now matches relational_loss's, and
phase2/train_relational_loss.py's new --relational_weight (alpha, default 1.0) / --cyc_weight
(beta) can be swept in the same small comparable range for both, the way SHIC's own beta/gamma
are directly comparable in their pipeline.

CONCLUSION (2026-08-15, Tal's call): PARKED, not deleted. --cyc_weight defaults to 0.0 (already
an exact no-op) and should stay 0 -- do not resume sweeping this loss without a new idea, see
below for why. The 2026-08-14 sweep (this hard-snap version, --cyc_weight 0.01-1.0) never beat
the no-cyc baseline on both drift and correspondence quality together. Suspecting the HARD SNAP
below (xk = argmin(...), under no_grad) as the cause -- it makes two pixels that collide onto
the same vertex computationally IDENTICAL to this loss (same vf_xk row -> same weights_back ->
same gradient), so it structurally cannot separate the exact many-to-one collapse it was meant
to fix -- a fully-soft, SHIC-faithful alternative was built (modules/cycle_consistency_loss_soft.py,
--cyc_soft) that never snaps, keeping the full top-K distribution on both legs via
soft_map_to_vertices(..., return_topk=True), the same technique relational_loss.py's own
expected_geodesic_distance_3d already uses.

That fix made things WORSE, not better, and worse the more weight it got: at cyc_weight=0.01,
refs (unique vertices touched by the 2048 centers) collapsed to 51/4036 (vs. this hard-snap
version's 334/4036 at the same weight); at cyc_weight=0.3, refs collapsed to 19/4036 within 100
steps, with drift crashing to ~0.11-0.16 by then too (worse than this version's *final* drift at
any weight tested). Ruled out "denser gradient at the same nominal weight" as the reason --
step-1 grad norms are nearly identical between hard-snap and soft at matching weights (0.01:
2.02e-04 vs 2.01e-04; 0.3: 5.97e-03 vs 5.92e-03), so the two mechanisms hadn't diverged yet at
step 1; whatever causes soft to collapse harder builds up over the run, not from the gradient
magnitude alone.

Re-reading SHIC (arXiv 2407.18907) surfaced two structural gaps neither variant's mechanics can
fix: (1) SHIC's L_cyc/L_eq are purely relative (same family as this loss) but are NEVER used
alone -- they always ride on top of L_pseudo/L_dist, an ABSOLUTE per-pixel-to-vertex anchor built
from a totally separate frozen pipeline (rendering the template from 72 fixed views, matching the
real photo against those renders with frozen SD-DINO features -- "we render a view of the 3D
template and, given a query location in the source image, we find the corresponding vertex as a
visual match on the rendered images"). This project has NO equivalent anchor anywhere --
mapping_loss/relational_loss/cycle_consistency_loss are all purely relative, so cyc_loss has been
asked to fight collapse with nothing holding the overall mapping in a sensible global
configuration, a job it was never designed for on its own. (2) SHIC trains one category's model on
hundreds to thousands of real images ("as few as a few hundred images," their own low-data
ablation bottoms out at 50); this project trains on a SINGLE image, a regime SHIC was never built
or tested for.

Neither gap is fixable by tuning --cyc_weight or the snap mechanics. Revisiting this would mean
building a real anchor (e.g. a scaled-down version of SHIC's render-and-match pseudo-label
pipeline against this project's own template mesh), not another sweep -- see PROJECT_NOTES.md,
2026-08-15, for the full investigation.
"""
import torch
import torch.nn.functional as F

from modules.mapping_loss import soft_map_to_vertices


def cycle_consistency_loss(center_pixel_features, vertex_features, vertices,
                           k_vertices_per_pixel, center_geo_dist):
    """center_pixel_features float [n, F]  this step's center-pixel DINO features (already
                                            normalized -- pixel_features[:num_centers] from
                                            train_relational_loss.py's own index_pixel_features)
       vertex_features        float [V, F]  live encoder output, same tensor passed to
                                            soft_map_to_vertices elsewhere this step
       vertices                float [V, 3]  mesh vertex positions
       k_vertices_per_pixel    int           same K used everywhere else this run
       center_geo_dist         float [n, n]  precomputed, no-grad geodesic pixel distance between
                                            every pair of this run's fixed centers, NORMALIZED by
                                            the 2D geodesic diameter (train_relational_loss.py's
                                            precompute_center_pixel_geodesic_distances output
                                            divided by cyc_geo_diag_2d there) -- same
                                            divide-by-diameter convention as relational_loss.py's
                                            own norm_d_2D, so this loss's raw scale is roughly
                                            [0, ~1] like relational_loss's, not raw pixels.
                                            Otherwise --relational_weight/--cyc_weight sweep
                                            values would be incomparable (SHIC's own stated
                                            principle: pick weights that make the losses
                                            "roughly of similar scale" -- normalizing first is
                                            what makes that comparison meaningful here).

    Returns (loss, loss.detach().item(), num_referenced_vertices) -- the last a coverage
    diagnostic (how many of the V vertices this step's n centers' own top-1 hard-snap actually
    touched), logged alongside mu_2D/mu_3D_live at the existing checkpoint-interval print.
    """
    n = center_pixel_features.shape[0]

    # forward: soft-to-hard vertex pick, same mechanism as color_mesh_from_soft_mapping.py.
    # Only used as an index below (gather), so the hard argmin costs nothing in gradient
    # fidelity -- see module docstring.
    soft_points = soft_map_to_vertices(center_pixel_features, vertex_features, vertices,
                                       k_vertices_per_pixel)                          # [n, 3]
    with torch.no_grad():
        xk = torch.cdist(soft_points, vertices).argmin(dim=1)                        # [n]

    # backward: soft distribution over the SAME center pool. Full softmax, no top-K truncation
    # -- n (num_centers) is small enough (~2048) that this is cheap without it, unlike the
    # V=4036-wide forward side inside soft_map_to_vertices.
    vf_xk = F.normalize(vertex_features, dim=-1)[xk]                                  # [n, F]
    weights_back = F.softmax(vf_xk @ center_pixel_features.t(), dim=1)                # [n, n] = p(v|xk)

    # expected GEODESIC round-trip distance -- weighted expectation over fixed distances, never
    # hard-snap the return pixel on the gradient path (see module docstring).
    loss = (weights_back * center_geo_dist).sum(dim=1).mean()

    num_referenced_vertices = int(torch.unique(xk).numel())
    return loss, loss.detach().item(), num_referenced_vertices
