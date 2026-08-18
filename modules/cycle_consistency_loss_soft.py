"""Fully-soft cycle-consistency loss (2026-08-15) -- SHIC-faithful alternative to
modules/cycle_consistency_loss.py's hard-snap version.

modules/cycle_consistency_loss.py hard-snaps the forward leg (pixel -> vertex) to a single
nearest-real-vertex index under torch.no_grad(), matching this project's own topk_idx
convention elsewhere. Checked directly against SHIC's paper text (arXiv 2407.18907, L_cyc,
Section 3.6): this is NOT what SHIC does. SHIC's p(xk|u) and p(v|xk) are both full softmax
distributions, never collapsed to a single index anywhere in L_cyc -- the paper's only argmax
is in the unrelated L_pseudo pseudo-label construction. L_cyc's own equation is a full
marginalization: p(v|u) = sum_xk p(v|xk) * p(xk|u).

The hard-snap version has a concrete failure mode this project's own training logs confirm:
when two different center pixels' soft points snap to the SAME vertex (which the 2026-08-14/15
cyc_weight sweep runs show happening constantly -- refs=196-334/4036 out of 2048 centers, i.e.
roughly 7-10x collision on average, throughout every run in that sweep regardless of
cyc_weight), their vf_xk row becomes literally identical, so their weights_back/loss
contribution is identical too -- the loss has no way left to tell two colliding pixels apart,
exactly the many-to-one collapse cyc_weight was meant to fix. This module fixes that by never
snapping: it keeps this pixel's own top-K forward weights
(soft_map_to_vertices(..., return_topk=True)) and marginalizes over all K candidates on both
legs -- the same technique modules/relational_loss.py's expected_geodesic_distance_3d already
uses for the (unrelated) 3D geodesic distance calc, reused here rather than invented fresh.

Restricting the forward side to the top-K=10 candidates (rather than SHIC's full softmax over
all V vertices) is kept as-is -- that is a property of this whole codebase's DPC-derived
mapping mechanism, not specific to cyc, and not the discrepancy this file targets.

Kept as a SEPARATE file from cycle_consistency_loss.py, not an edit to it -- Tal's explicit
call, so the already-run cyc_weight sweep (2026-08-14/15, hard-snap version) stays exactly
reproducible and comparable against this alternative, same "separate experiment, don't touch
the original" convention as train_relational_loss.py itself follows relative to
train_mapping_loss.py.

CONCLUSION (2026-08-15, Tal's call): PARKED, not deleted -- and this fix did NOT help. Tested at
cyc_weight 0.01 and 0.3 (--geodesic --mu_3d_scale 1.1, 300 steps each): refs (unique vertices
touched by the 2048 centers) collapsed to 51/4036 and 19/4036 respectively -- WORSE than
cycle_consistency_loss.py's hard-snap version at the same weights (334/4036 and, at cyc_weight=1,
233/4036), and worse the more weight it got, the opposite of the predicted direction. Drift also
crashed faster: cyc_weight=0.3 was down to ~0.11-0.16 by step 100 alone. Confirmed this isn't
just "more gradient terms means a bigger step at the same nominal weight" -- step-1 grad norms
are nearly identical to the hard-snap version at matching weights (0.01: 2.01e-04 vs 2.02e-04;
0.3: 5.92e-03 vs 5.97e-03), so whatever drives the extra collapse builds up over the run, not
from step-1 magnitude.

Removing the hard-snap's gradient-degeneracy bug was still the mathematically correct fix
relative to SHIC's actual L_cyc formula (no snap anywhere in the paper) -- but it turned out not
to be the thing standing between this loss and being useful. Re-reading SHIC surfaced two
structural gaps this fix can't address: SHIC's L_cyc is a small regularizer that only ever runs
on top of an absolute per-pixel-to-vertex anchor (L_pseudo/L_dist, built from an entirely
separate frozen render-and-match pipeline) that this project has no equivalent of anywhere, and
SHIC trains on hundreds-to-thousands of real images per category, not the single image this
project trains on. See modules/cycle_consistency_loss.py's own CONCLUSION note and
PROJECT_NOTES.md (2026-08-15) for the full investigation.
"""
import torch
import torch.nn.functional as F

from modules.mapping_loss import soft_map_to_vertices


def cycle_consistency_loss_soft(center_pixel_features, vertex_features, vertices,
                                k_vertices_per_pixel, center_geo_dist):
    """SHIC-faithful cycle-consistency loss -- see module docstring for why this differs from
    cycle_consistency_loss.cycle_consistency_loss's hard-snap version.

    Same args/return signature as that function (drop-in alternative -- see
    train_relational_loss.py's --cyc_soft flag):

       center_pixel_features float [n, F]  this step's center-pixel DINO features (already
                                            normalized -- pixel_features[:num_centers])
       vertex_features        float [V, F]  live encoder output
       vertices                float [V, 3]  mesh vertex positions
       k_vertices_per_pixel    int           same K used everywhere else this run
       center_geo_dist         float [n, n]  precomputed, no-grad, normalized geodesic pixel
                                            distance between every pair of fixed centers -- see
                                            cycle_consistency_loss.py's docstring, unchanged here

    Returns (loss, loss.detach().item(), num_referenced_vertices). num_referenced_vertices is a
    DIAGNOSTIC ONLY (how many distinct vertices this step's per-pixel TOP-1 forward weight would
    point at, computed under no_grad, mirroring the old function's printed coverage number) --
    it plays no role in the loss itself. Unlike the hard-snap version, there is no hard pick
    anywhere on the actual gradient path here.
    """
    n = center_pixel_features.shape[0]
    K = k_vertices_per_pixel

    # forward: KEEP the full top-K soft distribution -- p(xk|u) -- instead of collapsing to a
    # single vertex. return_topk=True gives exactly this, no extra computation beyond what
    # soft_map_to_vertices already does internally to produce the (unused here) soft point.
    _, topk_idx, weights_fwd = soft_map_to_vertices(
        center_pixel_features, vertex_features, vertices, K, return_topk=True)  # [n,K], [n,K]

    # backward: for EACH of the K forward candidates, its own p(v|xk) over the n centers --
    # never reduced to a single candidate's row. vf_topk[u, a, :] is candidate a's (normalized)
    # encoder feature; scores_back[u, a, v] is that candidate's similarity to center pixel v.
    # Size: n*K*n = 2048*10*2048 ~= 42M elements (~168MB float32) for scores_back/weights_back
    # each -- comparable to soft_map_to_vertices's own ~297MB similarity matrix, well within
    # this project's already-budgeted per-step memory.
    vf_topk = F.normalize(vertex_features, dim=-1)[topk_idx]                    # [n, K, F]
    scores_back = torch.einsum('nkf,mf->nkm', vf_topk, center_pixel_features)   # [n, K, n]
    weights_back = F.softmax(scores_back, dim=-1)                               # p(v|xk=a)

    # marginalize over the K forward candidates: p(v|u) = sum_a p(xk=a|u) * p(v|xk=a) -- the
    # actual SHIC equation, no shortcuts.
    p_v_given_u = (weights_fwd.unsqueeze(-1) * weights_back).sum(dim=1)         # [n, n]

    loss = (p_v_given_u * center_geo_dist).sum(dim=1).mean()

    with torch.no_grad():
        top1_vertex = topk_idx[torch.arange(n, device=topk_idx.device), weights_fwd.argmax(dim=1)]
        num_referenced_vertices = int(torch.unique(top1_vertex).numel())

    return loss, loss.detach().item(), num_referenced_vertices
