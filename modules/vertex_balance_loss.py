"""Vertex-usage load-balancing loss (2026-08-15) -- an auxiliary term that discourages many
pixels collapsing onto the same vertex, adapted from the Mixture-of-Experts load-balancing
auxiliary loss (Fedus, Zoph & Shazeer, "Switch Transformers", arXiv 2101.03961, Section 2.1) --
checked directly against the paper's exact formula, not from memory.

BACKGROUND: modules/cycle_consistency_loss[_soft].py's reciprocity approach (round-trip
pixel->vertex->pixel) was tried and PARKED 2026-08-15 -- see that module's CONCLUSION docstring
and PROJECT_NOTES.md. This is a DIFFERENT, independent idea for the same underlying problem
(many pixels mapping to the same handful of vertices -- PROJECT_NOTES.md 2026-08-14: only
603/4036 vertices, 14.9%, ever receive a mapping), borrowed from a totally different, well-
established body of literature: MoE expert-routing collapse, VQ-VAE codebook collapse, and
SwAV/DeepCluster online-clustering collapse are all the same underlying shape of problem (a
continuous feature hard-assigned to one of a fixed discrete set of targets, prone to collapsing
onto a handful of targets) -- this loss adapts the simplest, cheapest fix in that literature:
Switch Transformer's additive auxiliary loss, deliberately chosen over SwAV's Sinkhorn-Knopp
equipartition (Tal's call, 2026-08-15) because it is purely ADDITIVE -- it changes nothing about
soft_map_to_vertices or any existing loss, so it can be weighted to zero and walked away from
exactly as cleanly as cyc_loss was, unlike Sinkhorn balancing which would mean rewriting the
shared assignment mechanism every other loss in this codebase depends on.

THE TRICK (verified directly against the paper): for each expert i, f_i = fraction of tokens
whose ARGMAX pick is i (hard, no gradient -- the paper's own words: "the f-vector is not
[differentiable]") and P_i = mean SOFT router probability on expert i across the batch
(differentiable -- "the P-vector is differentiable"). loss = N * sum_i f_i * P_i. Despite f_i
carrying zero gradient, the PRODUCT still gives a real, useful gradient through P_i alone: for
whichever expert is over-used this batch (f_i large), gradient pushes P_i down, in direct
proportion to how overloaded it already is -- no need to differentiate through the hard argmax
itself, the same "hard pick only ever used to gather/index, never on the gradient path"
convention this whole codebase already uses for topk_idx everywhere (see e.g.
modules/mapping_loss.py's soft_map_to_vertices docstring).

ADAPTATION to this codebase's vocabulary: "expert" -> "vertex", "token" -> "center pixel",
"router probability" -> the same top-K softmax weights soft_map_to_vertices(..., return_topk=True)
already produces everywhere else in this codebase -- no full V-way softmax is computed; a vertex
outside every pixel's top-K never receives ANY weight from ANY pixel regardless, so it
contributes f_v=P_v=0 to the sum automatically, consistent with the top-K approximation already
used throughout mapping_loss.py/relational_loss.py. N (number of experts) -> V (number of mesh
vertices) -- the V* scaling is part of the loss's OWN definition (not an external weight): it
normalizes the loss to be ~1.0 at the perfectly-balanced optimum (f_v=P_v=1/V for every v),
regardless of V, matching the paper's own convention. The EXTERNAL weight on this whole term
(train_relational_loss.py's --balance_weight, beta alongside relational_loss's own
--relational_weight alpha) is separate, and is where a sweep should happen -- not inside this
function.

WHAT THIS DOES NOT CHECK: same caveat as cycle_consistency_loss.py's own KNOWN BLIND SPOT --
this loss only rewards spreading usage EVENLY across vertices, never checks whether the vertex a
pixel lands on is anywhere near geometrically correct. That check is still entirely
relational_loss's job.

CONCLUSION (2026-08-15, Tal's call): PARKED, not deleted -- --balance_weight defaults to 0.0
(already an exact no-op) and should stay 0 for now, may be revisited later. Swept at
balance_weight 1e-4/3e-4/1e-3, all on top of --geodesic --mu_3d_scale 1 (Tal's correction from
this session: 1.1 was the best Euclidean/bbox_norm value, not the geodesic one -- geodesic runs
should use --mu_3d_scale 1 going forward). All three DID increase vertex coverage over the
--cyc_weight=0 baseline (629/4036, 812/4036, 869/4036 at step300 vs. baseline's 603/4036) while
keeping drift healthy (0.975-0.989 vs. baseline's ~0.89) -- unlike every cycle_consistency_loss
config tried, which always traded drift away hard for little or no coverage gain. So on the
numbers alone this looked like the first real win in this whole investigation. **But Tal's visual
review of the renders found correspondence quality WORSE despite the better spread** -- more
vertices get used, but not in an organized/geometrically-sensible way. Confirms the loss's own
documented blind spot above in practice, not just in theory: rewarding even usage says nothing
about whether the newly-used vertices are anywhere near correct, and apparently they mostly
aren't. Not chased further with intermediate weights or renders reviewed in more depth -- Tal's
explicit call to stop here. If revisited, the open question is whether pairing this loss with
something that also checks geometric correctness (not just spread) does better than either alone
-- not attempted this session.
"""
import torch


def vertex_balance_loss(topk_idx, weights, num_vertices):
    """Switch-Transformer-style load-balancing loss over vertex usage -- see module docstring.

    Args:
        topk_idx     long  [n, K]  each center pixel's top-K feature-nearest vertex ids -- from
                                   the SAME soft_map_to_vertices(..., return_topk=True) call
                                   train_relational_loss.py already makes for
                                   relational_loss/relational_loss_geodesic this step, no extra
                                   forward pass needed
        weights      float [n, K]  softmax weights over those K candidates, same source
        num_vertices int          V, the mesh's total vertex count (vertices.shape[0])

    Returns (loss, f, P): the scalar loss, plus f/P (both [V], no-grad/detached) for logging --
    e.g. (f > 0).sum() is how many distinct vertices received at least one pixel's top-1 pick
    this step, the same coverage-style diagnostic cycle_consistency_loss[_soft].py's own
    refs=X/V print already uses.
    """
    n, K = topk_idx.shape
    device = topk_idx.device

    # P_v: mean SOFT probability mass on vertex v across all n pixels -- differentiable, the
    # only tensor on the gradient path (scatter_add is linear, so gradient flows straight back
    # to `weights`, and from there into vertex_features via the softmax that produced it).
    P = torch.zeros(num_vertices, device=device, dtype=weights.dtype)
    P.scatter_add_(0, topk_idx.reshape(-1), weights.reshape(-1))
    P = P / n

    # f_v: fraction of pixels whose HARD top-1 pick is vertex v -- no_grad, matches
    # cycle_consistency_loss[_soft].py's own top-1 diagnostic convention.
    with torch.no_grad():
        top1 = topk_idx[torch.arange(n, device=device), weights.argmax(dim=1)]     # [n]
        f = torch.bincount(top1, minlength=num_vertices).float() / n

    loss = num_vertices * (f * P).sum()
    return loss, f, P
