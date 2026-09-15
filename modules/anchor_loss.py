"""
ZS anchor loss (2026-09-10): the ABSOLUTE term phase-2 never had.

Every other phase-2 term is relative (pairwise distances, angles, cycles), so nothing says "pixel u
belongs at vertex x" and the map is free to slide to any self-consistent solution -- which is what
the step ladder shows (-11 AUC in the first 25 steps). This term holds each center pixel's
distribution over vertices near the frozen phase-1 (ZS) distribution.

Structure copied from DensePose-CSE, detectron2 `projects/DensePose/densepose/modeling/losses/soft_embed.py`:

    target = softmax(-geodist[gt, .] / sigma)                       (theirs: GT vertex, geodesic-smoothed)
    loss   = (-target * log_softmax(-sqdist(e_pix, E) / sigma)).sum(1).mean()

Ours differs ONLY in the target, which is the frozen phase-1 prediction instead of a labeled vertex:

    q(x|u)   = softmax(<f_u, E_ZS(x)> / T)        soft, all V vertices, no gradient
    p(x|u)   = log_softmax(<f_u, E(x)> / T)       live encoder, all V vertices
    L_anchor = -sum_x q(x|u) log p(x|u), mean over u

`hard=True` replaces q with the one-hot argmax of the same similarity -- exactly SHIC's L_pseudo
(arXiv 2407.18907), the control arm. T -> 0 makes the soft target converge to it.

WHY ALL VERTICES AND NOT THE top-k THE RELATIONAL TERM USES: d loss / d logit_x = (p(x) - q(x)) / T.
If p were restricted to the student's own top-k, a vertex q likes that the student has already
pushed out of its top-k would get NO gradient -- the anchor could not pull a drifted student back,
which is its whole job. It would also make log p = -inf wherever q > 0 outside that set.

NOTE: at initialization the student IS phase-1, so p = q and the soft term's gradient is exactly
zero -- it is a restoring force that grows with drift. Its weight therefore cannot be calibrated at
step 1; see train_relational_loss.py --anchor_balance_from.

Measured 2026-09-10 (analysis/zs_error_structure_check.py, 100 eval labels): at T=1, q over 3886
vertices is ~uniform (65% of its mass >20% of the diameter from the GT); at T=0.01 it is ~75
effective vertices with 49% of its mass within 3% of the GT. Hence the 0.01 default.
"""
import torch
import torch.nn.functional as F


def anchor_loss(pixel_features, vertex_features, teacher_vertex_features, temperature, hard=False):
    """
    Args:
        pixel_features           [n, F] frozen photo features, already L2-normalized
        vertex_features          [V, F] live encoder output (gradient flows here)
        teacher_vertex_features  [V, F] frozen phase-1 pred_f
        temperature              T for BOTH distributions (same T, so p = q is reachable
                                  without inflating similarity gaps)
        hard                     one-hot argmax target (SHIC L_pseudo) instead of soft q

    Returns (loss, kl, top1_agreement):
        loss            scalar, differentiable
        kl              float, loss - H(q): 0 when p == q (== loss for the hard target)
        top1_agreement  float, fraction of pixels whose student argmax equals the teacher's
    """
    logits = pixel_features @ F.normalize(vertex_features, dim=-1).t() / temperature
    log_p = F.log_softmax(logits, dim=1)

    with torch.no_grad():
        t_logits = pixel_features @ F.normalize(teacher_vertex_features, dim=-1).t() / temperature
        t_argmax = t_logits.argmax(dim=1)
        if hard:
            q = F.one_hot(t_argmax, num_classes=t_logits.shape[1]).to(log_p.dtype)
            entropy = torch.zeros((), device=log_p.device)
        else:
            q = F.softmax(t_logits, dim=1)
            entropy = -(q * F.log_softmax(t_logits, dim=1)).sum(1).mean()

    loss = -(q * log_p).sum(1).mean()
    kl = (loss.detach() - entropy).item()
    agree = (logits.detach().argmax(dim=1) == t_argmax).float().mean().item()
    return loss, kl, agree
