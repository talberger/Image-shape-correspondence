"""
Anchor-loss PRECONDITION CHECK (2026-09-10): are the ZS (phase-1-only) errors SCATTERED or SYSTEMATIC?

The soft ZS anchor, L_anchor = -sum_x q_ZS(x|u) log p(x|u), only helps if the teacher's errors do
not conspire: random noise cannot add up to a coherent wrong map, a systematic error gets LOCKED IN
by the relational term. Measured on the 100 hand-labeled eval points, no training.

Three questions:
  1. SYMMETRY: are wrong points explained by the mirror of the ground truth -- left/right (x) or
     front/back (z, the guitar's thickness axis)? Checked on the argmax AND on q's mass.
  2. COHERENT DISPLACEMENT: do points that are near each other in the photo get displaced in the
     same 3D direction? Compared against a random-pair baseline.
  3. q SHAPE vs TEMPERATURE: how much of q's mass sits near the GT / far from it / on the mirror,
     and how peaked it is -- this is what picks T_q for the anchor.

Run:
cd $DISC_ROOT && python analysis/zs_error_structure_check.py \
  --save_dir demo/guitar/features_analyze/anchor_precondition
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pixel_to_vertex_nn import load_features, load_vertex_features
from modules.utils import loadmesh


def mirror_vertex(vert_pos, axis):
    """Nearest vertex to each vertex's mirror image across the bbox-centre plane of `axis`.
    Returns (idx [V], euclidean distance from the mirrored position to that vertex [V])."""
    lo, hi = vert_pos.min(0).values, vert_pos.max(0).values
    centre = (lo[axis] + hi[axis]) / 2
    m = vert_pos.clone()
    m[:, axis] = 2 * centre - m[:, axis]
    d = torch.cdist(m, vert_pos)
    dist, idx = d.min(dim=1)
    return idx, dist


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    pairs = torch.load(args.eval_pairs_path)
    rows, cols, gt = pairs['row'], pairs['col'], pairs['vertex_idx'].long()
    n = len(gt)

    geo = torch.load(args.geo_table_path, map_location='cpu')
    G, D = geo['table'].float(), float(geo['diagonal'])

    print(f'loading pixel features {args.features_path} ...')
    feats = load_features(args.features_path)
    pix = F.normalize(feats[rows, cols].float(), dim=-1)
    del feats
    vert = F.normalize(load_vertex_features(args.vertex_features_path, pix.shape[-1]), dim=-1)
    vpos = loadmesh(dir=args.obj_path, name=args.name).vertices.float().cpu()
    V = vert.shape[0]

    S = pix @ vert.T                                               # [n, V]
    am = S.argmax(1)
    err = G[am, gt] / D                                            # fraction of diameter
    wrong = err > args.wrong_thr
    print(f'\nn={n}  mean argmax err {100*err.mean():.2f}%  (should match the ZS eval: 2.7%)')
    print(f'wrong (> {100*args.wrong_thr:g}% of D): {int(wrong.sum())}/{n}')

    # ---- 1. symmetry --------------------------------------------------------------------------
    print('\n=== 1. SYMMETRY: is a wrong prediction sitting on the MIRROR of the ground truth? ===')
    sym = {}
    for name, axis in (('left/right (x)', 0), ('front/back (z)', 2)):
        midx, mdist = mirror_vertex(vpos, axis)
        gt_m = midx[gt]
        valid = (mdist[gt] < args.mirror_snap_thr * D) & (G[gt_m, gt] / D > args.mirror_sep_thr)
        err_m = G[am, gt_m] / D
        explained = wrong & valid & (err_m < args.near_thr)
        sym[name] = dict(gt_mirror=gt_m, valid=valid, err_to_mirror=err_m, explained=explained)
        print(f'  {name:<15} mirror distinguishable from GT for {int(valid.sum()):>3}/{n} points; '
              f'wrong AND within {100*args.near_thr:g}% of the mirror: '
              f'{int(explained.sum())}/{int(wrong.sum())}')
        # the whole error distribution, not just the wrong ones: closer to mirror than to GT?
        closer = valid & (err_m < err)
        print(f'  {"":<15} argmax closer to mirror than to GT (any error size): '
              f'{int(closer.sum())}/{int(valid.sum())}')

    # ---- 2. coherent displacement -------------------------------------------------------------
    print('\n=== 2. COHERENT DISPLACEMENT: do photo-neighbours get pushed the same way? ===')
    disp = vpos[am] - vpos[gt]                                     # [n, 3]
    mag = disp.norm(dim=1)
    bias = (disp.mean(0).norm() / mag.mean()).item()
    bias_w = (disp[wrong].mean(0).norm() / mag[wrong].mean()).item() if wrong.any() else float('nan')
    print(f'  global bias |mean d| / mean|d|: all {bias:.3f} (random ~{1/n**0.5:.3f}), '
          f'wrong only {bias_w:.3f} (random ~{1/max(int(wrong.sum()),1)**0.5:.3f})')
    xy = torch.stack([rows, cols], 1).float()
    d2 = torch.cdist(xy, xy)
    d2.fill_diagonal_(float('inf'))
    k = args.k_photo_neighbours
    nn_idx = d2.topk(k, largest=False).indices                     # [n, k]
    unit = F.normalize(disp, dim=1)
    moving = mag > 1e-6
    def mean_cos(i_idx, j_idx):
        ok = moving[i_idx] & moving[j_idx]
        return (unit[i_idx[ok]] * unit[j_idx[ok]]).sum(1).mean().item(), int(ok.sum())
    src = torch.arange(n).repeat_interleave(k)
    src_w = src[wrong[src]]
    dst_w = nn_idx.reshape(-1)[wrong[src]]
    c_nn, m_nn = mean_cos(src_w, dst_w)
    g = torch.Generator().manual_seed(0)
    base = []
    for _ in range(200):
        base.append(mean_cos(src_w, torch.randint(0, n, (len(src_w),), generator=g))[0])
    base = torch.tensor(base)
    print(f'  wrong points vs their {k} nearest labeled photo-neighbours: mean displacement cosine '
          f'{c_nn:.3f} over {m_nn} pairs; random-partner baseline {base.mean():.3f} '
          f'(95% {base.quantile(0.025):.3f}..{base.quantile(0.975):.3f})')
    print(f'  median photo distance to those neighbours: {d2.topk(k, largest=False).values.median():.0f} px')

    # ---- 3. q shape vs temperature ------------------------------------------------------------
    print('\n=== 3. q_ZS SHAPE vs TEMPERATURE (mean over the 100 points) ===')
    near_gt = (G[:, gt].T / D) < args.near_thr                       # [n, V]
    far_gt = (G[:, gt].T / D) > 0.20
    lr = sym['left/right (x)']
    near_mirror = (G[:, lr['gt_mirror']].T / D < args.near_thr) & ~near_gt
    hdr = (f'  {"T":>6} {"eff #verts":>10} {"top1":>6} {"top10":>6} {"top50":>6} '
           f'{"<"+format(100*args.near_thr,"g")+"% of GT":>10} {">20% of GT":>10} {"L/R mirror":>10}')
    print(hdr)
    qstats = {}
    for T in args.temperatures:
        q = F.softmax(S / T, dim=1)
        ent = -(q * q.clamp_min(1e-30).log()).sum(1)
        srt = q.sort(1, descending=True).values
        row = dict(eff=ent.exp().mean().item(), top1=srt[:, 0].mean().item(),
                   top10=srt[:, :10].sum(1).mean().item(), top50=srt[:, :50].sum(1).mean().item(),
                   near=(q * near_gt).sum(1).mean().item(), far=(q * far_gt).sum(1).mean().item(),
                   mirror=(q * near_mirror)[lr['valid']].sum(1).mean().item())
        qstats[T] = row
        print(f'  {T:>6g} {row["eff"]:>10.1f} {row["top1"]:>6.3f} {row["top10"]:>6.3f} '
              f'{row["top50"]:>6.3f} {row["near"]:>10.3f} {row["far"]:>10.3f} {row["mirror"]:>10.3f}')
    print(f'  (V={V}; "L/R mirror" = mass within {100*args.near_thr:g}% of the mirrored GT and not '
          f'near the GT itself, over points where the mirror is distinguishable)')

    out = os.path.join(args.save_dir, 'zs_error_structure.pth')
    torch.save(dict(err=err, argmax=am, gt=gt, wrong=wrong, sym=sym, disp=disp, qstats=qstats,
                    nn_cos=c_nn, nn_cos_baseline=base, bias=bias, bias_wrong=bias_w,
                    args=vars(args)), out)
    print(f'\nsaved {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--save_dir', type=str, required=True)
    p.add_argument('--eval_pairs_path', type=str,
                   default='demo/guitar/eval_labels/v2_edge_inset/isobbox/pixel_vertex_eval_pairs.pth')
    p.add_argument('--features_path', type=str,
                   default='demo/guitar/features_analyze/dino_sd_aligned/guitar_01_pix_dino_sd_features_aligned.pth')
    p.add_argument('--vertex_features_path', type=str,
                   default='experiments/guitar/encoder_isobbox_dino_sd_dfd_crossview_phase1/pred_f.pth')
    p.add_argument('--geo_table_path', type=str,
                   default='experiments/guitar/geodesic_tables/guitar_isobbox_geo_table_3d.pth')
    p.add_argument('--obj_path', type=str, default='./meshes/guitar_isotropic_bboxnorm.obj')
    p.add_argument('--name', type=str, default='guitar_isobbox')
    p.add_argument('--wrong_thr', type=float, default=0.05, help='argmax error above this = wrong')
    p.add_argument('--near_thr', type=float, default=0.03, help='"near" radius, fraction of D')
    p.add_argument('--mirror_sep_thr', type=float, default=0.05,
                   help='mirror must be at least this far from the GT to be distinguishable')
    # 0.02, not 0.01: median vertex spacing on the isobbox mesh is 0.79% of D, so at 0.01 the
    # front/back mirror almost never snapped and the test silently reported 0/100 testable points.
    p.add_argument('--mirror_snap_thr', type=float, default=0.02,
                   help='mirrored position must land within this (Euclidean, fraction of D) of a vertex')
    p.add_argument('--k_photo_neighbours', type=int, default=3)
    p.add_argument('--temperatures', type=float, nargs='+', default=[1.0, 0.1, 0.05, 0.02, 0.01, 0.005])
    main(p.parse_args())
