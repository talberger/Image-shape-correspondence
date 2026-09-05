"""Intrinsic report on a run's FEATURE SPACE itself -- no ground truth, no correspondence.
Five diagnostics (Tal's list, 2026-08-30), computed for the MESH features (a run's pred_f.pth)
and, optionally, the IMAGE features inside the object mask:

  1. feature norm distribution
  2. feature similarity distribution (cosine, random pairs)
  3. effective rank / PCA spectrum
  4. nearest-neighbour distance distribution (in feature space)
  5. local smoothness on the mesh (1-ring neighbours vs random pairs)   [mesh only]

Writes one multi-panel PNG plus a plain-text table; every number printed is also saved to JSON so
later runs can be compared without re-reading the figure.

Run:
python analysis/feature_space_report.py \
  --pred_f_path experiments/guitar/<run>/pred_f.pth \
  --pix_features_path demo/guitar/features_analyze/<...>.pth \
  --mask_path demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --label "s7 + gridalign, T=1, seed 0" --save_dir demo/guitar/<run>/feature_space
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils import loadmesh


def norm_stats(f):
    n = f.norm(dim=-1)
    return dict(mean=n.mean().item(), std=n.std().item(), min=n.min().item(),
                median=n.median().item(), max=n.max().item(),
                cv=(n.std() / n.mean()).item()), n


def pair_sims(f, n_pairs, gen):
    """Cosine similarity over random pairs (features are L2-normalized first)."""
    fn = torch.nn.functional.normalize(f, dim=-1)
    n = fn.shape[0]
    a = torch.randint(0, n, (n_pairs,), generator=gen)
    b = torch.randint(0, n, (n_pairs,), generator=gen)
    keep = a != b
    return (fn[a[keep]] * fn[b[keep]]).sum(-1)


def pca_spectrum(f):
    """Eigenvalues of the covariance, plus three summaries of how many directions are in use."""
    x = (f - f.mean(0, keepdim=True)).double()
    # eigenvalues of the covariance == singular values^2 / (n-1); SVD is the stable route
    s = torch.linalg.svdvals(x)
    ev = (s ** 2) / max(f.shape[0] - 1, 1)
    p = ev / ev.sum()
    p = p[p > 0]
    eff_rank = torch.exp(-(p * p.log()).sum()).item()      # entropy-based effective rank
    part_ratio = (ev.sum() ** 2 / (ev ** 2).sum()).item()  # participation ratio
    cum = torch.cumsum(ev / ev.sum(), 0)
    k = {f'n_for_{int(q*100)}pct': int((cum < q).sum().item()) + 1 for q in (0.90, 0.95, 0.99)}
    return ev.float(), dict(effective_rank=eff_rank, participation_ratio=part_ratio,
                            ambient_dim=f.shape[1], **k)


def nn_distance(f, chunk=512):
    """Cosine distance (1 - cos) from each row to its nearest OTHER row."""
    fn = torch.nn.functional.normalize(f, dim=-1)
    out = torch.empty(fn.shape[0])
    for i in range(0, fn.shape[0], chunk):
        sim = fn[i:i + chunk] @ fn.T
        sim[torch.arange(sim.shape[0]), torch.arange(i, min(i + chunk, fn.shape[0]))] = -2.0
        out[i:i + chunk] = 1.0 - sim.max(dim=-1).values
    return out


def ring_neighbor_sims(f, faces):
    """Cosine similarity across every mesh EDGE (the 1-ring), deduplicated."""
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = torch.unique(torch.stack([e.min(-1).values, e.max(-1).values], -1), dim=0)
    fn = torch.nn.functional.normalize(f, dim=-1)
    return (fn[e[:, 0]] * fn[e[:, 1]]).sum(-1), e.shape[0]


def describe(t):
    # NOTE: the quantile list is kept as python floats and the key built with round(), not
    # int(100*q) off a float32 tensor -- float32 0.01 is 0.00999999978, so int() gave 'p0'.
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    v = torch.quantile(t.float(), torch.tensor(qs, dtype=torch.float64).float())
    return dict(mean=t.mean().item(), std=t.std().item(), min=t.min().item(), max=t.max().item(),
                **{f'p{round(100 * qq)}': vv.item() for qq, vv in zip(qs, v)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_f_path', required=True)
    ap.add_argument('--pix_features_path', default=None)
    ap.add_argument('--mask_path', default=None)
    ap.add_argument('--obj_path', default='./meshes/guitar.obj')
    ap.add_argument('--name', default='guitar')
    ap.add_argument('--label', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--n_pairs', type=int, default=200000)
    ap.add_argument('--max_pixels', type=int, default=20000,
                    help='random subsample of in-mask pixels (the full mask is too big for a '
                         'dense NN search)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)

    mesh_f = torch.load(args.pred_f_path, map_location='cpu').detach().float()
    mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)
    faces = mesh.faces.detach().cpu().long()
    verts = mesh.vertices.detach().cpu().float()
    assert verts.shape[0] == mesh_f.shape[0], 'vertex/feature count mismatch'

    sets = [('mesh vertices', mesh_f)]
    if args.pix_features_path and args.mask_path:
        pix = torch.load(args.pix_features_path, map_location='cpu').float()
        mask = torch.load(args.mask_path, map_location='cpu')
        mask = mask.squeeze().bool()
        inmask = pix[mask]                                   # [P, C]
        if inmask.shape[0] > args.max_pixels:
            sel = torch.randperm(inmask.shape[0], generator=gen)[:args.max_pixels]
            inmask = inmask[sel]
        sets.append(('image pixels (in mask)', inmask))

    report = {'label': args.label, 'pred_f_path': args.pred_f_path, 'sets': {}}
    fig, axes = plt.subplots(len(sets), 4, figsize=(17.5, 4.0 * len(sets)), dpi=150,
                             squeeze=False)

    for row, (sname, f) in enumerate(sets):
        R = {}
        R['n_items'], R['dim'] = f.shape[0], f.shape[1]
        # 1 -- norms
        R['norms'], nrm = norm_stats(f)
        ax = axes[row][0]
        ax.hist(nrm.numpy(), bins=60, color='tab:blue')
        ax.set_title(f'1. feature norm\nmean {R["norms"]["mean"]:.4f}  cv {R["norms"]["cv"]:.2e}',
                     fontsize=9)
        ax.set_xlabel('||f||', fontsize=8); ax.tick_params(labelsize=7)
        # 2 -- pairwise cosine
        sims = pair_sims(f, args.n_pairs, gen)
        R['pair_cosine'] = describe(sims)
        ax = axes[row][1]
        ax.hist(sims.numpy(), bins=80, color='tab:red')
        ax.set_title(f'2. random-pair cosine\nmean {R["pair_cosine"]["mean"]:.3f}  '
                     f'sd {R["pair_cosine"]["std"]:.3f}', fontsize=9)
        ax.set_xlabel('cos(f_i, f_j)', fontsize=8); ax.tick_params(labelsize=7)
        # 3 -- PCA spectrum
        ev, R['spectrum'] = pca_spectrum(f)
        ax = axes[row][2]
        ax.plot(np.arange(1, min(len(ev), 512) + 1), ev[:512].numpy(), color='tab:green')
        ax.set_yscale('log'); ax.set_xlabel('component', fontsize=8)
        ax.set_title(f'3. PCA spectrum (log)\neff. rank {R["spectrum"]["effective_rank"]:.1f} '
                     f'of {R["spectrum"]["ambient_dim"]}   90% @ {R["spectrum"]["n_for_90pct"]}',
                     fontsize=9)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        # 4 -- nearest-neighbour distance
        nnd = nn_distance(f)
        R['nn_cosine_distance'] = describe(nnd)
        ax = axes[row][3]
        ax.hist(nnd.numpy(), bins=60, color='tab:purple')
        ax.set_title(f'4. nearest-neighbour distance\nmean {R["nn_cosine_distance"]["mean"]:.4f}  '
                     f'median {R["nn_cosine_distance"]["p50"]:.4f}', fontsize=9)
        ax.set_xlabel('1 - cos to nearest other', fontsize=8); ax.tick_params(labelsize=7)
        axes[row][0].set_ylabel(f'{sname}\n(n={R["n_items"]})', fontsize=9.5, weight='bold')
        report['sets'][sname] = R

    # 5 -- local smoothness on the mesh (its own figure row would be empty for pixels)
    ring, n_edges = ring_neighbor_sims(mesh_f, faces)
    rnd = pair_sims(mesh_f, args.n_pairs, gen)
    smooth = {'n_edges': n_edges, 'ring_cosine': describe(ring), 'random_cosine': describe(rnd),
              'gap_mean': ring.mean().item() - rnd.mean().item()}
    # correlation of euclidean mesh distance against feature cosine, the 2026-08-19 diagnostic
    a = torch.randint(0, verts.shape[0], (args.n_pairs,), generator=gen)
    b = torch.randint(0, verts.shape[0], (args.n_pairs,), generator=gen)
    keep = a != b; a, b = a[keep], b[keep]
    d = (verts[a] - verts[b]).norm(dim=-1)
    fn = torch.nn.functional.normalize(mesh_f, dim=-1)
    s = (fn[a] * fn[b]).sum(-1)
    smooth['corr_dist_vs_cosine'] = float(np.corrcoef(d.numpy(), s.numpy())[0, 1])
    order = torch.argsort(d); ss = s[order]
    smooth['cosine_by_distance_quintile'] = [ss[i * len(ss) // 5:(i + 1) * len(ss) // 5].mean().item()
                                             for i in range(5)]
    report['local_smoothness_mesh'] = smooth

    fig.suptitle(f'feature-space report -- {args.label}', fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = os.path.join(args.save_dir, 'feature_space_report.png')
    fig.savefig(png, bbox_inches='tight', facecolor='white')

    fig2, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.hist(rnd.numpy(), bins=80, alpha=0.65, label='random vertex pairs', color='0.5', density=True)
    ax.hist(ring.numpy(), bins=80, alpha=0.75, label='1-ring mesh neighbours', color='tab:orange',
            density=True)
    ax.set_title(f'5. local smoothness on the mesh\nring mean {ring.mean():.4f} vs random '
                 f'{rnd.mean():.4f}   gap {smooth["gap_mean"]:+.4f}', fontsize=10)
    ax.set_xlabel('cosine similarity', fontsize=9); ax.legend(fontsize=8); ax.tick_params(labelsize=8)
    png2 = os.path.join(args.save_dir, 'feature_space_local_smoothness.png')
    fig2.savefig(png2, bbox_inches='tight', facecolor='white')

    with open(os.path.join(args.save_dir, 'feature_space_report.json'), 'w') as fh:
        json.dump(report, fh, indent=2)

    # printed table, so the numbers land in the log too
    print(f'\n===== FEATURE-SPACE REPORT -- {args.label} =====')
    for sname, R in report['sets'].items():
        print(f'\n--- {sname}   n={R["n_items"]}  dim={R["dim"]}')
        n = R['norms']
        print(f'  1. norm            mean {n["mean"]:.4f}  sd {n["std"]:.2e}  '
              f'min {n["min"]:.4f}  max {n["max"]:.4f}  cv {n["cv"]:.2e}')
        c = R['pair_cosine']
        print(f'  2. pair cosine     mean {c["mean"]:+.4f}  sd {c["std"]:.4f}  '
              f'p1 {c["p1"]:+.3f}  p50 {c["p50"]:+.3f}  p99 {c["p99"]:+.3f}')
        s = R['spectrum']
        print(f'  3. effective rank  {s["effective_rank"]:.1f} / {s["ambient_dim"]}   '
              f'participation ratio {s["participation_ratio"]:.1f}   '
              f'90%/95%/99% var @ {s["n_for_90pct"]}/{s["n_for_95pct"]}/{s["n_for_99pct"]} comps')
        d = R['nn_cosine_distance']
        print(f'  4. NN distance     mean {d["mean"]:.4f}  median {d["p50"]:.4f}  '
              f'p1 {d["p1"]:.4f}  p99 {d["p99"]:.4f}  min {d["min"]:.4f}')
    m = report['local_smoothness_mesh']
    print(f'\n--- 5. local smoothness on the mesh   ({m["n_edges"]} edges)')
    print(f'  1-ring cosine      mean {m["ring_cosine"]["mean"]:.4f}  sd {m["ring_cosine"]["std"]:.4f}')
    print(f'  random-pair cosine mean {m["random_cosine"]["mean"]:.4f}  sd {m["random_cosine"]["std"]:.4f}')
    print(f'  gap (ring - random) {m["gap_mean"]:+.4f}')
    print(f'  corr(mesh distance, cosine) {m["corr_dist_vs_cosine"]:+.3f}')
    print('  cosine by distance quintile (near->far): '
          + ' / '.join(f'{v:.3f}' for v in m['cosine_by_distance_quintile']))
    print(f'\nsaved {png}\nsaved {png2}\nsaved {os.path.join(args.save_dir, "feature_space_report.json")}')


if __name__ == '__main__':
    main()
