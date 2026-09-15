"""Is the feature-RKD TEACHER geometric? (2026-09-09, Tal's question)

Spatial RKD's teacher is the 2D geodesic distance between two photo pixels -- fixed, external,
purely geometric. Feature RKD's teacher is the RELATION BETWEEN THE TWO PIXELS' PHOTO FEATURES.
This script puts both teachers on the SAME pairs and asks how much they agree.

The hypothesis being tested (mine, 2026-09-09): the photo features confuse pixels that LOOK alike
but are geodesically FAR -- the guitar's two horns, the symmetric body edges -- so feature RKD asks
the mesh side to reproduce a confusion rather than a geometry. If true, the mean-cosine-vs-geodesic
curve does NOT decay monotonically: it flattens or rises again at large distance.

Nothing is trained and nothing is written except the figure + a printed table.
"""
import argparse, os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features_path', required=True)
    ap.add_argument('--geo_table_2d_path', required=True)
    ap.add_argument('--num_targets', type=int, default=20000,
                    help='random on-object target pixels sampled per center (fixed seed)')
    ap.add_argument('--query_pixels', type=int, nargs='*', default=[151, 94, 571, 206, 820, 360, 820, 740],
                    help='row col row col ... -- panels showing the full similarity field')
    ap.add_argument('--far_frac', type=float, default=0.30,
                    help='"geodesically far" = geodesic distance above this fraction of the diameter')
    ap.add_argument('--save_path', required=True)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('loading features ...')
    F = torch.load(args.features_path, map_location='cpu')          # [H, W, C]
    H, W, C = F.shape
    t2d = torch.load(args.geo_table_2d_path, map_location='cpu')
    table, p2n, coords = t2d['table'], t2d['pixel_to_node'], t2d['coords']
    D = float(t2d['diagonal'])
    n_centers, n_nodes = table.shape
    print(f'  features {tuple(F.shape)}  table {tuple(table.shape)}  diagonal {D:.2f}px')

    # node index -> (row, col), by inverting pixel_to_node
    rows, cols = torch.where(p2n >= 0)
    node_of = p2n[rows, cols]
    node_rc = torch.empty(n_nodes, 2, dtype=torch.long)
    node_rc[node_of] = torch.stack([rows, cols], 1)

    g = torch.Generator().manual_seed(0)
    tgt = torch.randperm(n_nodes, generator=g)[:args.num_targets]
    tgt_rc = node_rc[tgt]

    def feats(rc):
        f = F[rc[:, 0], rc[:, 1]].to(dev).float()
        return torch.nn.functional.normalize(f, dim=1)

    Fc, Ft = feats(coords), feats(tgt_rc)                            # [n_centers, C], [M, C]
    print('computing cosine similarities ...')
    sim = (Fc @ Ft.T).cpu()                                          # [n_centers, M]
    geo = (table[:, tgt].float() / D)                                # [n_centers, M], 0..1

    ok = torch.isfinite(geo) & (geo >= 0)
    s, d = sim[ok].numpy(), geo[ok].numpy()
    print(f'  {s.size:,} valid (center, target) pairs')

    pear = float(np.corrcoef(s, d)[0, 1])
    rs = np.argsort(np.argsort(s)); rd = np.argsort(np.argsort(d))
    spear = float(np.corrcoef(rs, rd)[0, 1])

    # --- the curve that answers the question -------------------------------------------------
    edges = np.linspace(0, 1, 41)
    idx = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
    mean = np.array([s[idx == b].mean() if (idx == b).sum() > 50 else np.nan
                     for b in range(len(edges) - 1)])
    std = np.array([s[idx == b].std() if (idx == b).sum() > 50 else np.nan
                    for b in range(len(edges) - 1)])
    ctr = 0.5 * (edges[:-1] + edges[1:])

    # --- how far away does a center's MOST SIMILAR other pixel sit? ---------------------------
    far = geo > args.far_frac
    sim_far = sim.clone(); sim_far[~far] = -2.0
    far_max, far_arg = sim_far.max(dim=1)
    near = geo < 0.05
    sim_near = sim.clone(); sim_near[~near] = float('nan')
    near_mean = torch.nanmean(sim_near, dim=1)
    valid = far.any(dim=1) & near.any(dim=1)
    fm, nm = far_max[valid].numpy(), near_mean[valid].numpy()
    # geodesic distance to the globally most similar target (self-neighbourhood excluded)
    sim_ex = sim.clone(); sim_ex[geo < 0.02] = -2.0
    best = sim_ex.argmax(dim=1)
    best_geo = geo[torch.arange(n_centers), best].numpy()

    print('\n=== TEACHER AGREEMENT (photo feature cosine  vs  2D geodesic distance) ===')
    print(f'  Pearson  r = {pear:+.4f}   (want strongly NEGATIVE: similar => close)')
    print(f'  Spearman r = {spear:+.4f}')
    print(f'\n  mean cosine at   0-5% of diameter : {np.nanmean(mean[:2]):.4f}')
    print(f'  mean cosine at  45-55%            : {np.nanmean(mean[18:22]):.4f}')
    print(f'  mean cosine at  90-100%           : {np.nanmean(mean[36:]):.4f}')
    print(f'\n  centers whose best FAR match beats their mean NEAR similarity: '
          f'{100.0*(fm > nm).mean():.1f}%  ({int((fm>nm).sum())}/{fm.size})')
    print(f'  median geodesic distance to the most-similar non-local pixel: '
          f'{100*np.median(best_geo):.1f}% of diameter')
    for q in (50, 75, 90):
        print(f'    p{q}: {100*np.percentile(best_geo, q):.1f}%')

    # --- figure ------------------------------------------------------------------------------
    qp = [(args.query_pixels[i], args.query_pixels[i + 1]) for i in range(0, len(args.query_pixels), 2)]
    nq = len(qp)
    fig = plt.figure(figsize=(4.6 * max(3, nq), 8.6))
    gs = fig.add_gridspec(2, max(3, nq))

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(100 * ctr, mean, color='C0', lw=2)
    ax.fill_between(100 * ctr, mean - std, mean + std, color='C0', alpha=0.18)
    ax.set_xlabel('2D geodesic distance (% of photo diameter)')
    ax.set_ylabel('mean photo-feature cosine')
    ax.set_title(f'the two teachers on the same pairs\nPearson {pear:+.3f}, Spearman {spear:+.3f}')
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(100 * best_geo, bins=50, color='C3')
    ax.axvline(100 * args.far_frac, color='k', ls=':', lw=1)
    ax.set_xlabel('geodesic distance to the MOST SIMILAR non-local pixel (% of diameter)')
    ax.set_ylabel('# centers')
    ax.set_title('where each center\'s best feature match actually lies\n'
                 f'median {100*np.median(best_geo):.1f}%')
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(nm, fm, s=4, alpha=0.25, color='C2')
    lo = float(min(nm.min(), fm.min())); hi = float(max(nm.max(), fm.max()))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_xlabel('mean cosine to GEODESICALLY NEAR pixels (<5%)')
    ax.set_ylabel(f'best cosine to FAR pixels (>{100*args.far_frac:.0f}%)')
    ax.set_title(f'points above the line = the teacher prefers a FAR pixel\n'
                 f'{100*(fm > nm).mean():.1f}% of centers')
    ax.grid(alpha=0.3)

    # full similarity field for a few query pixels
    on = (p2n >= 0)
    Fall = torch.nn.functional.normalize(F[on].to(dev).float(), dim=1)
    for i, (r, c) in enumerate(qp):
        q = torch.nn.functional.normalize(F[r, c].to(dev).float()[None], dim=1)
        field = torch.full((H, W), float('nan'))
        field[on] = (Fall @ q.T).squeeze(1).cpu()
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(field.numpy(), cmap='inferno', vmin=0, vmax=1)
        ax.plot(c, r, 'o', mfc='none', mec='cyan', ms=13, mew=2.5)
        ax.set_title(f'query ({r}, {c})', fontsize=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle('Is the feature-RKD teacher geometric?  photo-feature cosine vs 2D geodesic distance, '
                 f'{s.size:,} pairs from {n_centers} centers', fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    fig.savefig(args.save_path, dpi=110, bbox_inches='tight')
    print(f'\nsaved {args.save_path}')


if __name__ == '__main__':
    main()
