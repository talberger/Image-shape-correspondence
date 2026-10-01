"""Why did one segment's DIRECTIONAL mu stay a circle? (Tal, 2026-09-22 -- horse lvis164135 seg 2, the tail)

For ONE fine segment of a finished segment_aniso run, reproduce what the loss saw and answer:
  (1) SAMPLING  -- shape of the segment (PCA elongation/axis), how many of the 2048 fixed centers fall in
                   it, and the DIRECTION histogram of centre->neighbour offsets (neighbours = uniform over
                   the same segment, exactly as sample_in_segment_neighbors draws them).
  (2) 3D TARGET -- where the segment's pixels land on the mesh (phase-1 and final encoder), and the
                   per-direction ratio d3D/d2D the ellipse is trying to fit. s(u) = sqrt(u^T M u) fits
                   norm_d_3D / norm_d_2D per direction u; if that ratio is flat in angle, a circle is
                   the correct answer and the problem is upstream (the mapping), not the ellipse.
Outputs a figure + a json next to --out_dir.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.mapping_loss import soft_map_to_vertices
from modules.relational_loss import expected_geodesic_distance_3d
from modules.utils import loadmesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--seg', type=int, required=True)
    ap.add_argument('--n_nbr', type=int, default=16)
    ap.add_argument('--n_bins', type=int, default=12)
    ap.add_argument('--out_dir', required=True)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run_dir, 'run_config.json')))
    dev = 'cuda'
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(0)

    geo2 = torch.load(cfg['geo_table_2d_path'], map_location='cpu')
    coords, table2, p2n = geo2['coords'].long(), geo2['table'], geo2['pixel_to_node'].long()
    safe = geo2['geodesic_safe_mask'].numpy().astype(bool)
    fs = torch.load(cfg['fine_seg_path'], map_location='cpu').numpy().copy()
    fs[~safe] = 0
    img = plt.imread(os.path.join(os.path.dirname(cfg['fine_seg_path']),
                                  os.path.basename(cfg['fine_seg_path']).replace('_fine_segments.pth', '.png')))
    g3 = torch.load(cfg['geo_table_3d_path'], map_location='cpu')
    table3 = g3['table'].to(dev)
    mu2, mu3 = cfg['geo_diag_2d'], float(g3['diagonal'])

    # ---- (1) segment shape + sampling
    rr, cc = np.nonzero(fs == a.seg)
    P = np.stack([rr, cc], 1).astype(float)
    ev, evec = np.linalg.eigh(np.cov((P - P.mean(0)).T))
    elong = float(np.sqrt(ev[1] / ev[0]))
    ax_ang = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180)   # 0 = vertical (rows)
    seg_of_center = torch.from_numpy(fs[coords[:, 0], coords[:, 1]])
    cidx = torch.nonzero(seg_of_center == a.seg).flatten()
    C = coords[cidx]
    print('seg %d: %d px, PCA elongation %.2f, long axis %.1f deg (0=vertical); %d of %d centres in it'
          % (a.seg, len(rr), elong, ax_ang, len(cidx), len(coords)))
    # neighbours: uniform over the segment's pixels, 16 per centre, redrawn 300x like training -> take 300 draws
    seg_pix = torch.from_numpy(np.stack([rr, cc], 1)).long()
    draws = 300
    nb = seg_pix[torch.randint(len(seg_pix), (len(C), draws * a.n_nbr))]            # [n, D*k, 2]
    off = (nb - C[:, None]).float()
    keep = off.norm(dim=-1) > 0
    ang = (torch.rad2deg(torch.atan2(off[..., 1], off[..., 0])) % 180)            # 0 = vertical
    d2 = table2[cidx[:, None], p2n[nb[..., 0], nb[..., 1]]]                         # 2D geodesic px
    eucl = off.norm(dim=-1)

    # ---- (2) 3D: map every pixel of the segment + centres, phase-1 vs final
    pix = torch.load(cfg['pix_dino_features_path'], map_location='cpu')
    mesh = loadmesh(dir=cfg['obj_path'], name=cfg.get('name', 'horse'), load_rings=False)
    V = mesh.vertices.to(dev)
    enc = {'phase1': torch.load(os.path.join(cfg['init_encoder_dir'], 'pred_f.pth'), map_location=dev),
           'final': torch.load(os.path.join(a.run_dir, 'pred_f.pth'), map_location=dev)}
    mu_an = torch.load(os.path.join(a.run_dir, 'mu_aniso.pth'), map_location='cpu')
    p = mu_an['mu_aniso'][a.seg]
    L = torch.tensor([[p[0].exp(), 0.], [p[2], p[1].exp()]])
    res = {'seg': a.seg, 'n_pixels': int(len(rr)), 'pca_elongation': elong, 'pca_long_axis_deg': ax_ang,
           'n_centres': int(len(cidx)),
           'learned': {'major': float(mu_an['major'][a.seg]), 'minor': float(mu_an['minor'][a.seg]),
                       'angle': float(mu_an['major_angle_deg'][a.seg])}}
    bins = np.linspace(0, 180, a.n_bins + 1)
    mid = (bins[:-1] + bins[1:]) / 2
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))
    # panel: image + segment + centres
    axs[0, 0].imshow(img)
    ov = np.zeros(fs.shape + (4,)); ov[fs == a.seg] = [1, 0, 1, 0.45]
    axs[0, 0].imshow(ov)
    axs[0, 0].scatter(C[:, 1], C[:, 0], s=6, c='yellow', edgecolors='k', linewidths=0.3)
    axs[0, 0].set_title('seg %d (%d px, elong %.2f, axis %.0f deg), %d centres' % (a.seg, len(rr), elong, ax_ang, len(C)))
    # panel: direction histogram of pairs, weighted by count, and by 2D distance
    a_np, k_np = ang[keep].numpy(), keep
    axs[0, 1].hist(a_np, bins=bins, alpha=0.6, label='pairs (count)')
    axs[0, 1].set_xlabel('pair direction (deg, 0=vertical, 90=horizontal)'); axs[0, 1].legend()
    axs[0, 1].set_title('centre->neighbour directions (300 steps x 16 nbrs)')
    colors = {'phase1': 'tab:blue', 'final': 'tab:red'}
    for name, vf in enc.items():
        vfn = vf  # soft_map_to_vertices normalises as in training
        allc = torch.cat([C, nb[:, :draws * a.n_nbr:draws].reshape(-1, 2)], 0)  # centres + a 16-nbr subset for d3D
        pf = F.normalize(pix[allc[:, 0], allc[:, 1]].float().to(dev), dim=-1)
        _, tk, w = soft_map_to_vertices(pf, vfn, V, cfg['k_vertices_per_pixel'], return_topk=True,
                                        temperature=cfg['temperature'])
        n = len(C)
        # larger sample for the ratio: chunk the full draw set
        d3_all = []
        for j in range(0, draws * a.n_nbr, 512):
            sub = nb[:, j:j + 512]
            pfn = F.normalize(pix[sub[..., 0].reshape(-1), sub[..., 1].reshape(-1)].float().to(dev), dim=-1)
            _, tkn, wn = soft_map_to_vertices(pfn, vfn, V, cfg['k_vertices_per_pixel'], return_topk=True,
                                              temperature=cfg['temperature'])
            d3_all.append(expected_geodesic_distance_3d(
                tk[:n], w[:n], tkn.view(n, -1, tkn.shape[-1]), wn.view(n, -1, wn.shape[-1]), table3).cpu())
        d3 = torch.cat(d3_all, 1)
        ratio = (d3 / mu3) / (d2 / mu2).clamp_min(1e-9)
        # per-direction least-squares scale s = sum(b a)/sum(a a)? loss is huber(b/s, a) -> s = sum b^2/sum ab
        A, B = (d2 / mu2)[keep].numpy(), (d3 / mu3)[keep].numpy()
        sbin, mbin = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (a_np >= lo) & (a_np < hi)
            sbin.append(float((B[m] ** 2).sum() / max((A[m] * B[m]).sum(), 1e-12)) if m.sum() > 20 else np.nan)
            mbin.append(float(np.median(B[m] / np.maximum(A[m], 1e-9))) if m.sum() > 20 else np.nan)
        axs[0, 2].plot(mid, sbin, 'o-', color=colors[name], label='%s: LS scale per direction' % name)
        axs[0, 2].plot(mid, mbin, 'x--', color=colors[name], alpha=0.6, label='%s: median d3/d2' % name)
        # where on the mesh do the segment's pixels go (argmax vertex of every seg pixel)
        pfs = F.normalize(pix[seg_pix[:, 0], seg_pix[:, 1]].float().to(dev), dim=-1)
        top = (pfs @ F.normalize(vf, dim=-1).T).argmax(1).cpu()
        uv, cnt = np.unique(top.numpy(), return_counts=True)
        Vc = V.cpu().numpy()
        hit = Vc[top.numpy()]
        spread3 = float(table3[top[:, None].to(dev), top[None, :2000].to(dev)].mean().cpu() / mu3)
        res[name] = {'distinct_argmax_vertices': int(len(uv)), 'top10_share': float(np.sort(cnt)[::-1][:10].sum() / cnt.sum()),
                     'mean_pairwise_geo3d_norm': spread3,
                     'mean_pairwise_geo2d_norm': float((d2[keep] / mu2).mean()),
                     'ls_scale_per_dir': dict(zip([round(x) for x in mid], sbin)),
                     'hit_xyz_mean': hit.mean(0).tolist(), 'hit_xyz_std': hit.std(0).tolist()}
        print('%s: %d distinct argmax vertices (top-10 share %.2f); mean 3D geo spread (norm) %.4f vs 2D %.4f'
              % (name, len(uv), res[name]['top10_share'], spread3, res[name]['mean_pairwise_geo2d_norm']))
        print('   LS scale per direction:', ' '.join('%3.0f:%.3f' % (m_, s_) for m_, s_ in zip(mid, sbin)))
        r = 1 if name == 'phase1' else 1
        axv = axs[1, 0 if name == 'phase1' else 1]
        # 3 orthographic views collapsed into one: side view (x vs y) + top (x vs z) of the mesh, hits in colour
        axv.scatter(Vc[:, 0], Vc[:, 1], s=1, c='lightgray')
        axv.scatter(hit[:, 0], hit[:, 1], s=4, c=colors[name], alpha=0.3)
        axv.set_aspect('equal'); axv.set_title('%s: argmax vertex of every seg-%d pixel (mesh x/y view)' % (name, a.seg))
    # learned ellipse
    th = np.radians(mid)
    u = np.stack([np.cos(th), np.sin(th)], 1)  # (row, col)
    s_learn = np.linalg.norm(u @ L.numpy(), axis=1)
    axs[0, 2].plot(mid, s_learn, 'k-', lw=2, label='learned ellipse s(u)')
    axs[0, 2].set_xlabel('pair direction (deg)'); axs[0, 2].set_ylabel('scale'); axs[0, 2].legend(fontsize=8)
    axs[0, 2].set_title('what the ellipse fits: d3D/d2D by direction')
    # 2D geodesic vs euclid (thin segment: geodesic ~ euclid?)
    axs[1, 2].scatter(eucl[keep][:5000] / mu2, d2[keep][:5000] / mu2, s=2, c=a_np[:5000], cmap='hsv')
    axs[1, 2].set_xlabel('2D euclid (norm)'); axs[1, 2].set_ylabel('2D geodesic (norm)')
    axs[1, 2].set_title('2D pair distances (colour = direction)')
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'diag_seg%d.png' % a.seg), dpi=90)
    json.dump(res, open(os.path.join(a.out_dir, 'diag_seg%d.json' % a.seg), 'w'), indent=1)
    print('saved', a.out_dir)


if __name__ == '__main__':
    main()
