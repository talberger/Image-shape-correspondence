"""Same segment, several runs: why does the ellipse form in one run and not in another? (Tal, 2026-09-22)

For the pixels of ONE fine segment, per run (phase-1 + each --run_dirs):
  - where every pixel lands on the mesh (argmax vertex) and WHICH PART that is (4 legs / body+head);
    painted back onto the photo, so a leg mapped to the wrong leg shows up as a colour split
  - the per-direction d3D/d2D the ellipse fits (LS scale per direction bin), for ALL pairs and for
    pairs whose two ends land on the SAME part only (removes the cross-leg pairs)
  - the learned ellipse s(u) (per-seg: that segment's (a,b,c); grid: averaged over the segment's pixels)
Pairs = centres of the fixed 2048 set lying in the segment x uniform same-segment neighbours (as training).
Horse part labels assume the SHIC horse frame: y up, head at +z, legs = y < -0.3.
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
from phase2.train_relational_loss import grid_center_params, aniso_scale_from_params

PARTS = ['body/head', 'front x+', 'front x-', 'hind x+', 'hind x-']
PCOL = np.array([[.75, .75, .75], [.85, .1, .1], [1., .6, .1], [.1, .3, .9], [.2, .8, .9]])


def part_of(V):
    lab = np.zeros(len(V), int)
    leg = V[:, 1] < -0.3
    front, xp = V[:, 2] > 0, V[:, 0] > 0
    lab[leg & front & xp], lab[leg & front & ~xp] = 1, 2
    lab[leg & ~front & xp], lab[leg & ~front & ~xp] = 3, 4
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dirs', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--seg', type=int, required=True)
    ap.add_argument('--n_bins', type=int, default=12)
    ap.add_argument('--draws', type=int, default=2000, help='neighbours per centre')
    ap.add_argument('--out_dir', required=True)
    a = ap.parse_args()
    dev = 'cuda'
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(0)
    cfg = json.load(open(os.path.join(a.run_dirs[0], 'run_config.json')))

    geo2 = torch.load(cfg['geo_table_2d_path'], map_location='cpu')
    coords, table2, p2n = geo2['coords'].long(), geo2['table'], geo2['pixel_to_node'].long()
    fs = torch.load(cfg['fine_seg_path'], map_location='cpu').numpy().copy()
    fs[~geo2['geodesic_safe_mask'].numpy().astype(bool)] = 0
    img = plt.imread(cfg['fine_seg_path'].replace('_fine_segments.pth', '.png'))
    g3 = torch.load(cfg['geo_table_3d_path'], map_location='cpu')
    table3, mu2, mu3 = g3['table'].to(dev), cfg['geo_diag_2d'], float(g3['diagonal'])
    V = loadmesh(dir=cfg['obj_path'], name='mesh', load_rings=False).vertices.to(dev)
    vpart = part_of(V.cpu().numpy())
    pix = torch.load(cfg['pix_dino_features_path'], map_location='cpu')

    rr, cc = np.nonzero(fs == a.seg)
    seg_pix = torch.from_numpy(np.stack([rr, cc], 1)).long()
    cidx = torch.nonzero(torch.from_numpy(fs[coords[:, 0], coords[:, 1]]) == a.seg).flatten()
    C = coords[cidx]
    n = len(C)
    nb = seg_pix[torch.randint(len(seg_pix), (n, a.draws))]
    off = (nb - C[:, None]).float()
    keep = (off.norm(dim=-1) > 0).numpy()
    ang = (torch.rad2deg(torch.atan2(off[..., 1], off[..., 0])) % 180).numpy()
    d2 = (table2[cidx[:, None], p2n[nb[..., 0], nb[..., 1]]] / mu2).numpy()
    bins = np.linspace(0, 180, a.n_bins + 1)
    mid = (bins[:-1] + bins[1:]) / 2
    r0, r1, c0, c1 = rr.min() - 15, rr.max() + 15, cc.min() - 15, cc.max() + 15

    runs = [('phase1', os.path.join(cfg['init_encoder_dir'], 'pred_f.pth'), None)] + \
           [(l, os.path.join(d, 'pred_f.pth'), d) for l, d in zip(a.labels, a.run_dirs)]
    R = len(runs)
    fig, axs = plt.subplots(R, 3, figsize=(16, 4.6 * R))
    out = {'seg': a.seg, 'n_pixels': int(len(rr)), 'n_centres': n}
    for i, (name, pf_path, rdir) in enumerate(runs):
        vf = torch.load(pf_path, map_location=dev)
        pfs = F.normalize(pix[seg_pix[:, 0], seg_pix[:, 1]].float().to(dev), dim=-1)
        top = (pfs @ F.normalize(vf, dim=-1).T).argmax(1).cpu().numpy()
        plab = vpart[top]
        frac = np.bincount(plab, minlength=5) / len(plab)
        # (a) photo crop, pixels coloured by the mesh part they land on
        ov = np.zeros(fs.shape + (4,))
        ov[rr, cc, :3], ov[rr, cc, 3] = PCOL[plab], 0.85
        axs[i, 0].imshow(img); axs[i, 0].imshow(ov)
        axs[i, 0].set_xlim(c0, c1); axs[i, 0].set_ylim(r1, r0)
        axs[i, 0].set_title('%s: part hit per pixel\n%s' % (name, '  '.join(
            '%s %.0f%%' % (PARTS[k], 100 * frac[k]) for k in range(5) if frac[k] > 0.01)), fontsize=9)
        # (b) hits on the mesh, side view (z, y)
        Vc, hit = V.cpu().numpy(), V.cpu().numpy()[top]
        axs[i, 1].scatter(Vc[:, 2], Vc[:, 1], s=1, c='lightgray')
        axs[i, 1].scatter(hit[:, 2], hit[:, 1], s=5, c=PCOL[plab], alpha=0.4)
        axs[i, 1].set_aspect('equal'); axs[i, 1].set_title('%s: hits, side view (z right = head)' % name, fontsize=9)
        # (c) d3D/d2D by direction
        allpix = torch.cat([C, nb.reshape(-1, 2)], 0)
        tks, ws = [], []
        for j in range(0, len(allpix), 20000):
            p_ = F.normalize(pix[allpix[j:j + 20000, 0], allpix[j:j + 20000, 1]].float().to(dev), dim=-1)
            _, tk, w = soft_map_to_vertices(p_, vf, V, cfg['k_vertices_per_pixel'], return_topk=True,
                                            temperature=cfg['temperature'])
            tks.append(tk); ws.append(w)
        tk, w = torch.cat(tks), torch.cat(ws)
        d3 = []
        for j in range(0, a.draws, 250):
            sl = slice(n + 0, None)
            tkn = tk[n:].view(n, a.draws, -1)[:, j:j + 250]; wn = w[n:].view(n, a.draws, -1)[:, j:j + 250]
            d3.append(expected_geodesic_distance_3d(tk[:n], w[:n], tkn, wn, table3).cpu())
        d3 = (torch.cat(d3, 1) / mu3).numpy()
        # same-part mask: argmax part of centre vs neighbour
        cpart = vpart[tk[:n, 0].cpu().numpy()]          # top-1 of the soft map == argmax
        npart = vpart[tk[n:, 0].cpu().numpy()].reshape(n, a.draws)
        same = cpart[:, None] == npart
        prof = {}
        for tag, m_all, sty in (('all pairs', keep, 'o-'), ('same-part pairs', keep & same, 's--')):
            s = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = m_all & (ang >= lo) & (ang < hi)
                s.append(float((d3[m] ** 2).sum() / max((d2[m] * d3[m]).sum(), 1e-12)) if m.sum() > 30 else np.nan)
            prof[tag] = s
            axs[i, 2].plot(mid, s, sty, label=tag)
        if rdir is not None:
            m = torch.load(os.path.join(rdir, 'mu_aniso.pth'), map_location='cpu')
            th = np.radians(mid)
            U = torch.tensor(np.stack([np.cos(th), np.sin(th)], 1), dtype=torch.float32)   # (row, col)
            if m.get('scope') == 'grid_aniso' or m.get('grid'):
                P = grid_center_params(m['mu_aniso'], seg_pix, m['grid_bbox'], m['grid'])   # [npix, 3]
                sl = aniso_scale_from_params(P, torch.zeros(len(P), 2), U[None].expand(len(P), -1, -1)).mean(0)
                lab = 'learned grid ellipse (mean over seg px)'
            else:
                P = m['mu_aniso'][a.seg:a.seg + 1]
                sl = aniso_scale_from_params(P, torch.zeros(1, 2), U[None])[0]
                lab = 'learned per-seg ellipse'
            axs[i, 2].plot(mid, sl.detach().numpy(), 'k-', lw=2, label=lab)
            prof['learned'] = sl.detach().tolist()
        axs[i, 2].set_xlabel('pair direction (deg, 0=vertical)'); axs[i, 2].legend(fontsize=8)
        axs[i, 2].set_title('%s: d3D/d2D by direction (same-part pairs %.0f%%)' % (name, 100 * same[keep].mean()), fontsize=9)
        out[name] = {'part_frac': dict(zip(PARTS, frac.tolist())), 'same_part_pair_frac': float(same[keep].mean()),
                     'profile': prof, 'distinct_argmax': int(len(np.unique(top)))}
        print('%-10s parts %s | same-part pairs %.2f | distinct vtx %d' % (
            name, ' '.join('%.2f' % f for f in frac), same[keep].mean(), len(np.unique(top))))
        for k, v in prof.items():
            print('   %-16s %s' % (k, ' '.join('%.2f' % x for x in v)))
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'compare_seg%d.png' % a.seg), dpi=85)
    json.dump(out, open(os.path.join(a.out_dir, 'compare_seg%d.json' % a.seg), 'w'), indent=1)
    print('saved', a.out_dir)


if __name__ == '__main__':
    main()
