"""Paper figures, horse LVIS 164135 (Tal, 2026-10-01). Clean versions of diag_seg_mapping.py (Leg Collision) and
plot_mu_annotated.py (per-part ellipses): no diagnostic text, one colour scheme, vector PDF out.
  legs     : per arm, the photo leg (fine seg --seg) coloured by the mesh leg its pixels land on, and the mapped
             points on the mesh legs in FRONT view, coloured by image row (root -> hoof). Both rows read SOFT-SNAP
             (nearest vertex to the soft point, as training reads it; Tal 2026-10-04), not argmax.
  ellipses : the learned per-part ellipses of one run on the photo (dashed circle = mu 1), + a SIDE render of the mesh
             with the tail stub in the tail segment's colour (Tal 2026-10-04: shows why the tail's ellipse differs).
  loss     : the relational loss on one pixel pair of --seg: image geodesic vs the mesh geodesic between the pair's
             soft-snap vertices, for ZS and for --run (Tal 2026-10-04). --pair picks the pair by row fraction.
Usage: python analysis/paper_fig_horse.py legs --runs LABEL=RUN_DIR ... --out f.pdf
       python analysis/paper_fig_horse.py ellipses --run RUN_DIR --out f.pdf
       python analysis/paper_fig_horse.py loss --run RUN_DIR --out f.pdf"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Patch
from matplotlib.collections import PolyCollection
from scipy.ndimage import distance_transform_edt
from modules.mapping_loss import soft_map_to_vertices
from modules.utils import loadmesh
from analysis.diag_mu_compare import part_of

plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'pdf.fonttype': 42, 'mathtext.fontset': 'cm'})
ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['legs', 'ellipses', 'loss'])
ap.add_argument('--runs', nargs='+'); ap.add_argument('--run'); ap.add_argument('--seg', type=int, default=3)
ap.add_argument('--pred_rel', default='alpha/alpha0.5/pred_f.pth'); ap.add_argument('--out', required=True)
ap.add_argument('--pair', type=float, nargs=2, default=[0.15, 0.8]); ap.add_argument('--tail_seg', type=int, default=2)
ap.add_argument('--top', choices=['leg', 'row', 'key', 'softw'], default='leg', help='legs, top row colour: mesh leg of the soft-snap vertex / image row (as the bottom row) / leg + a leading image-row key column / key column + the SOFT weight on the correct (x-) side, col 2 of diag_seg_mapping.py (Tal 2026-10-04: we work with soft)')
ap.add_argument('--min_frac', type=float, default=0.01); ap.add_argument('--radius', type=float, default=34)
a = ap.parse_args(); dev = 'cuda'


def load_cfg(run_dir):
    cfg = json.load(open(os.path.join(run_dir, 'run_config.json')))
    img = plt.imread(cfg['fine_seg_path'].replace('_fine_segments.pth', '.png'))
    fs = torch.load(cfg['fine_seg_path'], map_location='cpu').numpy()
    return cfg, img, fs


def render_mesh(ax, Vc, Fc, h, v, d, face_rgb=None, base=(.8, .8, .8)):
    """Orthographic shaded render: h/v = (axis, sign) of the horizontal/vertical screen axis, d = axis towards the viewer."""
    P = np.stack([Vc[:, h[0]] * h[1], Vc[:, v[0]] * v[1], Vc[:, d[0]] * d[1]], 1)[Fc]             # [F, 3, 3] screen coords
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    light = np.array([.3, .5, .8]); light /= np.linalg.norm(light)
    shade = .45 + .55 * np.abs(nrm @ light)
    rgb = np.tile(np.array(base), (len(Fc), 1)) if face_rgb is None else face_rgb
    order = np.argsort(P[:, :, 2].mean(1))                                                    # far -> near
    ax.add_collection(PolyCollection(P[order][:, :, :2], facecolors=np.clip(rgb[order] * shade[order, None], 0, 1),
                                     edgecolors='face', linewidths=.15, rasterized=True))
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def read_softsnap(pf, p, V, cfg):
    """Soft point (weighted mean of the top-K vertices) and its nearest vertex, per pixel."""
    vf = torch.load(pf, map_location=dev)
    _, tk, w = soft_map_to_vertices(p, vf, V, cfg['k_vertices_per_pixel'], return_topk=True, temperature=cfg['temperature'])
    soft = (w[..., None] * V[tk]).sum(1)
    return torch.cdist(soft, V).argmin(1).cpu().numpy(), tk.cpu().numpy(), w.detach().cpu().numpy()


if a.mode == 'legs':
    runs = [r.split('=', 1) for r in a.runs]
    cfg, img, fs = load_cfg(runs[-1][1])
    geo2 = torch.load(cfg['geo_table_2d_path'], map_location='cpu')
    fs = fs.copy(); fs[~geo2['geodesic_safe_mask'].numpy().astype(bool)] = 0
    V = loadmesh(dir=cfg['obj_path'], name='mesh', load_rings=False).vertices.to(dev); Vc = V.cpu().numpy()
    vpart = part_of(Vc); pix = torch.load(cfg['pix_dino_features_path'], map_location='cpu')
    rr, cc = np.nonzero(fs == a.seg); rown = (rr - rr.min()) / max(rr.max() - rr.min(), 1)
    r0, r1, c0, c1 = rr.min() - 10, rr.max() + 10, cc.min() - 25, cc.max() + 25
    COL = {1: np.array([.85, .15, .15]), 2: np.array([1., .65, .1])}       # front x+ (red), front x- (orange)
    other = np.array([.6, .6, .6])
    legs = Vc[:, 1] < -0.2
    n = len(runs); k0 = int(a.top in ('key', 'softw'))
    fig, axs = plt.subplots(2, n + k0, figsize=(1.15 * (n + k0), 3.6), gridspec_kw={'height_ratios': [2.2, 1]})

    def photo(ax, rgb, title):
        ov = np.zeros(fs.shape + (4,)); ov[rr, cc, :3] = rgb; ov[rr, cc, 3] = .9
        ax.imshow(img); ax.imshow(ov); ax.set_xlim(c0, c1); ax.set_ylim(r1, r0)
        ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    if k0:
        photo(axs[0, 0], plt.cm.viridis(rown)[:, :3], 'Image'); axs[1, 0].axis('off')
    for i, (lab, rd) in enumerate(runs, start=k0):
        pf = os.path.join(cfg['init_encoder_dir'], 'pred_f.pth') if rd == 'ZS' else os.path.join(rd, a.pred_rel)
        p = F.normalize(pix[rr, cc].float().to(dev), dim=-1)
        snap, tk, w = read_softsnap(pf, p, V, cfg)
        plab = vpart[snap]
        xneg = (w * np.isin(vpart[tk], [2, 4])).sum(1)             # soft weight on the x- (correct) side, front + hind legs
        print(lab, 'mean soft weight on the correct side %.3f' % xneg.mean(), 'pixels with >10%% on both sides %.3f' % ((xneg > .1) & (xneg < .9)).mean())
        photo(axs[0, i], plt.cm.viridis(rown)[:, :3] if a.top == 'row' else plt.cm.magma(xneg)[:, :3] if a.top == 'softw'
              else np.array([COL.get(k, other) for k in plab]), lab)
        print(lab, {k: round(float((plab == k).mean()), 3) for k in np.unique(plab)}, 'unique snap vertices', len(np.unique(snap)))
        ax = axs[1, i]
        for k, col in COL.items():
            m = legs & (vpart == k); ax.scatter(Vc[m, 0], Vc[m, 1], s=.6, color=(.5, .5, .5) if a.top == 'softw' else col, alpha=.25, linewidths=0)
        soft = (w[..., None] * Vc[tk]).sum(1)                       # the SOFT point (before the snap): shows the bridge
        ax.scatter(soft[:, 0], soft[:, 1], s=.8, c=plt.cm.viridis(rown), linewidths=0)
        ax.set_xlim(-.3, .3); ax.set_ylim(-.75, -.2); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    axs[1, k0].set_ylabel('mesh, front view', fontsize=7)
    if a.top == 'softw':
        cax = fig.add_axes([.42, .035, .36, .014])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='magma'), cax=cax, orientation='horizontal', ticks=[0, .5, 1])
        cb.ax.tick_params(labelsize=6, length=2, pad=1); cb.outline.set_linewidth(.4)
        fig.text(.41, .042, 'soft weight on the correct leg', ha='right', va='center', fontsize=7)
    elif a.top != 'row':
        fig.legend(handles=[Patch(color=COL[2], label='correct front leg'), Patch(color=COL[1], label='opposite front leg')],
                   loc='lower center', ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(.5, -.01))
    plt.tight_layout(rect=(0, .05, 1, 1), w_pad=.3, h_pad=.3)

elif a.mode == 'ellipses':
    cfg, img, fs = load_cfg(a.run)
    mu = torch.load(os.path.join(a.run, 'mu_aniso.pth'), map_location='cpu')
    tot = (fs > 0).sum()
    segs = [int(s) for s in np.unique(fs) if s > 0 and (fs == s).sum() / tot >= a.min_frac]
    rr, cc = np.nonzero(fs > 0); pad = 15; R2 = a.radius * float(mu["major"][segs].max())
    fig, (ax, axm) = plt.subplots(1, 2, figsize=(3.3, 1.9), gridspec_kw={'width_ratios': [1.25, 1]})
    ax.imshow(img); ov = np.zeros(fs.shape + (4,)); ov[fs > 0] = [0, 0, 0, .25]; ax.imshow(ov)
    cmap = plt.cm.tab10

    def draw(r, c, major, minor, ang, col, lw=1.4):
        # image angle 0 = along rows (vertical); matplotlib Ellipse angle is from the x (column) axis, CCW in display
        R = a.radius
        ax.add_patch(Circle((c, r), R, fill=False, ls=(0, (2, 2)), lw=.7, color='white'))
        ax.add_patch(Ellipse((c, r), 2 * R * major, 2 * R * minor, angle=-(90 - ang), fill=False, lw=lw, color=col))
    for j, s in enumerate(segs):
        m = fs == s
        ov2 = np.zeros(fs.shape + (4,)); ov2[m, :3] = cmap(j % 10)[:3]; ov2[m, 3] = .35; ax.imshow(ov2)
        r, c = np.unravel_index(np.argmax(distance_transform_edt(m)), m.shape)
        draw(r, c, float(mu['major'][s]), float(mu['minor'][s]), float(mu['major_angle_deg'][s]), cmap(j % 10))
    ax.set_xlim(max(0, cc.min() - pad), cc.max() + R2); ax.set_ylim(rr.max() + pad, max(0, rr.min() - pad)); ax.axis('off')
    # mesh, side view with the head on the left like the photo; tail stub (region of diag_tail_mapping.py) in the tail colour
    mesh = loadmesh(dir=cfg['obj_path'], name='mesh', load_rings=False)
    Vc, Fc = mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy()
    stub = (Vc[:, 2] < -.66) & (np.abs(Vc[:, 0]) < .03) & (Vc[:, 1] > 0) & (Vc[:, 1] < .3)
    frgb = np.tile(np.array([.8, .8, .8]), (len(Fc), 1))
    if a.tail_seg in segs: frgb[stub[Fc].any(1)] = cmap(segs.index(a.tail_seg) % 10)[:3]
    render_mesh(axm, Vc, Fc, (2, -1), (1, 1), (0, 1), face_rgb=frgb)
    axm.set_xlim(-Vc[:, 2].max() - .03, -Vc[:, 2].min() + .03); axm.set_ylim(Vc[:, 1].min() - .03, Vc[:, 1].max() + .03)
    print('segments', segs, 'major/minor', [(round(float(mu['major'][s]), 2), round(float(mu['minor'][s]), 2)) for s in segs])
    plt.tight_layout(pad=.1)

else:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    cfg, img, fs = load_cfg(a.run)
    geo2 = torch.load(cfg['geo_table_2d_path'], map_location='cpu'); geo3 = torch.load(cfg['geo_table_3d_path'], map_location='cpu')
    fs = fs.copy(); fs[~geo2['geodesic_safe_mask'].numpy().astype(bool)] = 0
    mesh = loadmesh(dir=cfg['obj_path'], name='mesh', load_rings=False)
    V = mesh.vertices.to(dev); Vc, Fc = V.cpu().numpy(), mesh.faces.cpu().numpy()
    pix = torch.load(cfg['pix_dino_features_path'], map_location='cpu')
    # the pair: p = the CENTRE pixel of --seg nearest to row fraction pair[0] (the 2D table has centre rows only),
    # q = the segment pixel on the leg's midline at row fraction pair[1]
    rr, cc = np.nonzero(fs == a.seg); r_lo, r_hi = rr.min(), rr.max()
    cen = geo2['coords'].numpy(); ci = np.nonzero(fs[cen[:, 0], cen[:, 1]] == a.seg)[0]
    mid = lambda r: int(np.median(cc[rr == r]))
    tp = r_lo + a.pair[0] * (r_hi - r_lo)
    ip = ci[np.argmin(np.abs(cen[ci, 0] - tp) + np.abs(cen[ci, 1] - np.array([mid(r) for r in cen[ci, 0]])))]
    pr, pc = cen[ip]; qr = int(round(r_lo + a.pair[1] * (r_hi - r_lo))); qc = mid(qr)
    diag2 = cfg.get('geo_diag_2d') or geo2['true_diameter_2d']; diag3 = cfg.get('geo_diag_3d') or float(geo3['diagonal'])
    d2 = float(geo2['table'][ip, geo2['pixel_to_node'][qr, qc]]) / diag2
    # mesh edge graph, for drawing the geodesic PATH (the numbers come from the training table)
    E = np.concatenate([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]]); L = np.linalg.norm(Vc[E[:, 0]] - Vc[E[:, 1]], axis=1)
    G = coo_matrix((L, (E[:, 0], E[:, 1])), shape=(len(Vc),) * 2).tocsr()
    feat = F.normalize(pix[[pr, qr], [pc, qc]].float().to(dev), dim=-1)
    rr2, cc2 = np.nonzero(fs > 0)
    fig, axs = plt.subplots(1, 3, figsize=(3.3, 1.75), gridspec_kw={'width_ratios': [1.15, 1, 1]})
    ax = axs[0]; ax.imshow(img); ov = np.zeros(fs.shape + (4,)); ov[fs == a.seg] = [1., .65, .1, .45]; ax.imshow(ov)
    ax.plot([pc, qc], [pr, qr], color='k', lw=1.2); ax.scatter([pc, qc], [pr, qr], s=14, c=['#1f77b4', '#d62728'], edgecolors='k', linewidths=.5, zorder=3)
    ax.annotate('$p$', (pc, pr), xytext=(5, 0), textcoords='offset points', va='center'); ax.annotate('$q$', (qc, qr), xytext=(5, 0), textcoords='offset points', va='center')
    ax.set_xlim(cc.min() - 45, cc.max() + 45); ax.set_ylim(r_hi + 12, r_lo - 30); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('image: $\\hat d_\\Omega = %.2f$' % d2, fontsize=7)
    for ax, (lab, pf, col) in zip(axs[1:], [('zero-shot', os.path.join(cfg['init_encoder_dir'], 'pred_f.pth'), '#d62728'),
                                             ('ours', os.path.join(a.run, a.pred_rel), '#2ca02c')]):
        snap, tk, w = read_softsnap(pf, feat, V, cfg)
        g3 = geo3['table'].numpy()
        d_snap = float(g3[snap[0], snap[1]]) / diag3
        d_exp = float((w[0][:, None] * w[1][None, :] * g3[tk[0]][:, tk[1]]).sum()) / diag3    # Eq. (expected geodesic)
        _, pred = dijkstra(G, directed=False, indices=snap[0], return_predecessors=True)
        path = [snap[1]]
        while path[-1] != snap[0] and pred[path[-1]] >= 0: path.append(pred[path[-1]])
        render_mesh(ax, Vc, Fc, (0, 1), (1, 1), (2, 1))
        ax.plot(Vc[path, 0], Vc[path, 1], color=col, lw=1.2, zorder=3)
        ax.scatter(Vc[snap, 0], Vc[snap, 1], s=14, c=['#1f77b4', '#d62728'], edgecolors='k', linewidths=.5, zorder=4)
        ax.set_xlim(-.3, .3); ax.set_ylim(-.75, .12)
        ax.set_title('%s: $\\hat D_{\\mathcal{M}} = %.2f$' % (lab, d_exp), fontsize=7)
        print(lab, 'p,q =', (int(pr), int(pc)), (qr, qc), 'd2 %.3f' % d2, 'D3 expected %.3f snap %.3f' % (d_exp, d_snap),
              'snap vertices', snap.tolist(), 'x', Vc[snap, 0].round(3).tolist())
    plt.tight_layout(pad=.2, w_pad=.4)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
plt.savefig(a.out, bbox_inches='tight', pad_inches=.02); plt.savefig(a.out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight', pad_inches=.02)
print('saved', a.out)
