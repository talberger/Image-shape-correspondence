"""Where do the TAIL pixels (one fine segment) land on the mesh, per reading? (Tal, 2026-09-29)
Same idea as diag_seg_mapping.py (the leg study), but the mesh tail is a tiny stub at the rear top, so the views are
SIDE (z, y) and REAR (x, y) instead of front. Per reading: photo crop coloured by position ALONG the tail (0 = root,
1 = tip; PCA major axis of the segment's pixels, root = the end nearer the object centroid), the photo coloured by the
mesh REGION of the argmax vertex, and the soft-snap points on the mesh coloured by the along-tail position.
Column 2 = region of the SOFT-SNAP vertex (nearest vertex to the soft point, as training reads it); argmax kept in the json.
Regions (horse mesh, z = body axis, head at +z): tail stub = z < -0.66, |x| < 0.03, 0 < y < 0.3; rump = z < -0.45, y > -0.25;
hind legs = z < 0, y <= -0.25; front legs = z >= 0, y <= -0.25; rest of body/head = everything else.
Readings: --readings LABEL=PRED_F_PATH ... (phase-1 pred_f.pth, a step checkpoint export, the final alpha export)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from modules.mapping_loss import soft_map_to_vertices
from modules.utils import loadmesh

ap = argparse.ArgumentParser()
ap.add_argument('--run_config', required=True, help='run_config.json of any run on this instance (paths)')
ap.add_argument('--readings', nargs='+', required=True, metavar='LABEL=PRED_F')
ap.add_argument('--seg', type=int, default=2); ap.add_argument('--out', required=True)
ap.add_argument('--fs', type=int, default=16)
a = ap.parse_args(); dev = 'cuda'
plt.rcParams.update({'xtick.labelsize': a.fs - 4, 'ytick.labelsize': a.fs - 4, 'axes.labelsize': a.fs - 2})
cfg = json.load(open(a.run_config))
geo2 = torch.load(cfg['geo_table_2d_path'], map_location='cpu')
fs = torch.load(cfg['fine_seg_path'], map_location='cpu').numpy().copy()
fs[~geo2['geodesic_safe_mask'].numpy().astype(bool)] = 0
img = plt.imread(cfg['fine_seg_path'].replace('_fine_segments.pth', '.png'))
V = loadmesh(dir=cfg['obj_path'], name='mesh', load_rings=False).vertices.to(dev); Vc = V.cpu().numpy()
pix = torch.load(cfg['pix_dino_features_path'], map_location='cpu')

REG = ['tail stub', 'rump', 'hind legs', 'front legs', 'body/head']
RCOL = np.array([[1, 0, 1], [1, .6, 0], [0, .6, 1], [0, .8, .2], [.6, .6, .6]])
x, y, z = Vc[:, 0], Vc[:, 1], Vc[:, 2]
vreg = np.full(len(Vc), 4)
vreg[(z >= 0) & (y <= -.25)] = 3; vreg[(z < 0) & (y <= -.25)] = 2
vreg[(z < -.45) & (y > -.25)] = 1; vreg[(z < -.66) & (np.abs(x) < .03) & (y > 0) & (y < .3)] = 0

rr, cc = np.nonzero(fs == a.seg)
X = np.stack([rr, cc], 1).astype(float); mu = X.mean(0)
e1 = np.linalg.eigh(np.cov((X - mu).T))[1][:, 1]; t = (X - mu) @ e1
obj = np.stack(np.nonzero(fs > 0), 1).mean(0)
if abs(((X[np.argmin(t)] - obj) ** 2).sum()) > abs(((X[np.argmax(t)] - obj) ** 2).sum()): t = -t     # root = end nearer the body
tn = (t - t.min()) / max(t.max() - t.min(), 1e-9); cmap = plt.cm.viridis
r0, r1, c0, c1 = rr.min() - 15, rr.max() + 15, cc.min() - 15, cc.max() + 15

runs = [s.split('=', 1) for s in a.readings]
fig, axs = plt.subplots(len(runs), 4, figsize=(30, 7.5 * len(runs)), squeeze=False, gridspec_kw={'width_ratios': [1.3, 1.3, 2.4, 1.2]})
stats = {'seg': a.seg, 'n_pixels': int(len(rr)), 'n_tail_stub_vertices': int((vreg == 0).sum())}
rear = z < -.35
for i, (name, pf) in enumerate(runs):
    vf = torch.load(pf, map_location=dev)
    p = F.normalize(pix[rr, cc].float().to(dev), dim=-1)
    _, tk, w = soft_map_to_vertices(p, vf, V, cfg['k_vertices_per_pixel'], return_topk=True, temperature=cfg['temperature'])
    tk, w = tk.cpu().numpy(), w.detach().cpu().numpy()
    top = tk[:, 0]; afrac = np.bincount(vreg[top], minlength=5) / len(top)
    soft = (w[..., None] * Vc[tk]).sum(1)
    snap = torch.cdist(torch.from_numpy(soft).float().to(dev), V).argmin(1).cpu().numpy()   # soft-SNAP vertex (what training reads)
    lab = vreg[snap]; frac = np.bincount(lab, minlength=5) / len(lab)
    stats[name] = {'pixels_softsnap_region': dict(zip(REG, frac.round(3).tolist())),
                   'pixels_argmax_region': dict(zip(REG, afrac.round(3).tolist())), 'unique_softsnap_vertices': int(len(np.unique(snap))),
                   'unique_argmax_vertices': int(len(np.unique(top))),
                   'soft_point_mean_xyz': soft.mean(0).round(3).tolist(), 'soft_point_std_xyz': soft.std(0).round(3).tolist(),
                   'corr(along-tail, soft z)': round(float(np.corrcoef(tn, soft[:, 2])[0, 1]), 3),
                   'corr(along-tail, soft y)': round(float(np.corrcoef(tn, soft[:, 1])[0, 1]), 3)}
    ov = np.zeros(fs.shape + (4,)); ov[rr, cc] = cmap(tn); ov[rr, cc, 3] = .9
    ax = axs[i, 0]; ax.imshow(img); ax.imshow(ov); ax.set_xlim(c0, c1); ax.set_ylim(r1, r0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(name, fontsize=a.fs + 4, fontweight='bold'); ax.set_title('colour = along the tail\n(dark = root, yellow = tip)', fontsize=a.fs)
    ov2 = np.zeros(fs.shape + (4,)); ov2[rr, cc, :3] = RCOL[lab]; ov2[rr, cc, 3] = .9
    ax = axs[i, 1]; ax.imshow(img); ax.imshow(ov2); ax.set_xlim(c0, c1); ax.set_ylim(r1, r0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('SOFT-SNAP mesh region\n' + ' / '.join('%s %.0f%%' % (REG[k], 100 * frac[k]) for k in range(5) if frac[k] > .01), fontsize=a.fs)
    ax = axs[i, 2]; ax.scatter(z, y, s=1, c=[RCOL[k] * .5 + .5 for k in vreg])
    ax.scatter(soft[:, 2], soft[:, 1], s=6, c=cmap(tn), edgecolors='none')
    ax.set_aspect('equal'); ax.set_xlabel('z  (SIDE view, rear on the left)'); ax.set_ylabel('y')
    ax.set_title('soft-snap points, side view (mesh tinted by region)', fontsize=a.fs)
    ax = axs[i, 3]; ax.scatter(x[rear], y[rear], s=2, c=[RCOL[k] * .5 + .5 for k in vreg[rear]])
    ax.scatter(soft[:, 0], soft[:, 1], s=6, c=cmap(tn), edgecolors='none')
    ax.set_aspect('equal'); ax.set_xlabel('x  (REAR view, z < -0.35)'); ax.set_title('rear view', fontsize=a.fs)
fig.tight_layout(); fig.savefig(a.out, dpi=100)
json.dump(stats, open(a.out.replace('.png', '.json'), 'w'), indent=1); print(json.dumps(stats, indent=1))
