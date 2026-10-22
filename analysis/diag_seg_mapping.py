"""Where does each pixel of ONE fine segment land on the mesh, per run? (Tal, 2026-09-24)
Per run (phase-1 + --run_dirs): photo crop coloured by the argmax part (training centres = black dots),
the soft-map mass on the SECOND front leg (x-), and the argmax hits on the mesh in FRONT view (x,y: separates
left/right legs) and SIDE view (z,y), hits coloured by the pixel's image ROW (same colormap as the reference crop)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from modules.mapping_loss import soft_map_to_vertices
from modules.utils import loadmesh
from analysis.diag_mu_compare import part_of, PARTS, PCOL

ap = argparse.ArgumentParser()
ap.add_argument('--run_dirs', nargs='+', required=True); ap.add_argument('--labels', nargs='+', required=True)
ap.add_argument('--seg', type=int, required=True); ap.add_argument('--out', required=True)
ap.add_argument('--box', type=int, nargs=4, default=None, metavar=('R0', 'R1', 'C0', 'C1'),
                help='keep only the segment pixels inside rows [R0,R1] x cols [C0,C1] (a leg inside the big body segment)')
ap.add_argument('--name', default=None, help='title/stats tag, default seg<N>')
ap.add_argument('--pred_rel', default='alpha/alpha0.5/pred_f.pth', help='default = the production LORA-W reading (alpha 0.5, as the eval); pred_f.pth = full LORA')
ap.add_argument('--cols', type=int, nargs='+', default=None, help='keep only these columns, 1-based (Tal 2026-10-22: 2 3 4)')
ap.add_argument('--fs', type=int, default=8, help='title/label font size (Tal 2026-09-28: 18 for readability)')
a = ap.parse_args(); dev = 'cuda'
plt.rcParams.update({'xtick.labelsize': a.fs, 'ytick.labelsize': a.fs, 'axes.labelsize': a.fs})
cfg = json.load(open(os.path.join(a.run_dirs[0], 'run_config.json')))
geo2 = torch.load(cfg['geo_table_2d_path'], map_location='cpu'); coords = geo2['coords'].long()
fs = torch.load(cfg['fine_seg_path'], map_location='cpu').numpy().copy()
fs[~geo2['geodesic_safe_mask'].numpy().astype(bool)] = 0
_ip = cfg['fine_seg_path'].replace('_fine_segments.pth', '.png')
if not _ip.endswith('.png'): _ip = cfg['object_mask_path'].replace('_mask.pth', '.png')   # other map names (VLPart): the photo next to the mask
img = plt.imread(_ip)
V = loadmesh(dir=cfg['obj_path'], name='mesh', load_rings=False).vertices.to(dev); Vc = V.cpu().numpy()
vpart = part_of(Vc); pix = torch.load(cfg['pix_dino_features_path'], map_location='cpu')
sel = fs == a.seg
if a.box: _b = np.zeros_like(sel); _b[a.box[0]:a.box[1] + 1, a.box[2]:a.box[3] + 1] = True; sel &= _b
rr, cc = np.nonzero(sel)
cmask = sel[coords[:, 0], coords[:, 1]]; C = coords[torch.from_numpy(cmask)].numpy()
cpos = {(r, c): i for i, (r, c) in enumerate(zip(rr, cc))}; ci = np.array([cpos[(r, c)] for r, c in C])
cnum = np.empty(len(C), int); cnum[np.argsort(C[:, 0], kind='stable')] = np.arange(len(C))   # centre number = rank by image row
r0, r1, c0, c1 = rr.min() - 12, rr.max() + 12, cc.min() - 12, cc.max() + 12
rown = (rr - rr.min()) / max(rr.max() - rr.min(), 1); cmap = plt.cm.viridis
runs = [('phase1', os.path.join(cfg['init_encoder_dir'], 'pred_f.pth'))] + \
       [(l, os.path.join(d, a.pred_rel)) for l, d in zip(a.labels, a.run_dirs)]
WR = [2.2, 2.2, 2.2, 2, 2.5]
keep = [c - 1 for c in a.cols] if a.cols else list(range(5))
fig, _ax = plt.subplots(len(runs), len(keep), figsize=(34 * sum(WR[k] for k in keep) / sum(WR), 10 * len(runs)),
                        gridspec_kw={'width_ratios': [WR[k] for k in keep]}, squeeze=False)   # side view dropped (Tal 2026-09-28)
_, _dummy = plt.subplots(len(runs), 5, squeeze=False)                 # --cols: the dropped columns are drawn here and discarded
axs = _dummy.copy()
for j, k in enumerate(keep): axs[:, k] = _ax[:, j]
legs = Vc[:, 1] < -0.2
stats = {'seg': a.seg, 'box': a.box, 'name': a.name, 'n_pixels': int(len(rr)), 'n_centres': int(len(C))}
for i, (name, pf) in enumerate(runs):
    vf = torch.load(pf, map_location=dev)
    p = F.normalize(pix[rr, cc].float().to(dev), dim=-1)
    _, tk, w = soft_map_to_vertices(p, vf, V, cfg['k_vertices_per_pixel'], return_topk=True, temperature=cfg['temperature'])
    tk, w = tk.cpu().numpy(), w.detach().cpu().numpy()
    top = tk[:, 0]; plab = vpart[top]
    mass = np.stack([(w * (vpart[tk] == k)).sum(1) for k in range(5)], 1)       # soft mass per part [npix, 5]
    frac = np.bincount(plab, minlength=5) / len(plab); cfrac = np.bincount(plab[ci], minlength=5) / len(ci)
    xneg = mass[:, 2] + mass[:, 4]; xpos = mass[:, 1] + mass[:, 3]          # x- / x+ side, front + hind legs
    mixed = ((xpos > 0.1) & (xneg > 0.1)).mean()
    stats[name] = {'pixels_argmax_part': dict(zip(PARTS, frac.round(3).tolist())),
                   'centres_argmax_part': dict(zip(PARTS, cfrac.round(3).tolist())),
                   'mean_soft_mass_part': dict(zip(PARTS, mass.mean(0).round(3).tolist())),
                   'pixels_with_>10%_on_BOTH_sides': round(float(mixed), 3)}
    ov = np.zeros(fs.shape + (4,)); ov[rr, cc, :3], ov[rr, cc, 3] = PCOL[plab], .85
    ax = axs[i, 0]; ax.imshow(img); ax.imshow(ov); ax.scatter(C[:, 1], C[:, 0], s=8, c='k')
    ax.set_xlim(c0, c1); ax.set_ylim(r1, r0)
    ax.set_ylabel('ZS (phase-1)' if name == 'phase1' else name, fontsize=a.fs + 6, fontweight='bold')
    ax.set_title('px  %s\ncentres  %s' % (
        ' / '.join('%s %.0f%%' % (PARTS[k].replace('front ', ''), 100 * frac[k]) for k in range(5) if frac[k] > .01),
        ' / '.join('%s %.0f%%' % (PARTS[k].replace('front ', ''), 100 * cfrac[k]) for k in range(5) if cfrac[k] > .01)), fontsize=a.fs)
    ov2 = np.zeros(fs.shape); ov2[:] = np.nan; ov2[rr, cc] = xneg
    ax = axs[i, 1]; ax.imshow(img); im = ax.imshow(ov2, cmap='magma', vmin=0, vmax=1); ax.set_xlim(c0, c1); ax.set_ylim(r1, r0)
    ax.set_title('mixed: %.0f%% px' % (100 * mixed), fontsize=a.fs)
    ov3 = np.zeros(fs.shape + (4,)); ov3[rr, cc] = cmap(rown); ov3[rr, cc, 3] = .9
    ax = axs[i, 2]; ax.imshow(img); ax.imshow(ov3); ax.set_xlim(c0, c1); ax.set_ylim(r1, r0)
    for n, (r, c) in zip(cnum, C): ax.text(c, r, str(n), fontsize=min(a.fs + 1, 11), fontweight='bold', ha='center', va='center', color='k')

    soft = (w[..., None] * Vc[tk]).sum(1)                                        # soft-snap point, as in training
    stats[name]['soft_point_x_mean_std'] = [round(float(soft[:, 0].mean()), 3), round(float(soft[:, 0].std()), 3)]
    stats[name]['soft_points_between_legs_|x|<0.07'] = round(float((np.abs(soft[:, 0]) < .07).mean()), 3)
    ax = axs[i, 3]; ax.scatter(Vc[legs, 0], Vc[legs, 1], s=2, c='lightgray')
    ax.scatter(soft[:, 0], soft[:, 1], s=4, c=cmap(rown), alpha=.6)
    ax.set_aspect('equal'); ax.set_xlabel('x  (front view: left/right legs)'); ax.set_xlim(-.4, .4)
    ax = axs[i, 4]; ax.scatter(Vc[legs, 0], Vc[legs, 1], s=2, c='lightgray')
    ax.scatter(soft[ci, 0], soft[ci, 1], s=14, c=cmap(rown[ci]), edgecolors='k', linewidths=.3)
    for n, (x, y) in zip(cnum, soft[ci][:, [0, 1]]): ax.annotate(str(n), (x, y), fontsize=min(a.fs + 2, 14), xytext=(2, 2), textcoords='offset points')
    ax.set_aspect('equal'); ax.set_xlim(-.2, .25); ax.set_ylim(-.75, -.25); ax.set_xlabel('x  (front view, zoomed)')
for ax in axs[:, :3].flat: ax.set_xticks([]); ax.set_yticks([])
if a.cols:                                                           # the ZS/run name sits on column 1; move it to the first kept one
    for i, (name, _) in enumerate(runs): _ax[i, 0].set_ylabel('ZS (phase-1)' if name == 'phase1' else name, fontsize=a.fs + 6, fontweight='bold')
HEAD = ['argmax mesh part\nper pixel\n(dots = %d centres)' % len(C),
        'soft weight on x- legs\n(0 = all x+, 1 = all x-)\nmixed = >10% on both',
        'reference:\ncolour = image row\nnumbers = centres (0 = top)',
        'SOFT-SNAP points\nfront view\n(colour = image row)', 'CENTRES only, soft-snap\nzoomed\n(number = centre, as in col 3)']
import textwrap
fig.canvas.draw()
for ax in _ax.flat:                                                  # wrap titles to the axis width (large --fs)
    t = ax.title; wid = ax.get_window_extent().width / fig.dpi * 72
    t.set_text('\n'.join(textwrap.fill(l, max(int(wid / (0.55 * a.fs)), 12)) for l in t.get_text().split('\n')))
fig.tight_layout(rect=(0, 0, 1, 1 - 1.8 / (10 * len(runs))))
for ax, h in zip(_ax[0], [HEAD[k] for k in keep]):
    bb = ax.get_position(); fig.text((bb.x0 + bb.x1) / 2, 1 - 0.3 / (10 * len(runs)), h, ha='center', va='top', fontsize=a.fs + 2, fontweight='bold')
fig.savefig(a.out, dpi=130)
json.dump(stats, open(a.out.replace('.png', '.json'), 'w'), indent=1); print(json.dumps(stats, indent=1))
