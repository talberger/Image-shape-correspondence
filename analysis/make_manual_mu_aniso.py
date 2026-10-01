"""HAND-MADE per-segment ellipses from the segment's pixel spread (Tal, 2026-09-28).
Question: if the ellipses matched the part's shape, would the mapping improve? -> build them by hand, freeze, train.
Per fine segment: PCA of its pixel (row, col) coords -> major axis direction = the segment's long direction;
axis ratio = sqrt(l1/l2) ** --ratio_power (1 = raw spread, 0.5 = softened); size: sqrt(major*minor) = --scale
('iso' = the iso per-seg run's learned mu for that segment, or a number).
Writes <out_dir>/mu_aniso.pth in the training format (mu_aniso [G,3] = (log e_a, log e_b, c), L = [[e_a,0],[c,e_b]],
s(u) = |L^T u|, same as train_relational_loss.aniso_scale_from_params) + major/minor/major_angle_deg for the plotter."""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--fine_seg_path', required=True); ap.add_argument('--out_dir', required=True)
ap.add_argument('--ratio_power', type=float, default=0.5)
ap.add_argument('--scale', default='iso', help="'iso' (needs --iso_run) or a number")
ap.add_argument('--iso_run', default=None, help='iso per-seg run dir (mu_scale.pth)')
ap.add_argument('--add_global', action='store_true', help='append ONE global row (PCA of ALL object pixels, same ratio/size rule) -- the cross-seg ellipse of segment_aniso_xglobal')
a = ap.parse_args()
fs = torch.load(a.fine_seg_path, map_location='cpu').numpy()
G = int(fs.max()) + 1
if a.scale == 'iso':
    mu_iso = torch.load(os.path.join(a.iso_run, 'mu_scale.pth'), map_location='cpu')['mu_scale'].float().numpy()
params = np.zeros((G, 3), np.float32); major = np.ones(G, np.float32); minor = np.ones(G, np.float32); ang = np.zeros(G, np.float32)
info = {}
rows = list(range(1, G)) + (['global'] if a.add_global else [])
if a.add_global:
    params = np.concatenate([params, np.zeros((1, 3), np.float32)]); major, minor, ang = [np.append(v, v[0]) for v in (major, minor, ang)]
for g in rows:
    rr, cc = np.nonzero(fs > 0) if g == 'global' else np.nonzero(fs == g)
    if g == 'global': g = G
    if len(rr) < 3: continue
    X = np.stack([rr, cc], 1).astype(np.float64); X -= X.mean(0)
    ev, evec = np.linalg.eigh(np.cov(X.T))                                   # ascending
    e1 = evec[:, 1]                                                          # (row, col) of the long direction
    ratio = math.sqrt(ev[1] / max(ev[0], 1e-9)) ** a.ratio_power
    s0 = float(mu_iso[g]) if a.scale == 'iso' and g < G else 1.0 if a.scale == 'iso' else float(a.scale)
    M1, m1 = s0 * math.sqrt(ratio), s0 / math.sqrt(ratio)                    # geometric mean = s0
    e2 = np.array([-e1[1], e1[0]])
    M = M1 ** 2 * np.outer(e1, e1) + m1 ** 2 * np.outer(e2, e2)
    ea = math.sqrt(M[0, 0]); c = M[0, 1] / ea; eb = math.sqrt(M[1, 1] - c * c)   # Cholesky of M = L L^T
    params[g] = (math.log(ea), math.log(eb), c)
    major[g], minor[g] = M1, m1; ang[g] = math.degrees(math.atan2(e1[1], e1[0])) % 180.0
    info[g] = {'n_px': int(len(rr)), 'spread_ratio_raw': round(math.sqrt(ev[1] / max(ev[0], 1e-9)), 2),
               'major': round(M1, 3), 'minor': round(m1, 3), 'angle_deg': round(float(ang[g]), 1), 'size_s0': round(s0, 3)}
os.makedirs(a.out_dir, exist_ok=True)
torch.save({'scope': 'segment_aniso_xglobal' if a.add_global else 'segment_aniso', 'mu_aniso': torch.from_numpy(params), 'major': torch.from_numpy(major),
            'minor': torch.from_numpy(minor), 'major_angle_deg': torch.from_numpy(ang), 'manual': vars(a)},
           os.path.join(a.out_dir, 'mu_aniso.pth'))
json.dump(info, open(os.path.join(a.out_dir, 'manual_ellipses.json'), 'w'), indent=1)
print(json.dumps(info, indent=1))
