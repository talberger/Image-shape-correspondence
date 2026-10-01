"""Per-segment learned mu of several runs side by side (Tal, 2026-09-27): the isotropic mu (mu_scale.pth) and, for the
directional scopes, the mu along the ellipse's two principal axes (major / minor, = sqrt of M's eigenvalues) + the
major-axis image angle (0 = vertical / along rows, 90 = horizontal). The hybrid's cross-segment ellipse is row 'xglobal'.
Usage: python analysis/mu_table.py --fine_seg_path F --runs LABEL=RUN_DIR ... [--min_frac 0.01] [--out table.md]"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--fine_seg_path', required=True); ap.add_argument('--runs', nargs='+', required=True)
ap.add_argument('--min_frac', type=float, default=0.01); ap.add_argument('--out', default=None)
a = ap.parse_args()
fs = torch.load(a.fine_seg_path, map_location='cpu').numpy(); tot = (fs > 0).sum()
segs = [int(s) for s in np.unique(fs) if s > 0 and (fs == s).sum() / tot >= a.min_frac]
cols, rows = [], {s: ['seg %d' % s, '%.1f%%' % (100 * (fs == s).sum() / tot)] for s in segs}
rows['xglobal'] = ['xglobal', '-']
for spec in a.runs:
    label, d = spec.split('=', 1)
    if os.path.exists(os.path.join(d, 'mu_aniso.pth')):
        A = torch.load(os.path.join(d, 'mu_aniso.pth'), map_location='cpu')
        cols.append('%s major / minor @ angle' % label)
        fmt = lambda g: '%.3f / %.3f @ %3.0f' % (A['major'][g], A['minor'][g], A['major_angle_deg'][g])
        for s in segs: rows[s].append(fmt(s))
        rows['xglobal'].append(fmt(-1) if A['scope'] in ('segment_aniso_xglobal', 'global_then_segment_aniso') else '-')
    else:
        m = torch.load(os.path.join(d, 'mu_scale.pth'), map_location='cpu')['mu_scale']
        cols.append('%s iso mu' % label)
        for s in segs: rows[s].append('%.3f' % m[s])
        rows['xglobal'].append('-')
hdr = ['segment', '% of object'] + cols
lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
lines += ['| ' + ' | '.join(r) + ' |' for k, r in rows.items() if k != 'xglobal' or any(x != '-' for x in r[2:])]
txt = '\n'.join(lines); print(txt)
if a.out: open(a.out, 'w').write(txt + '\n')
