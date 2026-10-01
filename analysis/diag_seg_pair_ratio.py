"""Per-segment IN / OUT pair ratio 3D/2D, as the relational loss sees it (Tal, 2026-10-01).
Rebuilds the 2026-09-29 tail table ("in-seg tail pairs want 0.057 but the stub holds ~0.032"), whose script was not saved.
Same centres (sample_center_pixels, seed), same geodesic tables and normalizers (geo_diag_2d/3d) as the run_config.
Per reading: for every centre of --seg, draw --k IN and --k OUT neighbours (fixed seed, same pairs for every reading),
3D distance = expected geodesic (training's quantity) and soft-snap geodesic (nearest vertex to the soft point).
Reports medians of norm_d_2D, norm_d_3D and of the per-pair ratio norm_d_3D / norm_d_2D (1 = 3D matches the photo, mu NOT applied).
Readings: --readings LABEL=PRED_F_PATH ..."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from modules.mapping_loss import sample_center_pixels, soft_map_to_vertices
from modules.relational_loss import (build_segment_index, sample_in_segment_neighbors, sample_out_segment_neighbors,
                                     expected_geodesic_distance_3d, geodesic_distance_2d_lookup)
from modules.utils import loadmesh
from analysis.build_geodesic_tables import largest_connected_component
from phase2.train_relational_loss import load_geodesic_tables

ap = argparse.ArgumentParser()
ap.add_argument('--run_config', required=True); ap.add_argument('--readings', nargs='+', required=True)
ap.add_argument('--segs', type=int, nargs='+', default=[2]); ap.add_argument('--k', type=int, default=32)
ap.add_argument('--out', default=None)
a = ap.parse_args(); dev = 'cuda'
c = json.load(open(a.run_config))
cm = torch.load(c['object_mask_path'], map_location='cpu').numpy()
fs = torch.load(c['fine_seg_path'], map_location='cpu').numpy()
cm, _ = largest_connected_component(cm)
coords, _, segids = sample_center_pixels(cm, fs, c['num_center_pixels'], seed=c['seed'])
g3, gd3, g2, gd2, p2n, safe = load_geodesic_tables(c['geo_table_3d_path'], c['geo_table_2d_path'], c['object_mask_path'],
                                                   coords, c['num_center_pixels'], c['seed'], pool_mode=False)
gd3 = c.get('geo_diag_3d') or gd3; gd2 = c.get('geo_diag_2d') or gd2
g3 = g3.to(dev)
fs_safe = fs.copy(); fs_safe[~np.asarray(safe, bool)] = 0
seg_idx = build_segment_index(fs_safe, safe)
V = loadmesh(dir=c['obj_path'], name='mesh', load_rings=False).vertices.to(dev)
pix = torch.load(c['pix_dino_features_path'], map_location='cpu')
K, T = c['k_vertices_per_pixel'], c['temperature']

pairs = {}
for s in a.segs:
    sel = (segids == s).nonzero().flatten()
    cc, sg = coords[sel], segids[sel]
    torch.manual_seed(0); np.random.seed(0)
    nin = sample_in_segment_neighbors(cc, sg, seg_idx, k=a.k)
    nout = sample_out_segment_neighbors(cc, sg, seg_idx, k=a.k)
    rows = g2[sel] if g2.shape[0] == coords.shape[0] else g2[sel]
    pairs[s] = (cc, {'IN': nin, 'OUT': nout}, rows)

def mapf(vf, xy):
    p = F.normalize(pix[xy[..., 0].reshape(-1), xy[..., 1].reshape(-1)].float().to(dev), dim=-1)
    soft, tk, w = soft_map_to_vertices(p, vf, V, K, return_topk=True, temperature=T)
    snap = torch.cdist(soft.float(), V).argmin(1)
    return tk, w.detach(), snap

res = {'geo_diag_2d': float(gd2), 'geo_diag_3d': float(gd3), 'k': a.k}
print('normalizers: 2D %.2f px, 3D %.4f' % (gd2, gd3))
for lab, pf in (r.split('=', 1) for r in a.readings):
    vf = torch.load(pf, map_location=dev); res[lab] = {}
    for s, (cc, nb, rows) in pairs.items():
        ctk, cw, csn = mapf(vf, cc)
        for kind, nc in nb.items():
            n, k = nc.shape[:2]
            ntk, nw, nsn = mapf(vf, nc)
            d3 = expected_geodesic_distance_3d(ctk, cw, ntk.view(n, k, K), nw.view(n, k, K), g3) / gd3
            d3s = g3[csn.view(n, 1).expand(n, k), nsn.view(n, k)] / gd3
            d2 = geodesic_distance_2d_lookup(nc, rows, p2n).to(dev) / gd2
            ok = d2 > 1e-6
            st = {'n_centres': n, 'd2_med': d2[ok].median().item(), 'd3_med': d3[ok].median().item(),
                  'd3snap_med': d3s[ok].median().item(), 'ratio_med': (d3 / d2)[ok].median().item(),
                  'ratio_snap_med': (d3s / d2)[ok].median().item()}
            res[lab]['seg%d_%s' % (s, kind)] = st
            print('%-22s seg %d %-3s n=%3d | 2D %.3f | 3D exp %.3f (ratio %.2f) | 3D snap %.3f (ratio %.2f)'
                  % (lab, s, kind, n, st['d2_med'], st['d3_med'], st['ratio_med'], st['d3snap_med'], st['ratio_snap_med']))
if a.out: json.dump(res, open(a.out, 'w'), indent=1)
