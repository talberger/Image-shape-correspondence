"""Mesh view of diag_seg_mapping.py: where do the pixels of ONE fine segment land, per run? (Tal, 2026-09-27)
Per run writes two PLYs (in the training mesh frame, as loadmesh gives it):
  <label>_seg<N>_mass.ply    -- mesh coloured ONLY by this segment's soft-map weights (top-k, as in training): each vertex
                                gets the weight-averaged image-ROW colour (viridis, 0 = top of the segment) of the pixels
                                that hit it; brightness fades with the total weight; untouched vertices light grey.
  <label>_seg<N>_centres.ply -- grey mesh + one sphere per training centre at its SOFT-SNAP point (may float off the
                                surface -- that is the 'bridge'), same row colours."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, trimesh
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from modules.mapping_loss import soft_map_to_vertices
from modules.utils import loadmesh

ap = argparse.ArgumentParser()
ap.add_argument('--run_dirs', nargs='+', required=True); ap.add_argument('--labels', nargs='+', required=True)
ap.add_argument('--seg', type=int, required=True); ap.add_argument('--out_dir', required=True)
ap.add_argument('--sphere_r', type=float, default=0.008)
ap.add_argument('--pred_rel', default='alpha/alpha0.5/pred_f.pth', help='default = the production LORA-W reading (alpha 0.5, as the eval); pred_f.pth = full LORA')
a = ap.parse_args(); dev = 'cuda'; os.makedirs(a.out_dir, exist_ok=True)
cfg = json.load(open(os.path.join(a.run_dirs[0], 'run_config.json')))
geo2 = torch.load(cfg['geo_table_2d_path'], map_location='cpu'); coords = geo2['coords'].long()
fs = torch.load(cfg['fine_seg_path'], map_location='cpu').numpy().copy()
fs[~geo2['geodesic_safe_mask'].numpy().astype(bool)] = 0
mesh = loadmesh(dir=cfg['obj_path'], name='mesh', load_rings=False)
V = mesh.vertices.to(dev); Vc = V.cpu().numpy(); Fc = mesh.faces.cpu().numpy()
pix = torch.load(cfg['pix_dino_features_path'], map_location='cpu')
rr, cc = np.nonzero(fs == a.seg)
cmask = fs[coords[:, 0], coords[:, 1]] == a.seg; C = coords[torch.from_numpy(cmask)].numpy()
cpos = {(r, c): i for i, (r, c) in enumerate(zip(rr, cc))}; ci = np.array([cpos[(r, c)] for r, c in C])
rown = (rr - rr.min()) / max(rr.max() - rr.min(), 1); rowcol = plt.cm.viridis(rown)[:, :3]
GREY = np.array([.8, .8, .8])
runs = [('phase1', os.path.join(cfg['init_encoder_dir'], 'pred_f.pth'))] + \
       [(l, os.path.join(d, a.pred_rel)) for l, d in zip(a.labels, a.run_dirs)]
for name, pf in runs:
    vf = torch.load(pf, map_location=dev)
    p = F.normalize(pix[rr, cc].float().to(dev), dim=-1)
    _, tk, w = soft_map_to_vertices(p, vf, V, cfg['k_vertices_per_pixel'], return_topk=True, temperature=cfg['temperature'])
    tk, w = tk.cpu().numpy(), w.detach().cpu().numpy()
    tot = np.bincount(tk.ravel(), w.ravel(), len(Vc))
    col = np.stack([np.bincount(tk.ravel(), (w * rowcol[:, j:j + 1]).ravel(), len(Vc)) for j in range(3)], 1)
    hit = tot > 0; col[hit] /= tot[hit, None]
    s = np.clip(tot / np.percentile(tot[hit], 95), 0, 1)[:, None]            # fade weakly-hit vertices toward grey
    vcol = np.where(hit[:, None], s * col + (1 - s) * GREY, GREY)
    trimesh.Trimesh(Vc, Fc, vertex_colors=(vcol * 255).astype(np.uint8), process=False).export(
        os.path.join(a.out_dir, '%s_seg%d_mass.ply' % (name, a.seg)))
    soft = (w[..., None] * Vc[tk]).sum(1)
    parts = [trimesh.Trimesh(Vc, Fc, vertex_colors=np.tile((GREY * 255).astype(np.uint8), (len(Vc), 1)), process=False)]
    for j in ci:
        sp = trimesh.creation.icosphere(1, a.sphere_r); sp.apply_translation(soft[j])
        sp.visual.vertex_colors = np.tile((rowcol[j] * 255).astype(np.uint8), (len(sp.vertices), 1)); parts.append(sp)
    trimesh.util.concatenate(parts).export(os.path.join(a.out_dir, '%s_seg%d_centres.ply' % (name, a.seg)))
    print(name, 'vertices hit:', int(hit.sum()))
