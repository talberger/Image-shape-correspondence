"""
Mark which of the 100 ground-truth eval labels a run got right, directly on the mesh.

2026-09-08, Tal: *"תוכל לסמן לי איזה לייבלים נתפסו ואיזה לא? על רינדור של המאש. אחד ב step 0 ואחד
step 300 שאראה מה קרה"*. The eval reports aggregate AUC and mean error; this shows WHICH points
moved and WHERE they went, which is what an aggregate cannot say.

Every label is drawn at its GROUND-TRUTH vertex: green if the run put it within --threshold of the
diameter, red if not. A missed label also gets a smaller ORANGE sphere at the vertex the run
actually chose, so the size of the mistake is visible -- a miss that lands next door reads very
differently from one that lands on the other end of the guitar, and the two are the same red dot
without it.

Metric is `geometric snap` by default (`error_hard_snap`), i.e. the eval's first column.

Run (ZS vs phase-2 on the current baseline):
cd /workspace/bsb && python analysis/render_eval_hits.py \
  --evals demo/guitar/eval_labels/v2_edge_inset/isobbox/eval_isobbox_ZS_sdsquare_fuse16_v2.pth \
          demo/guitar/eval_labels/v2_edge_inset/isobbox/eval_isobbox_sdsquare_fuse16_T1_seed0_v2.pth \
  --labels "step 0 (ZS)" "step 300 (phase-2 T=1)" \
  --obj_path ./meshes/guitar_isotropic_bboxnorm.obj --name guitar_isobbox \
  --output_path <out>/eval_hits_ZS_vs_phase2.png
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.make_run_vs_baseline_comparison import HEADER_H, TITLE_H, label_band
from analysis.render_colored_mesh import RenderMesh, add_spheres_varying_radius, render_one_view
from modules.render import Renderer
from modules.utils import device, loadmesh, setcolor_mesh_batched

ERROR_KEY = {'geometric snap': 'error_hard_snap',
             'feature argmax': 'error_argmax',
             'soft mean distance': 'error_expected'}
PRED_KEY = {'geometric snap': 'snap_vertex_idx',
            'feature argmax': 'argmax_vertex_idx',
            'soft mean distance': 'argmax_vertex_idx'}

HIT = (0.05, 0.70, 0.15)      # green: within threshold
MISS = (0.85, 0.10, 0.10)     # red:   ground-truth vertex of a missed label
LANDED = (1.00, 0.60, 0.05)   # orange: where a missed label actually went
BASE = (0.82, 0.82, 0.84)     # neutral mesh, so the markers carry all the colour


def build_panel(mesh, eval_path, metric, threshold, renderer, elev, azim,
                radius, landed_radius, resolution):
    d = torch.load(eval_path, map_location='cpu')
    diameter = float(d['geo_diameter'])
    err = d[ERROR_KEY[metric]].float() / diameter * 100
    gt = d['gt_vertex_idx'].long()
    pred = d[PRED_KEY[metric]].long()
    hit = err <= threshold

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices.detach().cpu().numpy().astype(np.float64))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces.detach().cpu().numpy().astype(np.int32))
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
        np.tile(np.array(BASE, dtype=np.float64), (mesh.vertices.shape[0], 1)))

    verts = mesh.vertices.detach().cpu().numpy().astype(np.float64)
    points, colors, radii = [], [], []
    # Landed markers first, so a ground-truth sphere drawn at the same place stays on top.
    for i in torch.nonzero(~hit).flatten().tolist():
        points.append(verts[pred[i]]); colors.append(LANDED); radii.append(landed_radius)
    for i in range(gt.numel()):
        points.append(verts[gt[i]])
        colors.append(HIT if hit[i] else MISS)
        radii.append(radius)

    combined = add_spheres_varying_radius(mesh_o3d, np.array(points), np.array(colors, dtype=np.float64),
                                          np.array(radii), resolution=resolution)
    rmesh = RenderMesh(torch.from_numpy(np.asarray(combined.vertices, dtype=np.float32)).to(device),
                       torch.from_numpy(np.asarray(combined.triangles, dtype=np.int64)).to(device))
    # setcolor_mesh_batched wants a batch dimension: [1, V, 3], same as render_colored_mesh.py.
    setcolor_mesh_batched(rmesh, torch.from_numpy(
        np.asarray(combined.vertex_colors, dtype=np.float32)).to(device).unsqueeze(0))
    img = render_one_view(renderer, rmesh, elev, azim)

    n_hit = int(hit.sum())
    stats = dict(n_hit=n_hit, n=gt.numel(), mean=float(err.mean()),
                 median_miss=float(err[~hit].median()) if n_hit < gt.numel() else 0.0)
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert('RGB'), stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evals', nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--output_path', required=True)
    p.add_argument('--obj_path', default='./meshes/guitar_isotropic_bboxnorm.obj')
    p.add_argument('--name', default='guitar_isobbox')
    p.add_argument('--metric', default='geometric snap', choices=list(ERROR_KEY))
    p.add_argument('--threshold', type=float, default=3.0,
                   help='%% of mesh geodesic diameter counted as a hit. 3%% is where the ZS/phase-2 '
                        'gap is widest (78%% vs 17%% of points, 2026-09-08).')
    p.add_argument('--elev_deg', type=float, default=0)
    p.add_argument('--azim_deg', type=float, default=90)
    p.add_argument('--render_res', type=int, default=768)
    p.add_argument('--marker_radius', type=float, default=0.022)
    p.add_argument('--landed_radius', type=float, default=0.013)
    p.add_argument('--marker_resolution', type=int, default=12)
    args = p.parse_args()
    assert len(args.evals) == len(args.labels), '--evals and --labels must match'

    mesh = loadmesh(dir=args.obj_path, name=args.name)
    renderer = Renderer(dim=(args.render_res, args.render_res))

    panels, all_stats = [], []
    for path, label in zip(args.evals, args.labels):
        img, stats = build_panel(mesh, path, args.metric, args.threshold, renderer,
                                 args.elev_deg, args.azim_deg, args.marker_radius,
                                 args.landed_radius, args.marker_resolution)
        panels.append((img, label, stats))
        all_stats.append((label, stats))
        print(f'{label}: {stats["n_hit"]}/{stats["n"]} within {args.threshold}%  '
              f'(mean error {stats["mean"]:.2f}%, median miss {stats["median_miss"]:.2f}%)')

    w = sum(im.width for im, _, _ in panels)
    h = max(im.height for im, _, _ in panels)
    title = (f'{args.metric}: which of the {panels[0][2]["n"]} eval labels landed within '
             f'{args.threshold}% of the mesh diameter   '
             f'[green = hit, red = missed label, orange = where a missed label went]')
    canvas = Image.new('RGB', (w, TITLE_H + HEADER_H + h), (255, 255, 255))
    canvas.paste(label_band(w, title, font_size=20, bg=(10, 10, 60)).resize((w, TITLE_H)), (0, 0))
    x = 0
    for im, label, stats in panels:
        band = label_band(im.width, f'{label} -- {stats["n_hit"]}/{stats["n"]} hits, '
                                    f'mean {stats["mean"]:.2f}%', bg=(40, 40, 40))
        canvas.paste(band, (x, TITLE_H))
        canvas.paste(im, (x, TITLE_H + HEADER_H))
        x += im.width

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    canvas.save(args.output_path)
    print(f'saved {args.output_path}  ({canvas.width}x{canvas.height})')


if __name__ == '__main__':
    main()
