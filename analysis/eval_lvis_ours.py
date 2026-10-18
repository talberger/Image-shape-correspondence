"""Score ONE DensePose-LVIS instance with OUR correspondence, in SHIC's Table 1 metric.

The counterpart of analysis/eval_densepose_lvis.py, which scores SHIC's released weights over the
whole class in one process. Ours cannot work that way: phase-2 trains per image, so scoring is a
per-instance step inside scripts/run_lvis_instance.sh, run while that instance's features and
encoder are still on disk and before they are deleted (7.4 GB/instance, 132 instances -> the
intermediates cannot all coexist). Hence: one instance in, one small JSON out.

Metric, identical to eval_densepose_lvis.py so the numbers are comparable to the 33.09 that script
reproduces for SHIC's bear (paper: 33.0):
  * error  = geodesic(GT vertex, predicted vertex) on bear.obj, heat-method geodesics with the max
             normalized to 228 -- the SAME cached [V, V] matrix that script built.
  * report = mean over points (SHIC's number), plus median and the mean of per-instance means.

PREDICTION RULE (Tal, 2026-09-19). Two predictions are reported per point, and they are NOT the
same rule:
  * soft_snap -- OURS, the main number. softmax over the pixel's top-k vertex similarities at the
                 READ temperature -> weighted 3D position -> nearest actual vertex. Exactly
                 analysis/color_mesh_from_soft_mapping.py::compute_soft_snap, the same k=10 and
                 read T=0.01 every bear analysis uses (scripts/analyze_shicbear_arm.sh).
  * argmax    -- top-1 vertex by similarity. NOT our rule; it is SHIC's, reported so there is an
                 apples-to-apples line against their 33.09 rather than a comparison across two
                 different decoding rules.

IN-MASK SPLIT: prep_lvis_instance.py records, per GT point, whether it landed inside the LVIS
polygon mask (`in_mask`). Points outside it still get scored in the `all` block -- SHIC's own eval
scores every point, so dropping them would not be comparable -- but the `in_mask` block is reported
alongside, because phase-2 only ever trained on mask pixels and a GT point outside the mask is
asking the mapping about a pixel it never saw. If the two blocks diverge a lot, that gap is about
mask quality, not about the correspondence.

Run (one instance, after scripts/run_lvis_instance.sh has produced the alpha export):
cd $DISC_ROOT && python analysis/eval_lvis_ours.py \
  --run_dir data/densepose_lvis/runs/lvis117683 --tag lvis117683 \
  --vertex_features_path data/densepose_lvis/runs/lvis117683/encoder_loraW_all_r4_phase2/alpha/alpha0.5/pred_f.pth \
  --obj_path ./demo/shic/shapes/bear.obj --name shic_bear \
  --geodesics_path data/densepose_lvis/eval_shic_bear/bear_heat_geodesics_228.npy \
  --out_json data/densepose_lvis/runs/lvis117683/eval_ours_alpha0.5.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.color_mesh_from_soft_mapping import compute_soft_snap
from modules.utils import loadmesh


def predict(pixel_features, vertex_features, vertex_positions, k, temperature, device, chunk_size):
    """Both decoding rules for the same [P, F] pixel features -> (argmax_idx, soft_snap_idx).

    argmax is computed here rather than reusing analysis/pixel_to_vertex_nn.py because that script
    is a file-in/file-out tool over a whole mask; this is the same arithmetic (cosine similarity,
    top-1) on a handful of GT points. Pixel features are L2-normalized to match compute_soft_snap,
    which normalizes them too; vertex features come out of EncoderL2Norm already normalized, so the
    dot product is a cosine either way and the argmax is unaffected by the normalization.
    """
    pf = F.normalize(pixel_features.detach().to(device).float(), dim=-1)
    vf = vertex_features.to(device).float()
    argmax_idx = (pf @ vf.T).argmax(1).cpu()
    soft_idx, _ = compute_soft_snap(pixel_features, vertex_features, vertex_positions,
                                    k=k, chunk_size=chunk_size, device=device,
                                    temperature=temperature)
    return argmax_idx.numpy(), soft_idx.numpy()


def summarize(errs):
    errs = np.asarray(errs, dtype=np.float64)
    if errs.size == 0:
        return {'n_points': 0, 'mean': None, 'median': None}
    return {'n_points': int(errs.size), 'mean': float(errs.mean()),
            'median': float(np.median(errs)), 'max': float(errs.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True, help="the instance's directory (holds <tag>_gt.pth)")
    ap.add_argument('--tag', required=True, help='e.g. lvis117683')
    ap.add_argument('--arm', action='append', required=True, metavar='NAME=PATH',
                    help='an arm to score, as <name>=<pred_f.pth>. Repeatable, and REPEAT IT rather '
                         'than invoking this script once per arm: the dominant cost here is not the '
                         'scoring (a handful of GT points) but the startup -- ~60s per process for '
                         'imports plus the 4.8GB pixel-feature file. Every arm named in one call '
                         'shares that, so two arms cost ~60s instead of ~120s. Typical: '
                         '--arm ours=<p2>/alpha/alpha0.5/pred_f.pth --arm zs=<phase1>/pred_f.pth')
    ap.add_argument('--pixel_features_path', default=None,
                    help='defaults to <run_dir>/<tag>_pix_sdsquare_fuse16.pth')
    ap.add_argument('--obj_path', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--geodesics_path', required=True,
                    help='cached [V, V] heat geodesics, max normalized to 228 (built by '
                         'analysis/eval_densepose_lvis.py; same mesh, so it is reused verbatim)')
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--k_vertices_per_pixel', type=int, default=10)
    ap.add_argument('--read_temperature', type=float, default=0.01)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    device = torch.device(args.device)

    gt = torch.load(os.path.join(args.run_dir, f'{args.tag}_gt.pth'), map_location='cpu')
    rows, cols = gt['rows'].numpy(), gt['cols'].numpy()
    gt_vertex = gt['vertex'].numpy()
    in_mask = gt['in_mask'].numpy().astype(bool)

    pix_path = args.pixel_features_path or os.path.join(
        args.run_dir, f'{args.tag}_pix_sdsquare_fuse16.pth')
    pix = torch.load(pix_path, map_location='cpu')          # [H, W, F]
    pixel_features = pix[rows, cols]                         # [P, F] -- only the GT pixels
    del pix

    mesh = loadmesh(dir=args.obj_path, name=args.name)
    vertex_positions = mesh.vertices.detach().cpu()
    G = np.load(args.geodesics_path)

    out = {'tag': args.tag, 'ann_id': int(gt['ann_id']),
           'read_temperature': args.read_temperature, 'k_vertices_per_pixel': args.k_vertices_per_pixel,
           'n_points': int(len(gt_vertex)), 'n_points_in_mask': int(in_mask.sum()),
           'gt_vertex': [int(v) for v in gt_vertex], 'in_mask': [bool(v) for v in in_mask],
           'arms': {}}

    for spec in args.arm:
        if '=' not in spec:
            raise ValueError(f'--arm must be <name>=<path>, got {spec!r}')
        arm_name, arm_path = spec.split('=', 1)

        vertex_features = torch.load(arm_path, map_location='cpu')
        if isinstance(vertex_features, dict):
            vertex_features = vertex_features.get('pred_f', vertex_features)
        if G.shape[0] != vertex_features.shape[0]:
            raise ValueError(f'geodesic matrix is {G.shape} but {arm_name}\'s pred_f has '
                             f'{vertex_features.shape[0]} vertices -- different mesh, refusing to score')

        argmax_idx, soft_idx = predict(pixel_features, vertex_features, vertex_positions,
                                       args.k_vertices_per_pixel, args.read_temperature,
                                       device, args.chunk_size)

        arm = {'vertex_features_path': arm_path}
        for rule, pred in (('soft_snap', soft_idx), ('argmax', argmax_idx)):
            errs = G[gt_vertex, pred]
            arm[rule] = {'all': summarize(errs), 'in_mask': summarize(errs[in_mask]),
                         'per_point': [float(e) for e in errs],
                         'pred_vertex': [int(v) for v in pred]}
        out['arms'][arm_name] = arm

        s, a = arm['soft_snap']['all'], arm['argmax']['all']
        print(f"{args.tag}  {arm_name:>8s}: {out['n_points']} pts "
              f"({out['n_points_in_mask']} in mask)  "
              f"SOFT-SNAP mean {s['mean']:6.2f} (median {s['median']:6.2f})  |  "
              f"argmax mean {a['mean']:6.2f} (median {a['median']:6.2f})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(out, open(args.out_json, 'w'), indent=1)
    print(f'wrote {args.out_json}')


if __name__ == '__main__':
    main()
