"""Spatial-smoothness diagnostic for a phase-1/phase-2 pred_f.pth checkpoint: samples random
vertex pairs, correlates geometric distance (bounding-sphere-normalized mesh space) against
feature cosine similarity. Reproduces the method described in PROJECT_NOTES.md's 2026-08-19
entries (plain DINO phase-1 corr -0.850, SD-DINO Tanh+MSE -0.270, SD-DINO L2norm -0.80) -- no
script implementing it was ever saved at the time (confirmed via repo-wide search), so this is
that diagnostic made reusable rather than ad hoc, kept additive/standalone per this project's
convention.

Run:
cd $DISC_ROOT && python analysis/vertex_smoothness_corr.py \
  --pred_f_path experiments/guitar/encoder_dino_sd_frontcentered_l2norm_axisfix_phase1/pred_f.pth \
  --obj_path ./meshes/guitar.obj --n_pairs 3000 --seed 0
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils import loadmesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_f_path', type=str, required=True)
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')
    parser.add_argument('--n_pairs', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)
    vertices = mesh.vertices.detach().cpu().numpy()  # already bounding-sphere-normalized by loadmesh/MeshNormalizer

    pred_f = torch.load(args.pred_f_path, map_location='cpu').detach()
    features = pred_f.numpy()

    assert vertices.shape[0] == features.shape[0], \
        f"vertex count {vertices.shape[0]} != feature count {features.shape[0]}"

    rng = np.random.default_rng(args.seed)
    n = vertices.shape[0]
    idx_a = rng.integers(0, n, size=args.n_pairs)
    idx_b = rng.integers(0, n, size=args.n_pairs)
    keep = idx_a != idx_b
    idx_a, idx_b = idx_a[keep], idx_b[keep]

    dist = np.linalg.norm(vertices[idx_a] - vertices[idx_b], axis=1)

    fa, fb = features[idx_a], features[idx_b]
    fa_n = fa / (np.linalg.norm(fa, axis=1, keepdims=True) + 1e-8)
    fb_n = fb / (np.linalg.norm(fb, axis=1, keepdims=True) + 1e-8)
    sim = (fa_n * fb_n).sum(axis=1)

    corr = np.corrcoef(dist, sim)[0, 1]
    print(f"n_pairs used: {len(dist)}")
    print(f"corr(dist, sim): {corr:.3f}")

    order = np.argsort(dist)
    dist_sorted, sim_sorted = dist[order], sim[order]
    quintile_means = [sim_sorted[i * len(sim_sorted) // 5:(i + 1) * len(sim_sorted) // 5].mean()
                       for i in range(5)]
    print("sim by distance quintile (near->far):", " / ".join(f"{v:.3f}" for v in quintile_means))

    norms = np.linalg.norm(features, axis=1)
    print(f"feature norm: min {norms.min():.3f}, median {np.median(norms):.3f}, "
          f"max {norms.max():.3f}, max/median ratio {norms.max() / np.median(norms):.2f}x")


if __name__ == '__main__':
    main()
