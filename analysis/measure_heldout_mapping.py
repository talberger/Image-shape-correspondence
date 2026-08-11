"""Does the mapping loss generalize, or only fit the 2048 training pixels?

There is no validation set here -- one image, one shape -- so this is the only
generalization evidence available. It evaluates the mapping objective itself,
mean ||y_hat_i - y_hat_l|| over a pixel and its 8 image neighbours, on two
disjoint pixel sets:

  train    the exact 2048 centers the encoder was trained on, reproduced by
           calling sample_center_pixels with the training seed
  held-out an equally-sized sample drawn from the same eligible pixels with
           those 2048 removed

A large train/held-out gap means the encoder memorized its training pixels. A
small one means it learned a property of the feature field. (On the 2026-08-01
run the gap was 9%, which is what said the objective itself was being learned
even though the correspondence degraded -- so read this alongside
pixel_to_vertex_nn.py's coverage and similarity, never on its own.)

Takes a pred_f.pth rather than a checkpoint, so it works on phase-1 directly and
on any phase-2 step via analysis/export_pred_f.py.

Run from /workspace/bsb, e.g.:
  python analysis/measure_heldout_mapping.py \
      --vertex_features_path ./experiments/guitar/encoder_dino/pred_f.pth
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.mapping_loss import (build_adjacent_neighbors, sample_center_pixels,
                                  soft_map_to_vertices)
from modules.utils import device, loadmesh


def load_map(path):
    tensor = torch.load(path, map_location='cpu')
    return np.asarray(tensor).squeeze()


def evaluate(name, coords, segment_ids, fine_segments, all_pixel_features, offset,
             vertex_features, vertices, k, stride):
    """Mean/median ||y_hat_i - y_hat_l|| over one pixel set and its neighbours.

    `all_pixel_features` holds every pixel this script needs, in the flat
    [centers, then neighbours row-major] layout the trainer uses; `offset` is
    where this set starts in it.
    """
    num_centers = coords.shape[0]
    _, neighbor_valid = build_adjacent_neighbors(
        coords, segment_ids, fine_segments, stride=stride)
    num_neighbors = neighbor_valid.shape[1]
    span = num_centers * (1 + num_neighbors)

    with torch.no_grad():
        soft = soft_map_to_vertices(all_pixel_features[offset:offset + span],
                                    vertex_features, vertices, k)
    centers = soft[:num_centers]
    neighbors = soft[num_centers:].view(num_centers, num_neighbors, 3)

    valid = neighbor_valid.to(soft.device)
    distance = (centers.unsqueeze(1) - neighbors).pow(2).sum(-1).sqrt()[valid]
    squared = distance.pow(2)

    print('%-9s %d centers, %d valid pairs' % (name, num_centers, int(valid.sum())))
    print('    loss (mean d^2)     %.6e' % squared.mean().item())
    print('    mean   ||y_i-y_l||  %.6e' % distance.mean().item())
    print('    median ||y_i-y_l||  %.6e' % distance.median().item())
    return distance.mean().item(), span


def main(args):
    coarse_mask = load_map(args.object_mask_path).astype(bool)
    fine_segments = load_map(args.fine_seg_path)

    train_coords, _, train_segments = sample_center_pixels(
        coarse_mask, fine_segments, args.num_center_pixels, seed=args.seed)

    # Held-out = the same eligible population with the training centers removed.
    # Punching them out of the coarse mask reuses sample_center_pixels exactly
    # rather than reimplementing its eligibility rule, which is what keeps the
    # two sets drawn the same way.
    remaining = coarse_mask.copy()
    remaining[train_coords[:, 0].numpy(), train_coords[:, 1].numpy()] = False
    heldout_coords, _, heldout_segments = sample_center_pixels(
        remaining, fine_segments, args.num_heldout_pixels, seed=args.heldout_seed)
    assert not set(map(tuple, train_coords.tolist())) & set(map(tuple, heldout_coords.tolist()))

    # One pass over the 3.6 GB feature file for both sets.
    train_neighbors, _ = build_adjacent_neighbors(
        train_coords, train_segments, fine_segments, stride=args.neighbor_stride)
    heldout_neighbors, _ = build_adjacent_neighbors(
        heldout_coords, heldout_segments, fine_segments, stride=args.neighbor_stride)
    all_coords = torch.cat([train_coords, train_neighbors.reshape(-1, 2),
                            heldout_coords, heldout_neighbors.reshape(-1, 2)], dim=0)

    print('loading pixel features for %d pixels...' % len(all_coords))
    features = torch.load(args.pix_dino_features_path, map_location='cpu')
    pixel_features = features[all_coords[:, 0], all_coords[:, 1]].detach().to(device)
    del features
    pixel_features = torch.nn.functional.normalize(pixel_features.float(), dim=-1)

    mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)
    vertex_features = torch.load(args.vertex_features_path, map_location=device).detach()
    print('vertex features %s from %s\n'
          % (tuple(vertex_features.shape), args.vertex_features_path))

    train_mean, span = evaluate('train', train_coords, train_segments, fine_segments,
                                pixel_features, 0, vertex_features, mesh.vertices,
                                args.k_vertices_per_pixel, args.neighbor_stride)
    heldout_mean, _ = evaluate('held-out', heldout_coords, heldout_segments, fine_segments,
                               pixel_features, span, vertex_features, mesh.vertices,
                               args.k_vertices_per_pixel, args.neighbor_stride)

    print('\nheld-out / train = %.4f  (1.00 = generalizes perfectly; '
          'much above 1 = memorized the training pixels)'
          % (heldout_mean / train_mean))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vertex_features_path', type=str,
                        default='./experiments/guitar/encoder_dino/pred_f.pth',
                        help='pred_f.pth to evaluate; phase-1 by default')
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')
    parser.add_argument('--object_mask_path', type=str, required=True)
    parser.add_argument('--fine_seg_path', type=str,
                        default='./demo/guitar/phase2/guitar_01_fine_segments.pth')
    parser.add_argument('--pix_dino_features_path', type=str, required=True)

    # These three must match the training run or "train" is not the train set.
    parser.add_argument('--num_center_pixels', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--neighbor_stride', type=int, default=1)

    parser.add_argument('--k_vertices_per_pixel', type=int, default=10)
    parser.add_argument('--num_heldout_pixels', type=int, default=2048)
    parser.add_argument('--heldout_seed', type=int, default=1)
    main(parser.parse_args())
