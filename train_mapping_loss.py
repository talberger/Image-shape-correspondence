"""Phase-2 encoder training: DPC-style mapping loss.

Phase 1 (encoder.py's train_encoder) trains the per-vertex DINO encoder by
photo-consistency: render the mesh, MSE the rendered features against DINOv2
features of that render. This is a second phase that starts from the phase-1
checkpoint and trains on a completely different signal, and on that signal only:

    pixels close together in the photo should map to nearby points on the mesh.

Combining the two losses in one run is a deliberate later step, not this one.

This is a separate entrypoint rather than another stage inside encoder.py so
that the upstream (threedle/bsb) files stay untouched -- the same pattern the
analysis/ scripts already follow. `git diff baseline-fixed` then shows all of our
work without any interleaving into the original code.

Run:
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python train_mapping_loss.py

See /workspace/FINE_TUNING_PLAN.md for the full design and the reasoning behind
each default.
"""
import argparse
import json
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from modules.encoder import Encoder
from modules.mapping_loss import (build_pixel_neighbors, feature_drift, mapping_loss,
                                  sample_center_pixels, soft_map_to_vertices)
from modules.utils import device, load_state_dict, loadmesh

# Reused from the standalone gate script so the thresholds cannot drift apart.
from analysis.verify_encoder_resume import (MAX_ABS_DIFF, MIN_MEAN_COSINE,
                                            seed_everything)


def get_fourier_basis(encoder):
    """The Fourier basis `_B`, or None when positional encoding is off.

    modules/utils.py:272 keeps `_B` as a plain attribute rather than a registered
    buffer, so state_dict() does not contain it. Our checkpoints therefore have
    to carry it explicitly, otherwise they inherit the same silent-corruption
    trap as the phase-1 ones: they would load fine and be wrong unless the reader
    happens to reproduce the exact RNG sequence.
    """
    first_layer = encoder.mlp[0]
    return getattr(first_layer, '_B', None)


def set_fourier_basis(encoder, basis):
    if basis is not None:
        encoder.mlp[0]._B = basis


def load_phase1_encoder(args):
    """Rebuild the phase-1 Encoder, load its weights, and prove it is intact.

    The seeding is load-bearing, not hygiene. FourierFeatureTransform's random
    basis `_B` (modules/utils.py:272) is a plain attribute rather than a
    registered buffer, so it is NOT in state_dict() and NOT in the checkpoint.
    Re-seeding identically to phase 1 and constructing the Encoder as the first
    RNG consumer is what restores it; otherwise the model loads without error
    and is silently wrong.

    Verified against pred_f.pth rather than assumed -- that file was written by
    the process that still held the correct `_B` in memory (encoder.py:363-364),
    which makes it exact ground truth.
    """
    seed_everything(args.seed)
    encoder = Encoder(args.depth, args.width, out_dim=args.n_classes,
                      positional_encoding=args.positional_encoding,
                      sigma=args.sigma, network_verbose=0).to(device)

    checkpoint = torch.load(args.init_encoder_checkpoint, map_location=device)
    load_state_dict(encoder, checkpoint['encoder_state_dict'])
    return encoder


def assert_resume_is_sane(encoder, vertices, pred_f_path):
    # detach: pred_f.pth was also saved with requires_grad=True upstream
    # (encoder.py:363 runs outside no_grad), and it is a fixed reference here.
    reference = torch.load(pred_f_path, map_location=device).detach()
    with torch.no_grad():
        pred = encoder(vertices)

    mean_cosine = F.cosine_similarity(pred, reference, dim=-1).mean().item()
    max_abs_diff = (pred - reference).abs().max().item()
    print('[gate] resume check: mean cosine %.6f (need >= %.4f), '
          'max abs diff %.3e (need <= %.0e)'
          % (mean_cosine, MIN_MEAN_COSINE, max_abs_diff, MAX_ABS_DIFF))

    if mean_cosine < MIN_MEAN_COSINE or max_abs_diff > MAX_ABS_DIFF:
        raise RuntimeError(
            'Reloaded encoder does not reproduce %s. The Fourier basis _B was '
            'most likely re-randomized (see modules/utils.py:272). Training now '
            'would produce meaningless results. Run '
            'analysis/verify_encoder_resume.py --control to investigate.' % pred_f_path)
    print('[gate] passed -- Fourier basis restored, features match phase 1.')
    return reference


def load_pixel_features(path, coords):
    """Read the per-pixel DINO features for the center pixels only.

    The file is ~3.6 GB ([H, W, 1024] over the whole image). We index out the
    few thousand rows we need and drop the rest before training starts, so the
    loop never touches the big tensor.

    The detach is required, not cosmetic: this file was saved upstream with
    requires_grad=True, so without it the normalize below becomes a graph node
    built once outside the training loop, and the second step's backward dies
    with "trying to backward through the graph a second time". These features
    are a frozen target -- no gradient should reach them.
    """
    all_features = torch.load(path, map_location='cpu')
    features = all_features[coords[:, 0], coords[:, 1]].clone().to(device).detach()
    del all_features
    # Frozen, so normalize once here instead of every step.
    return F.normalize(features.float(), dim=-1)


def save_loss_plot(losses, drifts, checkpoint_every, path):
    fig, (ax_loss, ax_drift) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_loss.plot(losses)
    ax_loss.set_ylabel('mapping loss')
    ax_loss.set_title('phase-2 mapping loss training')
    ax_loss.grid(alpha=0.3)

    if drifts:
        steps, values = zip(*drifts)
        ax_drift.plot(steps, values, marker='o')
        ax_drift.set_ylabel('mean cosine vs phase 1')
        ax_drift.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax_drift.set_xlabel('step')
    ax_drift.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def train(args):
    mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)
    vertices = mesh.vertices
    print('device=%s  vertices=%s' % (device, tuple(vertices.shape)))

    encoder = load_phase1_encoder(args)
    phase1_features = assert_resume_is_sane(
        encoder, vertices, os.path.join(args.init_encoder_dir, 'pred_f.pth'))

    coarse_mask = torch.load(args.object_mask_path, map_location='cpu').numpy()
    fine_segments = torch.load(args.fine_seg_path, map_location='cpu').numpy()

    coords, normalized_coords, segment_ids = sample_center_pixels(
        coarse_mask, fine_segments, args.num_center_pixels, seed=args.seed)
    coords = coords.to(device)
    normalized_coords = normalized_coords.to(device)
    segment_ids = segment_ids.to(device)

    neighbor_idx, neighbor_valid = build_pixel_neighbors(
        coords, segment_ids, args.k_pixel_neighbors)

    num_segments = len(torch.unique(segment_ids))
    pairs = int(neighbor_valid.sum().item())
    full = int((neighbor_valid.sum(dim=1) == args.k_pixel_neighbors).sum().item())
    print('sampled %d center pixels across %d segments' % (len(coords), num_segments))
    print('  %d neighbour pairs; %d/%d pixels have the full %d neighbours'
          % (pairs, full, len(coords), args.k_pixel_neighbors))

    pixel_features = load_pixel_features(args.pix_dino_features_path, coords.cpu())
    print('  pixel features %s' % (tuple(pixel_features.shape),))

    optimizer = torch.optim.Adam(encoder.parameters(), args.lr)
    os.makedirs(args.out_dir, exist_ok=True)

    losses = []
    drifts = []
    for step in range(1, args.num_iters + 1):
        optimizer.zero_grad()

        # Live encoder output, NOT the saved pred_f.pth: that file is a frozen
        # snapshot with no link to the weights, so using it would give zero
        # gradient and train nothing.
        vertex_features = encoder(vertices)

        soft_points = soft_map_to_vertices(
            pixel_features, vertex_features, vertices, args.k_vertices_per_pixel)
        loss = mapping_loss(soft_points, neighbor_idx, neighbor_valid,
                            normalized_coords=normalized_coords,
                            weighting=args.pixel_weighting, alpha=args.alpha)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % args.checkpoint_every == 0 or step == args.num_iters:
            with torch.no_grad():
                drift = feature_drift(encoder(vertices), phase1_features)
            drifts.append((step, drift))
            print('step %4d/%d  loss %.6f  drift(cos vs phase 1) %.4f'
                  % (step, args.num_iters, loss.item(), drift))

            torch.save({'encoder_state_dict': encoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'fourier_basis': get_fourier_basis(encoder),
                        'step': step, 'loss': loss.item(), 'drift': drift},
                       os.path.join(args.out_dir, 'encoder_checkpoint_step%d.pth' % step))

            if args.stop_at_drift is not None and drift < args.stop_at_drift:
                print('drift %.4f fell below --stop_at_drift %.4f; stopping.'
                      % (drift, args.stop_at_drift))
                break

    # Final artifacts, mirroring phase 1's layout so the analysis scripts can
    # consume this checkpoint exactly like the phase-1 one.
    torch.save({'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'fourier_basis': get_fourier_basis(encoder),
                'losses': losses, 'drifts': drifts},
               os.path.join(args.out_dir, 'encoder_checkpoint.pth'))
    with torch.no_grad():
        torch.save(encoder(vertices), os.path.join(args.out_dir, 'pred_f.pth'))

    save_loss_plot(losses, drifts, args.checkpoint_every,
                   os.path.join(args.out_dir, 'mapping_loss.png'))
    with open(os.path.join(args.out_dir, 'run_config.json'), 'w') as f:
        json.dump({k: v for k, v in vars(args).items()}, f, indent=2)

    print('\nloss %.6f -> %.6f over %d steps' % (losses[0], losses[-1], len(losses)))
    print('final drift vs phase 1: %.4f' % drifts[-1][1])
    print('saved to %s' % args.out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Mesh / phase-1 checkpoint. The Encoder args MUST match the phase-1 run
    # (README step 2 used --model_type dino --width 1024 --n_classes 1024) or
    # the state dict will not load.
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')
    parser.add_argument('--init_encoder_dir', type=str,
                        default='./experiments/guitar/encoder_dino')
    parser.add_argument('--init_encoder_checkpoint', type=str, default=None,
                        help='defaults to <init_encoder_dir>/encoder_checkpoint.pth')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--n_classes', type=int, default=1024)
    parser.add_argument('--positional_encoding', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=5.0)

    # Data
    parser.add_argument('--object_mask_path', type=str,
                        default='./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2.pth',
                        help='coarse whole-object mask; center pixels are drawn from it')
    parser.add_argument('--fine_seg_path', type=str,
                        default='./demo/guitar/phase2/guitar_01_fine_segments.pth',
                        help='part-level label map from analysis/build_fine_segments.py; '
                             'pixel neighbours are restricted to the same part')
    parser.add_argument('--pix_dino_features_path', type=str,
                        default='./demo/guitar/bsb/guitar_01_pix_dino_features.pth')

    # Loss. Names spell out which space each neighbourhood lives in, since that
    # distinction is the whole point (DPC calls them k_cc and k_m).
    parser.add_argument('--num_center_pixels', type=int, default=2048,
                        help='pixels the loss is computed on; sampled once, then fixed')
    parser.add_argument('--k_vertices_per_pixel', type=int, default=10,
                        help="mesh vertices each pixel is softly mapped onto, nearest in "
                             "FEATURE space (DPC's k_cc)")
    parser.add_argument('--k_pixel_neighbors', type=int, default=7,
                        help="nearby pixels each center pixel is compared against, nearest "
                             "in 2D image space and restricted to the same segment "
                             "(DPC's k_m)")
    parser.add_argument('--pixel_weighting', type=str, default='uniform',
                        choices=['uniform', 'gaussian'],
                        help="'uniform' is the default and is what DPC effectively does; "
                             "'gaussian' only becomes meaningful once k_pixel_neighbors is "
                             'large enough for neighbour distances to vary')
    parser.add_argument('--alpha', type=float, default=8.0,
                        help='gaussian weighting only; applies to normalized coordinates')

    # Optimization. Deliberately gentler than phase 1 (3000 steps at 1e-3):
    # phase 1 saw a different rendered view every step, whereas this objective is
    # fixed, so the same budget would drift much further from the checkpoint.
    parser.add_argument('--num_iters', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_every', type=int, default=25)
    parser.add_argument('--stop_at_drift', type=float, default=None,
                        help='stop early if mean cosine vs phase-1 features drops below '
                             'this (e.g. 0.9). Off by default: log it and pick a '
                             'checkpoint afterwards.')

    parser.add_argument('--seed', type=int, default=0,
                        help='must match the phase-1 seed -- it is what restores the '
                             'Fourier basis (see load_phase1_encoder)')
    parser.add_argument('--out_dir', type=str,
                        default='./experiments/guitar/encoder_dino_mapping')
    args = parser.parse_args()

    if args.init_encoder_checkpoint is None:
        args.init_encoder_checkpoint = os.path.join(args.init_encoder_dir,
                                                    'encoder_checkpoint.pth')
    train(args)
