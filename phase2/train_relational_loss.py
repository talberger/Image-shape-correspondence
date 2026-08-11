"""Phase-2 encoder training: unified relational (distance-proportional) loss.

`train_mapping_loss.py` trains on two separately-weighted signals -- an
attractive mapping_loss and a repulsive hinge repulsion_loss, combined as
`lambda_map * map_loss + lambda_repel * repel_loss` -- which needed a
hand-derived `repel_margin` to reconcile pixel-space and mesh-space distance
scales. This is a separate experiment, starting from the same phase-1
checkpoint, that replaces the split with one continuous loss instead: for a
sample of center pixels and randomly-drawn in-segment/out-of-segment
neighbours (resampled every step), penalize pixel-space distance and
mesh-space distance for not being proportional -- see
modules/relational_loss.py for the full derivation and the Relational
Knowledge Distillation (Park et al., CVPR 2019) formalization it's based on.

train_mapping_loss.py and modules/mapping_loss.py are not modified by this
script -- those runs (encoder_dino_mapping_adjacent, encoder_dino_mapping_repel,
encoder_dino_mapping_repel_wide300) stay exactly reproducible. This script
imports several of train_mapping_loss.py's helpers directly (safe: that file
guards its argument-parsing/training invocation behind `if __name__ ==
'__main__':`) rather than duplicating them.

FIRST RUN (300 steps, 2026-08-10) COLLAPSED -- vertex coverage fell to 0.27%,
worse than mapping_loss's own original collapse. Root cause: normalizing
d_3D by a live, every-step mean is invariant to every pairwise 3D distance
shrinking together in proportion, and nothing stopped the model drifting
there over 300 steps of resampled-every-step neighbours. Two independent
fixes were designed to isolate which change actually matters (2026-08-11,
A/B, see `--live_mu_3d`/`--fixed_neighbors` below and modules/relational_loss.py's
module docstring for the full argument):

  Version 1 (default, both flags off): FIXED mu_3D_reference, measured once
    from the phase-1 encoder (measure_mu_3D_reference below); neighbours
    still resampled every step. The 2026-08-10 end-of-day fix.
  Version 2 (--live_mu_3d --fixed_neighbors): mu_3D back to LIVE (the
    original behavior); neighbours sampled ONCE and reused every step
    instead. Tal's 2026-08-11 hypothesis.

Run (from /workspace/bsb, so `modules.*` and `phase2.*` both resolve):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python -m phase2.train_relational_loss \\
    --object_mask_path <coarse mask> --pix_dino_features_path <pixel features> \\
    [--live_mu_3d --fixed_neighbors]   # add for Version 2

See /workspace/PROJECT_NOTES.md for the full design discussion.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from modules.mapping_loss import feature_drift, sample_center_pixels, soft_map_to_vertices
from modules.relational_loss import (build_segment_index, relational_loss,
                                     sample_in_segment_neighbors, sample_out_segment_neighbors)
from modules.utils import device, loadmesh
from phase2.train_mapping_loss import (assert_resume_is_sane, get_fourier_basis, hash_input,
                                       load_phase1_encoder, set_fourier_basis)


def index_pixel_features(all_features, coords):
    """Per-step equivalent of train_mapping_loss.py's load_pixel_features,
    minus the disk read.

    That function re-reads the 3.6 GB [H, W, 1024] file from disk every call,
    which is fine there -- it's called once, since that script's neighbours
    are fixed for the whole run. Ours are resampled every step, so all_coords
    changes every step; re-reading the file from disk each time would mean
    a full 3.6 GB load + index + free per step, and was in fact the dominant
    cost the first time this script ran (~5.5s/step just in this call,
    measured). Fixed by reading the file once into `all_features` before the
    loop and only indexing it here -- same normalization, same detach, same
    reasoning (see load_pixel_features's docstring), just no repeated I/O.
    """
    features = all_features[coords[:, 0], coords[:, 1]].detach().to(device)
    return F.normalize(features.float(), dim=-1)


def measure_mu_3D_reference(encoder, vertices, coords, segment_ids, seg_idx, all_pixel_features,
                            k_vertices_per_pixel, num_neighbors_in_segment,
                            num_neighbors_out_segment, num_samples=5):
    """The FIXED mu_3D normalizer relational_loss() needs -- measured once,
    here, from the untouched phase-1 encoder, before any relational-loss
    training happens.

    The first version of relational_loss() recomputed mu_3D from the live,
    currently-training encoder's own output every step. That collapsed the
    first time it ran (postmortem in PROJECT_NOTES.md, 2026-08-10): a loss
    that only ever compares distances to its OWN batch's mean cannot see or
    resist every pairwise 3D distance shrinking together in proportion, so
    that direction was a free ride for 300 steps of gradient noise to drift
    down, and it did. A reference measured once, from a model this script
    has not yet touched, has nothing to do with wherever training later
    drifts -- restoring the role `repel_margin` played before normalization
    replaced it, just measured empirically instead of hand-picked.

    Averaged over `num_samples` independent random neighbour draws (the same
    sample_in_segment_neighbors/sample_out_segment_neighbors the training
    loop itself calls) rather than trusting a single draw, since any one
    draw of 2048*16 pairs is itself a sample with some variance.

    Returns a plain float.
    """
    num_centers = coords.shape[0]
    num_neighbors = num_neighbors_in_segment + num_neighbors_out_segment
    with torch.no_grad():
        vertex_features = encoder(vertices)
        draws = []
        for _ in range(num_samples):
            neighbor_coords_in = sample_in_segment_neighbors(
                coords, segment_ids, seg_idx, k=num_neighbors_in_segment)
            neighbor_coords_out = sample_out_segment_neighbors(
                coords, segment_ids, seg_idx, k=num_neighbors_out_segment)
            neighbor_coords = torch.cat([neighbor_coords_in, neighbor_coords_out], dim=1)

            all_coords = torch.cat([coords, neighbor_coords.reshape(-1, 2)], dim=0)
            pixel_features = index_pixel_features(all_pixel_features, all_coords)
            soft_points = soft_map_to_vertices(
                pixel_features, vertex_features, vertices, k_vertices_per_pixel)

            center_points = soft_points[:num_centers]
            neighbor_points = soft_points[num_centers:].view(num_centers, num_neighbors, 3)
            d_3D = (center_points.unsqueeze(1) - neighbor_points).pow(2).sum(-1).sqrt()
            draws.append(d_3D.mean().item())

    return sum(draws) / len(draws)


def save_loss_plot(losses, drifts, checkpoint_every, path):
    fig, (ax_loss, ax_drift) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_loss.plot(losses)
    ax_loss.set_ylabel('relational loss')
    ax_loss.set_title('phase-2 relational loss training')
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

    # Everything here stays on CPU: these coordinates index the 3.6 GB pixel
    # feature file below, and torch rejects a CUDA index against a CPU source.
    coarse_mask = torch.load(args.object_mask_path, map_location='cpu').numpy()
    fine_segments = torch.load(args.fine_seg_path, map_location='cpu').numpy()

    # Centers: sampled once, fixed for the whole run (same convention as
    # train_mapping_loss.py). Only the 16 neighbours per center are resampled
    # every step -- see modules/relational_loss.py for why.
    coords, _, segment_ids = sample_center_pixels(
        coarse_mask, fine_segments, args.num_center_pixels, seed=args.seed)
    num_centers = coords.shape[0]
    print('sampled %d center pixels across %d segments'
          % (num_centers, len(torch.unique(segment_ids))))

    seg_idx = build_segment_index(fine_segments, coarse_mask)
    num_neighbors = args.num_neighbors_in_segment + args.num_neighbors_out_segment

    # 2026-08-11 A/B test: which 2026-08-10 collapse fix actually matters,
    # the FIXED mu_3D_reference or FIXED (not resampled) neighbours? Both
    # default to the already-verified-safe combination (fixed reference,
    # resampled neighbours -- Version 1 below); --fixed_neighbors/
    # --live_mu_3d switch each axis independently. See
    # modules/relational_loss.py's module docstring for the full argument.
    if args.fixed_neighbors:
        neighbor_coords_in = sample_in_segment_neighbors(
            coords, segment_ids, seg_idx, k=args.num_neighbors_in_segment)
        neighbor_coords_out = sample_out_segment_neighbors(
            coords, segment_ids, seg_idx, k=args.num_neighbors_out_segment,
            max_retries=args.out_segment_max_retries)
        fixed_neighbor_coords = torch.cat([neighbor_coords_in, neighbor_coords_out], dim=1)
        print('  %d in-segment + %d out-of-segment neighbours per center, sampled ONCE '
              '(--fixed_neighbors) and reused every step'
              % (args.num_neighbors_in_segment, args.num_neighbors_out_segment))
    else:
        print('  %d in-segment + %d out-of-segment neighbours per center, resampled every step'
              % (args.num_neighbors_in_segment, args.num_neighbors_out_segment))

    # Loaded once, kept resident for the whole run (~3.6 GB) and indexed
    # in-memory every step -- see index_pixel_features's docstring for why
    # this can't reuse load_pixel_features's per-call disk read here.
    all_pixel_features = torch.load(args.pix_dino_features_path, map_location='cpu')

    if args.live_mu_3d:
        # LIVE mode: mu_3D recomputed from the live encoder every step, same
        # as the original (pre-2026-08-10-fix) relational_loss(). Known to
        # collapse when paired with resampled-every-step neighbours; this
        # session is testing it paired with --fixed_neighbors instead.
        mu_3D_reference = None
        print('  mu_3D: LIVE (--live_mu_3d) -- recomputed from the training encoder every step')
    else:
        # FIXED normalizer for relational_loss's d_3D, measured once from the
        # untouched phase-1 encoder -- see measure_mu_3D_reference's docstring
        # for why this must not be recomputed from the live encoder every step.
        mu_3D_reference = measure_mu_3D_reference(
            encoder, vertices, coords, segment_ids, seg_idx, all_pixel_features,
            args.k_vertices_per_pixel, args.num_neighbors_in_segment,
            args.num_neighbors_out_segment, num_samples=args.mu_3d_reference_samples)
        print('  mu_3D reference (fixed, phase-1 encoder, averaged over %d draws): %.5f'
              % (args.mu_3d_reference_samples, mu_3D_reference))

    optimizer = torch.optim.Adam(encoder.parameters(), args.lr, eps=args.adam_eps)
    os.makedirs(args.out_dir, exist_ok=True)

    losses = []
    drifts = []
    for step in range(1, args.num_iters + 1):
        optimizer.zero_grad()

        if args.fixed_neighbors:
            neighbor_coords = fixed_neighbor_coords
        else:
            neighbor_coords_in = sample_in_segment_neighbors(
                coords, segment_ids, seg_idx, k=args.num_neighbors_in_segment)
            neighbor_coords_out = sample_out_segment_neighbors(
                coords, segment_ids, seg_idx, k=args.num_neighbors_out_segment,
                max_retries=args.out_segment_max_retries)
            neighbor_coords = torch.cat([neighbor_coords_in, neighbor_coords_out], dim=1)

        # Same flat-batch pattern as train_mapping_loss.py: centers are rows
        # [:n], neighbour (i, j) is row n + i*num_neighbors + j.
        all_coords = torch.cat([coords, neighbor_coords.reshape(-1, 2)], dim=0)
        pixel_features = index_pixel_features(all_pixel_features, all_coords)

        # Live encoder output, NOT the saved pred_f.pth -- see
        # train_mapping_loss.py's identical comment for why.
        vertex_features = encoder(vertices)
        soft_points = soft_map_to_vertices(
            pixel_features, vertex_features, vertices, args.k_vertices_per_pixel)

        center_points = soft_points[:num_centers]
        neighbor_points = soft_points[num_centers:].view(num_centers, num_neighbors, 3)

        loss, mu_2D, mu_3D_live = relational_loss(
            center_points, neighbor_points, coords, neighbor_coords,
            mu_3D_reference=mu_3D_reference, huber_delta=args.huber_delta)

        loss.backward()

        if step == 1:
            grad_norm = torch.cat([p.grad.reshape(-1) for p in encoder.parameters()
                                   if p.grad is not None]).norm().item()
            print('step    1  total grad norm %.4e' % grad_norm)

        optimizer.step()
        losses.append(loss.item())

        if step % args.checkpoint_every == 0 or step == args.num_iters:
            with torch.no_grad():
                drift = feature_drift(encoder(vertices), phase1_features)
            drifts.append((step, drift))
            ref_str = 'live' if mu_3D_reference is None else '%.5f' % mu_3D_reference
            print('step %4d/%d  loss %.6f (mu_2D=%.3f mu_3D_live=%.5f, ref=%s)  '
                  'drift(cos vs phase 1) %.4f'
                  % (step, args.num_iters, loss.item(), mu_2D, mu_3D_live,
                     ref_str, drift))

            torch.save({'encoder_state_dict': encoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'fourier_basis': get_fourier_basis(encoder),
                        'step': step, 'loss': loss.item(), 'drift': drift},
                       os.path.join(args.out_dir, 'encoder_checkpoint_step%d.pth' % step))

            if args.stop_at_drift is not None and drift < args.stop_at_drift:
                print('drift %.4f fell below --stop_at_drift %.4f; stopping.'
                      % (drift, args.stop_at_drift))
                break

    # Final artifacts, mirroring phase 1's / train_mapping_loss.py's layout so
    # the analysis scripts can consume this checkpoint exactly like theirs.
    torch.save({'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'fourier_basis': get_fourier_basis(encoder),
                'losses': losses, 'drifts': drifts},
               os.path.join(args.out_dir, 'encoder_checkpoint.pth'))
    with torch.no_grad():
        torch.save(encoder(vertices), os.path.join(args.out_dir, 'pred_f.pth'))

    save_loss_plot(losses, drifts, args.checkpoint_every,
                   os.path.join(args.out_dir, 'relational_loss.png'))
    config = {k: v for k, v in vars(args).items()}
    config['input_sha256'] = {
        path: hash_input(path) for path in (args.object_mask_path, args.fine_seg_path,
                                            args.pix_dino_features_path)}
    with open(os.path.join(args.out_dir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

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
    parser.add_argument('--object_mask_path', type=str, required=True,
                        help='coarse whole-object mask; center pixels are drawn from it')
    parser.add_argument('--fine_seg_path', type=str,
                        default='./demo/guitar/phase2/guitar_01_fine_segments.pth',
                        help='part-level label map from analysis/build_fine_segments.py; '
                             'defines the in-segment/out-of-segment split neighbours are '
                             'drawn from')
    parser.add_argument('--pix_dino_features_path', type=str, required=True)

    # Loss
    parser.add_argument('--num_center_pixels', type=int, default=2048,
                        help='pixels the loss is computed on; sampled once, then fixed')
    parser.add_argument('--k_vertices_per_pixel', type=int, default=10,
                        help="mesh vertices each pixel is softly mapped onto, nearest in "
                             "FEATURE space (DPC's k_cc)")
    parser.add_argument('--num_neighbors_in_segment', type=int, default=8,
                        help='random pixels drawn from a center\'s own fine segment, '
                             'without replacement, resampled every step')
    parser.add_argument('--num_neighbors_out_segment', type=int, default=8,
                        help='random pixels drawn from outside a center\'s fine segment '
                             '(segment-exclusive), resampled every step')
    parser.add_argument('--out_segment_max_retries', type=int, default=10,
                        help='vectorized rejection-sampling rounds for out-of-segment '
                             'draws that land back in the center\'s own segment')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                        help='Huber transition point on the normalized-distance residual; '
                             "1.0 is RKD's own fixed choice")
    parser.add_argument('--mu_3d_reference_samples', type=int, default=5,
                        help='independent random neighbour draws averaged together to '
                             'measure the FIXED mu_3D normalizer from the phase-1 encoder, '
                             'before training starts -- see measure_mu_3D_reference. Ignored '
                             'when --live_mu_3d is set.')
    parser.add_argument('--live_mu_3d', action='store_true',
                        help='2026-08-11 A/B test, axis 1: normalize d_3D by THIS STEP\'s own '
                             'live mean (the original, pre-2026-08-10-fix behavior) instead of '
                             'a fixed reference measured once from the phase-1 encoder. Off by '
                             'default (matches the verified-safe 2026-08-10 fix).')
    parser.add_argument('--fixed_neighbors', action='store_true',
                        help='2026-08-11 A/B test, axis 2: sample each center\'s in-segment/'
                             'out-of-segment neighbours ONCE, before training, and reuse them '
                             'every step, instead of resampling every step. Off by default '
                             '(matches the verified-safe 2026-08-10 fix).')

    # Optimization -- same discipline as train_mapping_loss.py.
    parser.add_argument('--num_iters', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_eps', type=float, default=1e-12,
                        help="far below torch's 1e-8 default -- see "
                             'train_mapping_loss.py\'s identical flag for why')
    parser.add_argument('--checkpoint_every', type=int, default=25)
    parser.add_argument('--stop_at_drift', type=float, default=None,
                        help='stop early if mean cosine vs phase-1 features drops below '
                             'this (e.g. 0.9). Off by default.')

    parser.add_argument('--seed', type=int, default=0,
                        help='must match the phase-1 seed -- it is what restores the '
                             'Fourier basis (see load_phase1_encoder)')
    parser.add_argument('--out_dir', type=str,
                        default='./experiments/guitar/encoder_dino_relational')
    args = parser.parse_args()

    if args.init_encoder_checkpoint is None:
        args.init_encoder_checkpoint = os.path.join(args.init_encoder_dir,
                                                    'encoder_checkpoint.pth')
    train(args)
