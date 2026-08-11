"""Phase-2 encoder training: map+repel loss with resampled neighbours.

train_mapping_loss.py's --lambda_repel path draws its near ring (attraction,
build_adjacent_neighbors, same segment, stride=1) and far ring (repulsion,
require_same_segment=False, stride=100) ONCE, before the loop, and reuses
them for the whole run -- the natural choice for a fixed-neighbourhood DPC-
style loss, since pixel coordinates never change. This is the same map+repel
run (same mapping_loss()/repulsion_loss() functions, same
lambda_map/lambda_repel/repel_margin knobs) with that one change: both rings
are resampled every training step instead of fixed for the whole run, same
idea as train_relational_loss.py's 16 neighbours -- near == in-segment
(sample_in_segment_neighbors: random pixels from the center's own fine
segment), far == out-of-segment (sample_out_segment_neighbors: random pixels
from any other segment). Segment-exclusive is stricter than the fixed run's
far ring (which only required "on the object", not a different segment) --
there's no longer a stride keeping a resampled far pixel spatially distant,
so segment exclusivity is what stands in for "far" here. Worth watching in
the results: repel_margin=0.05 was sized against the fixed run's ~100px-away
far pairs, not this one's segment-exclusive-but-possibly-nearby pairs.

Two performance traps hit while building train_relational_loss.py, avoided
here from the start rather than rediscovered:
  1. Sampling a center's same-segment neighbours by gathering its ENTIRE
     padded segment pixel array before picking a few is fine at small scale,
     catastrophic on this mesh's 136596-pixel body segment (multi-GB gather
     PER STEP, ~8.6s/step measured in relational_loss.py's own history).
     sample_in_segment_neighbors (modules/relational_loss.py) already fixes
     this -- samples indices directly, no gather -- so this script imports it
     rather than reimplementing.
  2. Re-reading the 3.6 GB pixel-DINO-feature file from disk every step
     (train_mapping_loss.py's load_pixel_features does this, harmless there
     since its neighbours are fixed and it's called once) was the dominant
     per-step cost the first time neighbours became resample-every-step
     (~5.5s/step, measured in train_relational_loss.py's history). Fixed
     there by index_pixel_features: read the file once, index the resident
     tensor per step. Reused here via the same import.

train_mapping_loss.py, modules/mapping_loss.py, and train_relational_loss.py
are none of them modified by this script -- separate experiment, so every
existing run stays exactly reproducible.

Run (from /workspace/bsb, so `modules.*` and `phase2.*` both resolve):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python -m phase2.train_mapping_repel_resampled \\
    --object_mask_path <coarse mask> --pix_dino_features_path <pixel features> \\
    --lambda_map 0.5 --lambda_repel 0.5 --num_iters 300 --checkpoint_every 25

See /workspace/PROJECT_NOTES.md for the full design discussion.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from modules.mapping_loss import (feature_drift, mapping_loss, repulsion_loss,
                                  sample_center_pixels, soft_map_to_vertices)
from modules.relational_loss import (build_segment_index, sample_in_segment_neighbors,
                                     sample_out_segment_neighbors)
from modules.utils import device, loadmesh
from phase2.train_mapping_loss import (assert_resume_is_sane, get_fourier_basis, hash_input,
                                       load_phase1_encoder)
from phase2.train_relational_loss import index_pixel_features


def save_loss_plot(losses, drifts, checkpoint_every, path):
    fig, (ax_loss, ax_drift) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_loss.plot(losses)
    ax_loss.set_ylabel('map+repel loss')
    ax_loss.set_title('phase-2 map+repel loss training (resampled neighbours)')
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
    # train_mapping_loss.py / train_relational_loss.py). Only the neighbours
    # are resampled every step.
    coords, _, segment_ids = sample_center_pixels(
        coarse_mask, fine_segments, args.num_center_pixels, seed=args.seed)
    num_centers = coords.shape[0]
    print('sampled %d center pixels across %d segments'
          % (num_centers, len(torch.unique(segment_ids))))

    seg_idx = build_segment_index(fine_segments, coarse_mask)
    use_repel = args.lambda_repel > 0
    print('  %d in-segment neighbours (map), resampled every step'
          % args.num_neighbors_in_segment)
    if use_repel:
        print('  %d out-of-segment neighbours (repel), resampled every step'
              % args.num_neighbors_out_segment)

    # Loaded once, kept resident for the whole run (~3.6 GB) and indexed
    # in-memory every step via index_pixel_features -- see the module
    # docstring above for why a per-step disk read (load_pixel_features)
    # is wrong once neighbours stop being fixed for the whole run.
    all_pixel_features = torch.load(args.pix_dino_features_path, map_location='cpu')

    optimizer = torch.optim.Adam(encoder.parameters(), args.lr, eps=args.adam_eps)
    os.makedirs(args.out_dir, exist_ok=True)

    losses = []
    drifts = []
    for step in range(1, args.num_iters + 1):
        optimizer.zero_grad()

        neighbor_coords = sample_in_segment_neighbors(
            coords, segment_ids, seg_idx, k=args.num_neighbors_in_segment)
        all_coords = torch.cat([coords, neighbor_coords.reshape(-1, 2)], dim=0)

        if use_repel:
            far_coords = sample_out_segment_neighbors(
                coords, segment_ids, seg_idx, k=args.num_neighbors_out_segment,
                max_retries=args.out_segment_max_retries)
            all_coords = torch.cat([all_coords, far_coords.reshape(-1, 2)], dim=0)

        pixel_features = index_pixel_features(all_pixel_features, all_coords)

        # Live encoder output, NOT the saved pred_f.pth -- see
        # train_mapping_loss.py's identical comment for why.
        vertex_features = encoder(vertices)
        soft_points = soft_map_to_vertices(
            pixel_features, vertex_features, vertices, args.k_vertices_per_pixel)

        near_end = num_centers + num_centers * args.num_neighbors_in_segment
        center_points = soft_points[:num_centers]
        # Both sampling functions always fill every slot (rejection-sampled
        # against collisions/self-matches, per their own docstrings), so
        # every pair here is valid -- no invalid-slot bookkeeping needed,
        # same as train_relational_loss.py's identical assumption.
        neighbor_valid = torch.ones(num_centers, args.num_neighbors_in_segment,
                                    dtype=torch.bool, device=device)
        map_loss = mapping_loss(
            center_points,
            soft_points[num_centers:near_end].view(num_centers, args.num_neighbors_in_segment, 3),
            neighbor_valid)

        if use_repel:
            far_points = soft_points[near_end:].view(num_centers, args.num_neighbors_out_segment, 3)
            far_valid = torch.ones(num_centers, args.num_neighbors_out_segment,
                                   dtype=torch.bool, device=device)
            repel_loss = repulsion_loss(center_points, far_points, far_valid, args.repel_margin)
            loss = args.lambda_map * map_loss + args.lambda_repel * repel_loss
        else:
            loss = args.lambda_map * map_loss

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
            if use_repel:
                print('step %4d/%d  loss %.6f (%.4g*map %.6f + %.4g*repel %.6f)  '
                      'drift(cos vs phase 1) %.4f'
                      % (step, args.num_iters, loss.item(), args.lambda_map, map_loss.item(),
                         args.lambda_repel, repel_loss.item(), drift))
            else:
                print('step %4d/%d  loss %.6f (%.4g*map %.6f)  drift(cos vs phase 1) %.4f'
                      % (step, args.num_iters, loss.item(), args.lambda_map, map_loss.item(), drift))

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
                   os.path.join(args.out_dir, 'mapping_repel_resampled_loss.png'))
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
                        help='random pixels drawn from a center\'s own fine segment for the '
                             'attractive mapping loss, resampled every step (replaces the '
                             'fixed 8-pixel adjacent ring)')
    parser.add_argument('--num_neighbors_out_segment', type=int, default=8,
                        help='random pixels drawn from outside a center\'s fine segment for '
                             'the repulsion loss, resampled every step (replaces the fixed '
                             'far ring)')
    parser.add_argument('--out_segment_max_retries', type=int, default=10,
                        help='vectorized rejection-sampling rounds for out-of-segment '
                             'draws that land back in the center\'s own segment')

    parser.add_argument('--lambda_repel', type=float, default=0.0,
                        help='weight of the repulsion term; 0 (default) disables it '
                             'entirely and skips far-neighbour sampling')
    parser.add_argument('--lambda_map', type=float, default=1.0,
                        help='weight of the (attractive) mapping loss')
    parser.add_argument('--repel_margin', type=float, default=0.05,
                        help='far pairs whose soft-mapped points are closer than this (mesh '
                             'coordinate scale, bbox diagonal ~1.89) are penalized -- sized '
                             'against the fixed-far-ring run\'s ~100px-away pairs, not '
                             'necessarily calibrated for this run\'s segment-exclusive but '
                             'possibly-nearby pairs; watch the results')

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
                        default='./experiments/guitar/encoder_dino_mapping_repel_resampled')
    args = parser.parse_args()

    if args.init_encoder_checkpoint is None:
        args.init_encoder_checkpoint = os.path.join(args.init_encoder_dir,
                                                    'encoder_checkpoint.pth')
    train(args)
