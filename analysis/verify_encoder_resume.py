"""Verify that reloading a phase-1 encoder checkpoint reproduces its pred_f.pth.

Why this exists
---------------
FourierFeatureTransform (modules/utils.py:272) assigns its random Gaussian
projection matrix as `self._B = torch.stack(B_sort)` -- a plain attribute, NOT
`register_buffer`. So `_B` is absent from `state_dict()` and therefore absent
from the saved `encoder_checkpoint.pth`. Rebuilding an `Encoder` and loading the
checkpoint into it re-randomizes the Fourier basis: the model loads without any
error, produces plausible-looking features, and is silently wrong.

Nothing in the upstream repo was ever hurt by this, because nothing reloads the
encoder's weights -- every consumer reads `pred_f.pth` (the feature tensor), and
the one path that rebuilds the Encoder (encoder.py:68-72) is dead code behind a
hardcoded `load_features = True`. Phase-2 mapping-loss training is the first code
to reload the weights, so it has to check.

`pred_f.pth` is the ground truth for that check: it was written by the very
process that still held the correct `_B` in memory (encoder.py:363-364).

Measured result (2026-08-01, guitar / encoder_dino): seeding with 0 and then
constructing the Encoder reproduces pred_f.pth to float32 noise (mean cosine
1.000000, max abs diff 2.1e-05), while a deliberately wrong seed gives mean
cosine 0.673 and max abs diff 2.0. The gate below separates those cleanly.

Run:
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python \
  analysis/verify_encoder_resume.py
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.encoder import Encoder
from modules.utils import device, load_state_dict, loadmesh

# A wrong Fourier basis shows up as cosine ~0.67 / max diff ~2.0, so these
# thresholds pass a correct reload by five orders of magnitude and still fail a
# wrong one by a wide margin.
MIN_MEAN_COSINE = 0.9999
MAX_ABS_DIFF = 1e-3


def seed_everything(seed):
    """Reproduce encoder.py:290-296 exactly, so `_B` comes out the same."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_and_load_encoder(checkpoint_path, args):
    """Rebuild the phase-1 Encoder and load its weights.

    The seeding is load-bearing, not hygiene: it is what restores `_B`. Encoder
    construction must stay the first RNG consumer after the seed, matching
    encoder.py:298-300 (verified there that the Renderer built in between
    consumes no RNG, so its absence here is safe).
    """
    seed_everything(args.seed)
    encoder = Encoder(args.depth, args.width, out_dim=args.n_classes,
                      positional_encoding=args.positional_encoding,
                      sigma=args.sigma, network_verbose=0).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_state_dict(encoder, checkpoint['encoder_state_dict'])
    return encoder


def check_reproduces_pred_f(encoder, vertices, pred_f_path, verbose=True):
    """Return (ok, mean_cosine, max_abs_diff) for encoder(vertices) vs pred_f."""
    target = torch.load(pred_f_path, map_location=device)
    with torch.no_grad():
        pred = encoder(vertices)

    if pred.shape != target.shape:
        raise ValueError('shape mismatch: encoder gives %s, pred_f.pth is %s'
                         % (tuple(pred.shape), tuple(target.shape)))

    mean_cosine = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
    max_abs_diff = (pred - target).abs().max().item()
    ok = mean_cosine >= MIN_MEAN_COSINE and max_abs_diff <= MAX_ABS_DIFF

    if verbose:
        print('  mean cosine vs pred_f.pth : %.6f   (need >= %.4f)'
              % (mean_cosine, MIN_MEAN_COSINE))
        print('  max abs diff              : %.3e   (need <= %.0e)'
              % (max_abs_diff, MAX_ABS_DIFF))
        print('  -> %s' % ('PASS: Fourier basis restored correctly'
                           if ok else 'FAIL: _B differs -- do NOT train on this'))
    return ok, mean_cosine, max_abs_diff


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')
    parser.add_argument('--encoder_model_dir', type=str,
                        default='./experiments/guitar/encoder_dino')
    # Must match the phase-1 training run (README step 2: dino, width/n_classes
    # 1024); everything else was left at encoder.py's defaults.
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--n_classes', type=int, default=1024)
    parser.add_argument('--positional_encoding', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=5.0)
    parser.add_argument('--control', action='store_true',
                        help='also run a wrong-seed control, which MUST fail. Use this to '
                             'confirm the check is actually sensitive rather than passing '
                             'trivially.')
    args = parser.parse_args()

    mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)
    checkpoint_path = os.path.join(args.encoder_model_dir, 'encoder_checkpoint.pth')
    pred_f_path = os.path.join(args.encoder_model_dir, 'pred_f.pth')

    print('device=%s  vertices=%s' % (device, tuple(mesh.vertices.shape)))
    print('\n[resume check] seed=%d' % args.seed)
    encoder = build_and_load_encoder(checkpoint_path, args)
    ok, _, _ = check_reproduces_pred_f(encoder, mesh.vertices, pred_f_path)

    if args.control:
        print('\n[control] seed=%d (wrong _B on purpose -- must FAIL)' % (args.seed + 1))
        control_args = argparse.Namespace(**vars(args))
        control_args.seed = args.seed + 1
        control_encoder = build_and_load_encoder(checkpoint_path, control_args)
        control_ok, _, _ = check_reproduces_pred_f(control_encoder, mesh.vertices, pred_f_path)
        if control_ok:
            print('\nCONTROL DID NOT FAIL -- the check is insensitive and proves nothing.')
            sys.exit(2)
        print('  (control failed as expected, so the check above is meaningful)')

    sys.exit(0 if ok else 1)
