"""Export a pred_f.pth (per-vertex features) from any encoder checkpoint.

Phase-2 training writes a checkpoint every --checkpoint_every steps, and the
right one to keep is chosen after the fact from the loss/drift trade-off rather
than fixed in advance. The existing analysis pipeline
(pixel_to_vertex_nn.py -> color_mesh_from_knn.py -> render_colored_mesh.py)
consumes a pred_f.pth tensor via --vertex_features_path, so this turns any
checkpoint into something that pipeline can read.

Fourier basis handling, which is the whole reason this needs care: `_B`
(modules/utils.py:272) is a plain attribute, not a registered buffer, so it is
absent from state_dict(). Checkpoints written by train_mapping_loss.py carry it
explicitly under 'fourier_basis'. Older ones (phase 1's) do not, and for those
we fall back to reproducing phase 1's RNG sequence -- verified to work, see
analysis/verify_encoder_resume.py.

Run:
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python \
  analysis/export_pred_f.py \
  --checkpoint ./experiments/guitar/encoder_dino_mapping/encoder_checkpoint_step150.pth \
  --output ./experiments/guitar/encoder_dino_mapping/pred_f_step150.pth

BATCH MODE (--checkpoints/--outputs, 2026-08-12): a run's own step_progression_grid
analysis calls this script once per checkpoint (typically 12), and every one of
those calls repeats the same mesh load (loadmesh(..., load_rings=True), which builds
a ring dictionary in a Python loop over every vertex) and Encoder(...) construction --
identical work every time, since only the checkpoint (its state_dict + stored Fourier
basis) actually differs between calls. Measured at ~25s/checkpoint, ~5 minutes total
for 12 checkpoints in single-file mode. Batch mode does the mesh load and Encoder
construction ONCE, then loops only over load_state_dict + the forward pass -- mirrors
pixel_to_vertex_nn.py's own --vertex_features_paths/--output_paths batch mode, same
reasoning. Single-checkpoint mode below is byte-for-byte unchanged; batch mode is
additive, verified to produce identical pred_f tensors (torch.equal) to running this
script once per checkpoint.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.verify_encoder_resume import seed_everything
from modules.encoder import Encoder
from modules.utils import device, load_state_dict, loadmesh


def export_one(encoder, mesh, checkpoint_path, output_path):
    """Load one checkpoint's weights + Fourier basis onto an already-constructed
    `encoder`/`mesh` (both checkpoint-independent), forward, save. Factored out so
    single-checkpoint mode and batch mode share identical logic -- single mode is
    just a batch of size 1, not a separately maintained copy."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_state_dict(encoder, checkpoint['encoder_state_dict'])

    basis = checkpoint.get('fourier_basis')
    if basis is not None:
        encoder.mlp[0]._B = basis
        print('restored Fourier basis from the checkpoint')
    else:
        print('checkpoint has no stored Fourier basis; relying on seed to '
              'reproduce it (valid for phase-1 checkpoints)')

    with torch.no_grad():
        pred_f = encoder(mesh.vertices)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(pred_f, output_path)

    step = checkpoint.get('step')
    drift = checkpoint.get('drift')
    print('exported %s from %s%s%s'
          % (tuple(pred_f.shape), os.path.basename(checkpoint_path),
             '' if step is None else ' (step %d' % step,
             '' if step is None else (')' if drift is None else ', drift %.4f)' % drift)))
    print('saved %s' % output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='required unless using batch mode (--checkpoints/--outputs)')
    parser.add_argument('--output', type=str, default=None,
                        help='required unless using batch mode (--checkpoints/--outputs)')
    parser.add_argument('--checkpoints', type=str, default=None,
                        help='BATCH MODE: comma-separated checkpoint paths (e.g. every step of one '
                             'training run) to export, sharing ONE mesh load + Encoder construction '
                             'instead of repeating it per checkpoint. All must share the same --seed/'
                             '--depth/--width/--n_classes/--positional_encoding/--sigma/--obj_path -- '
                             'true for every checkpoint of one training run. Must be paired 1:1 with '
                             '--outputs. Mutually exclusive with --checkpoint/--output.')
    parser.add_argument('--outputs', type=str, default=None,
                        help='BATCH MODE: comma-separated output paths, 1:1 with --checkpoints.')
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')
    # Must match the training run that produced the checkpoint.
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--n_classes', type=int, default=1024)
    parser.add_argument('--positional_encoding', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=5.0)
    args = parser.parse_args()

    if args.checkpoints or args.outputs:
        if not (args.checkpoints and args.outputs):
            parser.error('--checkpoints and --outputs must both be given together')
        if args.checkpoint or args.output:
            parser.error('--checkpoints/--outputs (batch mode) is mutually exclusive with '
                         '--checkpoint/--output')
        checkpoint_paths = args.checkpoints.split(',')
        output_paths = args.outputs.split(',')
        if len(checkpoint_paths) != len(output_paths):
            parser.error(f'--checkpoints has {len(checkpoint_paths)} entries but --outputs has '
                         f'{len(output_paths)} -- must match 1:1')
    else:
        if not (args.checkpoint and args.output):
            parser.error('--checkpoint and --output are required unless using batch mode '
                         '(--checkpoints/--outputs)')
        checkpoint_paths = [args.checkpoint]
        output_paths = [args.output]

    mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)

    seed_everything(args.seed)
    encoder = Encoder(args.depth, args.width, out_dim=args.n_classes,
                      positional_encoding=args.positional_encoding,
                      sigma=args.sigma, network_verbose=0).to(device)

    for i, (checkpoint_path, output_path) in enumerate(zip(checkpoint_paths, output_paths)):
        if len(checkpoint_paths) > 1:
            print(f'=== batch {i + 1}/{len(checkpoint_paths)}: {checkpoint_path} -> {output_path} ===')
        export_one(encoder, mesh, checkpoint_path, output_path)
