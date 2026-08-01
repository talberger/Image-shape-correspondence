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
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.verify_encoder_resume import seed_everything
from modules.encoder import Encoder
from modules.utils import device, load_state_dict, loadmesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
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

    mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)

    seed_everything(args.seed)
    encoder = Encoder(args.depth, args.width, out_dim=args.n_classes,
                      positional_encoding=args.positional_encoding,
                      sigma=args.sigma, network_verbose=0).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    load_state_dict(encoder, checkpoint['encoder_state_dict'])

    basis = checkpoint.get('fourier_basis')
    if basis is not None:
        encoder.mlp[0]._B = basis
        print('restored Fourier basis from the checkpoint')
    else:
        print('checkpoint has no stored Fourier basis; relying on seed=%d to '
              'reproduce it (valid for phase-1 checkpoints)' % args.seed)

    with torch.no_grad():
        pred_f = encoder(mesh.vertices)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(pred_f, args.output)

    step = checkpoint.get('step')
    drift = checkpoint.get('drift')
    print('exported %s from %s%s%s'
          % (tuple(pred_f.shape), os.path.basename(args.checkpoint),
             '' if step is None else ' (step %d' % step,
             '' if step is None else (')' if drift is None else ', drift %.4f)' % drift)))
    print('saved %s' % args.output)
