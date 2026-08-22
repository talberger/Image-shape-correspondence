"""Cross-view point batching variant of train_encoder_dfd.py, mirroring threedle/dfd's own
distillation.py training loop (confirmed by reading it directly, PROJECT_NOTES.md 2026-08-22 entry).

train_encoder_dfd.py batches by VIEW -- each DataLoader item is one view's own foreground pixels,
processed as one step, sequentially by view index. DFD's real repo instead flattens EVERY view's
(xyz, feature) pairs into one big pool up front, then each epoch draws a random permutation over
the WHOLE pool and chunks it into fixed-size minibatches -- so a single optimizer step can (and
usually does) mix points from several different views. Only possible because, like
train_encoder_dfd.py, there is no render call anywhere in this training loop -- once training data
is just flat (xyz, feature) tuples with no remaining notion of "which view", nothing requires
points from the same view to stay together.

Per the DFD paper's own ablation (Fig. 13, Appendix A), the smoothness win over vertex-only
distillation comes from dense per-pixel (not per-vertex) supervision, not from how those pixels get
batched -- so this script is a training-speed experiment, not expected to change coverage/smoothness
outcomes versus train_encoder_dfd.py. That's the comparison this script exists to let Tal make.

New, additive, does not modify train_encoder_dfd.py or anything it imports.

Run (same cache prerequisites as train_encoder_dfd.py -- xyz_bary_f/ and dino_sd_f/ must already
exist under --encoder_data_dir):
cd /workspace/bsb && python train_encoder_dfd_crossview.py \
  --encoder_data_dir ./data/guitar/encoder_data_frontcentered_dinosd \
  --encoder_model_dir ./experiments/guitar/encoder_dino_sd_dfd_crossview_phase1 \
  --width 2304 --n_classes 2304 --batch_size 4096
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.barycentric_targets import BACKGROUND_SENTINEL
from modules.encoder_l2norm import EncoderL2Norm
from modules.utils import device, loadmesh


def load_all_points(args):
    """Loads every view's paired (xyz, feature) cache and flattens foreground points across all
    views into one pool -- the up-front, once-only construction distillation.py itself does for
    its own `xyz`/`view_features` tensors. Same explicit-index pairing + elev/azim sanity check as
    PairedXYZFeatureDataset (train_encoder_dfd.py), since EncoderDataset's unordered glob is still
    not safe here for the same reason explained in that file's docstring."""
    xyz_dir = os.path.join(args.encoder_data_dir, 'xyz_bary_f', str(args.render_res))
    feat_dir = os.path.join(args.encoder_data_dir, f'{args.feature_model_type}_f', str(args.render_res))

    xyz_chunks, feat_chunks = [], []
    for i in tqdm(range(args.n_views), desc='loading+flattening all views'):
        xyz = torch.load(os.path.join(xyz_dir, f'target_xyz_bary_f_{i}.pt'), map_location='cpu')
        feat = torch.load(os.path.join(feat_dir, f'target_{args.feature_model_type}_f_{i}.pt'), map_location='cpu')
        assert torch.allclose(xyz['elev'], feat['elev']) and torch.allclose(xyz['azim'], feat['azim']), (
            f"view {i}: xyz_bary_f/{args.feature_model_type}_f cache elev/azim mismatch")

        xyz_f = xyz['model_f']  # [3, H, W]
        feat_f = feat['model_f']
        if feat_f.dim() == 4:
            feat_f = feat_f.squeeze(0)  # dino_sd_f keeps postprocess_features' leading dim of 1

        spatial_mask = (xyz_f != BACKGROUND_SENTINEL).any(dim=0)  # [H, W]
        xyz_chunks.append(xyz_f.permute(1, 2, 0)[spatial_mask])   # [n_fg_i, 3]
        feat_chunks.append(feat_f.permute(1, 2, 0)[spatial_mask])  # [n_fg_i, C]

    all_xyz = torch.cat(xyz_chunks, dim=0).contiguous()
    all_feat = torch.cat(feat_chunks, dim=0).contiguous()
    return all_xyz, all_feat


def train_encoder_dfd_crossview(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Same seed-then-construct load-bearing ordering as train_encoder_dfd.py -- see that file's
    # comment for why (phase-2's FourierFeatureTransform resume-gate depends on it).
    encoder = EncoderL2Norm(args.depth, args.width, out_dim=args.n_classes,
                             positional_encoding=args.positional_encoding,
                             sigma=args.sigma, network_verbose=args.network_verbose).to(device)
    optim = torch.optim.Adam(encoder.parameters(), args.learning_rate)

    all_xyz, all_feat = load_all_points(args)
    n_total_points = all_xyz.shape[0]
    print(f'Flattened {n_total_points} foreground points across {args.n_views} views '
          f'({n_total_points / args.n_views:.1f} avg points/view)')

    os.makedirs(args.encoder_model_dir, exist_ok=True)

    losses = []
    train_start = time.time()
    for epoch in range(args.num_epochs):
        perm = torch.randperm(n_total_points)  # cross-view: mixes points from different views
        epoch_xyz = all_xyz[perm]
        epoch_feat = all_feat[perm]

        for start in tqdm(range(0, n_total_points, args.batch_size), desc=f'epoch {epoch}'):
            end = min(start + args.batch_size, n_total_points)
            xyz_batch = epoch_xyz[start:end].to(device, non_blocking=True)
            feat_batch = epoch_feat[start:end].to(device, non_blocking=True)

            optim.zero_grad()
            pred = encoder(xyz_batch)
            cos_sim = F.cosine_similarity(pred, feat_batch, dim=-1)
            loss = (1.0 - cos_sim).mean()
            loss.backward()
            optim.step()

            with torch.no_grad():
                losses.append(loss.item())

        print(f'epoch {epoch} mean loss: {np.mean(losses[-(n_total_points // args.batch_size + 1):]):.4f}')

    train_wall_clock_sec = time.time() - train_start
    print(f'Training wall-clock: {train_wall_clock_sec:.1f}s ({train_wall_clock_sec / 60:.2f} min)')

    torch.save({'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'losses': losses,
                'train_wall_clock_sec': train_wall_clock_sec,
                }, os.path.join(args.encoder_model_dir, 'encoder_checkpoint.pth'))

    with torch.no_grad():
        pred_f = encoder(args.mesh.vertices)
    torch.save(pred_f, os.path.join(args.encoder_model_dir, 'pred_f.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')

    parser.add_argument('--encoder_data_dir', type=str, required=True)
    parser.add_argument('--encoder_model_dir', type=str, required=True)

    parser.add_argument('--render_res', type=int, default=224)

    parser.add_argument('--n_views', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4096)

    parser.add_argument('--feature_model_type', type=str, default='dino_sd')

    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--n_classes', type=int, required=True)
    parser.add_argument('--positional_encoding', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=5.0)
    parser.add_argument('--network_verbose', type=int, default=1)

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=3)

    args = parser.parse_args()
    args.mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)

    train_encoder_dfd_crossview(args)
