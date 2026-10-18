"""DFD-style (threedle/dfd, CVPR 2026) barycentric feature distillation for phase-1 training.

Every existing phase-1 script (encoder.py's train_encoder, train_encoder_dino_sd.py,
train_encoder_dino_sd_l2norm.py) queries the encoder only at the mesh's ~4036 fixed vertices,
paints that (changing, every step) prediction as the mesh's render color, and re-renders every
step to get a per-pixel comparable value. This script instead queries the encoder directly at
every foreground pixel's own exact barycentric-interpolated xyz location (from
modules/barycentric_targets.py's cache, itself rendered once since positions don't change) --
so there is no render call anywhere in this training loop at all.

New, additive, does not modify encoder.py, modules/encoder_l2norm.py, or any existing
train_encoder* script. Builds on EncoderL2Norm + cosine loss (this project's current-best-recipe
convention) -- confirmed against threedle/dfd's own distillation.py: they use mse_loss on
L2-normalized features, which is mathematically equivalent (up to a constant) to a cosine loss
when both sides are unit-norm, so no different loss is needed here.

Why not modules/dataset.py's EncoderDataset for the paired loader: EncoderDataset.prepare_path_list
builds its file list via a bare glob() with no defined ordering. Every EXISTING script gets away
with this because it re-derives the camera pose from each loaded file's own elev/azim and renders
fresh at that exact pose every step -- so whatever order the DataLoader yields files in, the
prediction is always rendered at the SAME pose as the target it's compared against. This script
skips that per-step render entirely, which means we lose that self-correcting mechanism: two
independently-glob'd EncoderDatasets (xyz_bary_f/ vs dino_sd_f/) are NOT guaranteed to yield view i's
xyz and view i's feature target at the same DataLoader step, and nothing would ever throw an error --
it would silently train the encoder to map wrong-view xyz points to wrong-view features. So this
file pairs the two caches by their shared explicit numeric view index (target_..._f_{i}.pt) instead,
and asserts each pair's elev/azim actually match as a cheap sanity check that both caches were
really built from the same random_viewing_angles.pt.

Run (assumes the existing dino_sd_f/ cache and this run's own xyz_bary_f/ cache both already exist
under --encoder_data_dir; generate the latter with --generate_xyz_targets 1 first if missing):
cd $DISC_ROOT && python train_encoder_dfd.py \
  --encoder_data_dir ./data/guitar/encoder_data_frontcentered_dinosd \
  --encoder_model_dir ./experiments/guitar/encoder_dino_sd_dfd_phase1 \
  --width 2304 --n_classes 2304 --generate_xyz_targets 1 --start_training 1
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from modules.barycentric_targets import BACKGROUND_SENTINEL, encode_random_views_xyz_bary
from modules.encoder_l2norm import EncoderL2Norm
from modules.utils import device, loadmesh


class PairedXYZFeatureDataset(Dataset):
    """Loads by explicit numeric view index (see module docstring for why EncoderDataset's
    unordered glob isn't safe here), asserting both caches' elev/azim match per index as a cheap
    guard that they were built from the same random_viewing_angles.pt."""

    def __init__(self, xyz_data_dir, feature_data_dir, feature_model_type, render_res, n_views):
        self.xyz_dir = os.path.join(xyz_data_dir, 'xyz_bary_f', str(render_res))
        self.feature_dir = os.path.join(feature_data_dir, f'{feature_model_type}_f', str(render_res))
        self.feature_model_type = feature_model_type
        self.n_views = n_views

    def __len__(self):
        return self.n_views

    def __getitem__(self, i):
        xyz = torch.load(os.path.join(self.xyz_dir, f'target_xyz_bary_f_{i}.pt'))
        feat = torch.load(os.path.join(self.feature_dir, f'target_{self.feature_model_type}_f_{i}.pt'))
        assert torch.allclose(xyz['elev'], feat['elev']) and torch.allclose(xyz['azim'], feat['azim']), (
            f"view {i}: xyz_bary_f/dino_sd_f cache elev/azim mismatch -- caches were not built "
            f"from the same random_viewing_angles.pt")
        xyz_f = xyz['model_f']  # [3, H, W]
        feat_f = feat['model_f'].float()  # cache may be float16 (--cache_dtype); always train in float32
        if feat_f.dim() == 4:
            # dino_sd_f's cached model_f keeps postprocess_features' own leading dim of 1
            # ([1, C, H, W], see train_encoder_dino_sd.py:87) -- unlike our xyz cache, which is
            # saved pre-squeezed. Normalize here so both sides are [C, H, W] before the DataLoader
            # adds its own batch dim.
            feat_f = feat_f.squeeze(0)
        return xyz_f, feat_f


def train_encoder_dfd(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # EncoderL2Norm must be the first RNG-consuming call after seeding -- load-bearing for phase-2
    # compatibility (FourierFeatureTransform's random basis _B is a plain attribute, not in
    # state_dict(); reconstructing it identically depends on this exact seed-then-construct order,
    # see phase2/train_mapping_loss.py's load_phase1_encoder docstring).
    encoder = EncoderL2Norm(args.depth, args.width, out_dim=args.n_classes,
                             positional_encoding=args.positional_encoding,
                             sigma=args.sigma, network_verbose=args.network_verbose).to(device)
    optim = torch.optim.Adam(encoder.parameters(), args.learning_rate)

    dataset = PairedXYZFeatureDataset(args.encoder_data_dir, args.encoder_data_dir,
                                       args.feature_model_type, args.render_res, args.n_views)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    os.makedirs(args.encoder_model_dir, exist_ok=True)

    losses = []
    for _ in range(args.num_epochs):
        for i, (xyz_batch, feat_batch) in enumerate(tqdm(dataloader)):
            optim.zero_grad()
            xyz_batch = xyz_batch.squeeze(0).to(device)   # [3, 64, 64], channel-first
            feat_batch = feat_batch.squeeze(0).to(device)  # [C, 64, 64], channel-first

            spatial_mask = (xyz_batch != BACKGROUND_SENTINEL).any(dim=0)  # [64, 64]
            n_fg = spatial_mask.sum().clamp(min=1)

            xyz_pts = xyz_batch.permute(1, 2, 0)[spatial_mask]    # [n_fg, 3]
            target_pts = feat_batch.permute(1, 2, 0)[spatial_mask]  # [n_fg, C]

            pred = encoder(xyz_pts)  # [n_fg, C] -- no render call anywhere in this loop
            cos_sim = F.cosine_similarity(pred, target_pts, dim=-1)
            loss = (1.0 - cos_sim).sum() / n_fg
            loss.backward()
            optim.step()

            with torch.no_grad():
                losses.append(loss.item())

            if i % 20 == 0:
                print("Last 20 cosine loss: {}".format(np.mean(losses[-20:])))

    torch.save({'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'losses': losses
                }, os.path.join(args.encoder_model_dir, 'encoder_checkpoint.pth'))

    # Encoder is domain-agnostic (any [N,3] xyz in, no notion of "is a mesh vertex" -- see
    # modules/encoder_l2norm.py's forward), so this final full-mesh dump is still needed even
    # though training itself never queried the discrete vertex set: every downstream phase-2/
    # analysis script expects exactly this pred_f.pth in [num_vertices, channels] shape.
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
    parser.add_argument('--radius', type=float, default=2.0)

    parser.add_argument('--n_views', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--feature_model_type', type=str, default='dino_sd')

    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--n_classes', type=int, required=True)
    parser.add_argument('--positional_encoding', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=5.0)
    parser.add_argument('--network_verbose', type=int, default=1)

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=3)

    parser.add_argument('--generate_xyz_targets', type=int, default=0)
    parser.add_argument('--start_training', type=int, default=0)

    args = parser.parse_args()
    args.mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)

    if args.generate_xyz_targets == 1:
        encode_random_views_xyz_bary(args)

    if args.start_training == 1:
        train_encoder_dfd(args)
