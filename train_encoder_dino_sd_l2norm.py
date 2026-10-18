"""
Experimental phase-1 variant: L2-normalized encoder output + cosine loss, instead of the baseline's
Tanh output + masked-MSE loss. New, additive script -- does not modify encoder.py,
modules/encoder.py, or train_encoder_dino_sd.py. Reuses encode_random_views_dino_sd from
train_encoder_dino_sd.py directly (the DINO+SD extraction/fusion step is unaffected by this
change -- only the vertex-encoder's architecture and loss differ), so the already-cached
data/guitar/encoder_data_frontcentered_dinosd/dino_sd_f/ does not need to be regenerated.

Why: see modules/encoder_l2norm.py's docstring. In one line -- both Diff3F (arXiv 2311.17024) and
Surface-Aware Distilled 3D Semantic Features (arXiv 2503.18254), the two papers this project's
DINO+SD fusion recipe is modeled on, make their final per-point 3D feature unit-norm by
construction; our Tanh-bounded encoder does not, and phase-1 SD-DINO's vertex features are far
less spatially smooth than plain DINO's (distance-vs-cosine-similarity corr ~-0.27 vs ~-0.85,
PROJECT_NOTES.md 2026-08-19 entry). This is an isolated experiment to test whether that gap is
explained by the magnitude/scale problem: does removing it (L2-normalize output + cosine loss,
matching both papers' own reconstruction-loss choice) restore smoothness closer to plain DINO's?

Writes to its own experiments/guitar/encoder_dino_sd_frontcentered_l2norm_phase1/ -- the existing
encoder_dino_sd_frontcentered/ (Tanh+MSE) is untouched, so this can be discarded without affecting
anything already relied on if the experiment doesn't pan out.

Run (reuses the already-encoded data, so only --start_training is needed):
cd $DISC_ROOT && python train_encoder_dino_sd_l2norm.py \
  --encoder_data_dir ./data/guitar/encoder_data_frontcentered_dinosd \
  --encoder_model_dir ./experiments/guitar/encoder_dino_sd_frontcentered_l2norm_phase1 \
  --width 2304 --n_classes 2304 --start_training 1
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from encoder import generate_random_views, preprocess_image
from modules.dataset import EncoderDataset
from modules.dino import DINOWrapper
from modules.dino_sd_fusion import SDFeatureExtractor
from modules.encoder_l2norm import EncoderL2Norm
from modules.render import Renderer
from modules.utils import device, loadmesh, setcolor_mesh
from train_encoder_dino_sd import encode_random_views_dino_sd


def train_encoder_l2norm(args):
    """Mirrors encoder.py's train_encoder exactly (same seeding, same EncoderDataset/Renderer/
    setcolor_mesh, same per-step structure), except: (a) EncoderL2Norm instead of Encoder, (b) the
    loss is per-pixel cosine distance (1 - cosine_similarity) instead of masked MSE. Mask handling
    differs accordingly: the baseline's per-ELEMENT mask (rendered_prob_views != 1.0) doesn't make
    sense for a per-PIXEL cosine similarity, so it's reduced to a per-pixel mask via .any(dim=-1)
    (background pixels are exactly 1.0 in every channel; any real foreground pixel will differ in
    at least one channel).
    """
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    render = Renderer(dim=(64, 64), radius=args.radius)
    vertices = args.mesh.vertices
    encoder = EncoderL2Norm(args.depth, args.width, out_dim=args.n_classes,
                            positional_encoding=args.positional_encoding,
                            sigma=args.sigma, network_verbose=args.network_verbose).to(device)
    optim = torch.optim.Adam(encoder.parameters(), args.learning_rate)

    losses = []
    torch.cuda.empty_cache()

    sampled_mesh = args.mesh
    dataset = EncoderDataset(args.encoder_data_dir, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    if not os.path.exists(args.encoder_model_dir):
        os.makedirs(args.encoder_model_dir, exist_ok=True)

    for _ in range(args.num_epochs):
        for i, (batch_sam_f, batch_elevs, batch_azims) in enumerate(tqdm(dataloader)):
            optim.zero_grad()
            pred_f = encoder(vertices)

            setcolor_mesh(sampled_mesh, pred_f)
            rendered_prob_views, elev, azim, mask = render.render_views(sampled_mesh, num_views=1,
                                                                                show=False,
                                                                                std=args.frontview_std,
                                                                                return_views=True,
                                                                                center_azim=batch_azims,
                                                                                center_elev=batch_elevs,
                                                                                return_features=True,
                                                                                lighting=False,
                                                                                background=torch.ones(args.n_classes).to(device),
                                                                                return_mask=True)
            batch_sam_f = batch_sam_f.squeeze(0)
            # --- bug fix 2026-08-21 start ---
            # rendered_prob_views/batch_sam_f are channel-FIRST [1, n_classes, 64, 64]
            # (render_views' final .permute(0,3,1,2), postprocess_features' BxCxHxW), not
            # channel-last as this line originally assumed. dim=-1 was therefore reducing
            # over WIDTH (64), not the n_classes=2304 feature axis -- confirmed empirically
            # (real cached data): "foreground" count under the old dim=-1 mask was 110592,
            # more than the total 4096 pixels in a 64x64 frame, only possible by flattening
            # across channels too; the correct per-pixel count for the same view is 170.
            # This affected only this cosine-loss phase-1 script -- the Tanh+MSE loss in
            # encoder.py uses elementwise masking with no dim-reduction, unaffected; phase-2
            # (train_relational_loss.py) operates on already-flat [N,F] tensors, unaffected.
            elementwise_mask = (rendered_prob_views != 1.0)
            spatial_mask = elementwise_mask.any(dim=1)
            n_fg = spatial_mask.sum().clamp(min=1)

            cos_sim = F.cosine_similarity(batch_sam_f, rendered_prob_views, dim=1)
            loss = (1.0 - cos_sim)[spatial_mask].sum() / n_fg
            # --- bug fix 2026-08-21 end ---
            loss.backward(retain_graph=True)
            optim.step()

            with torch.no_grad():
                losses.append(loss.item())

            if i % 20 == 0:
                print("Last 20 cosine loss: {}".format(np.mean(losses[-20:])))
                current_lr = optim.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr}")

    torch.save({'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'losses': losses
                }, os.path.join(args.encoder_model_dir, 'encoder_checkpoint.pth'))

    pred_f = encoder(args.mesh.vertices)
    torch.save(pred_f, os.path.join(args.encoder_model_dir, 'pred_f.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')

    parser.add_argument('--encoder_data_dir', type=str, required=True)
    parser.add_argument('--encoder_model_dir', type=str, required=True)

    parser.add_argument('--target_image_size', type=int, default=224)

    parser.add_argument('--background', nargs=3, type=float, default=[1., 1., 1.])
    parser.add_argument('--frontview_std', type=float, default=4)
    parser.add_argument('--render_res', type=int, default=224)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.])
    parser.add_argument('--mesh_color', nargs=3, type=float, default=[2. / 3., 2. / 3., 2. / 3.])

    parser.add_argument('--n_views', type=int, default=1000)
    parser.add_argument('--view_sampling', type=str, default='random', choices=['random', 'fibonacci'])
    parser.add_argument('--data_percentage', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, required=True,
                        help='must match the fused feature dim (2304 = 1280 SD up_blocks[1] + 1024 DINO)')
    parser.add_argument('--n_classes', type=int, required=True, help='same as --width')
    parser.add_argument('--positional_encoding', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=5.0)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--network_verbose', type=int, default=1)

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=3)

    parser.add_argument('--generate_random_views', type=int, default=0)
    parser.add_argument('--encode_random_views', type=int, default=0)
    parser.add_argument('--start_training', type=int, default=0)

    args = parser.parse_args()
    args.model_type = 'dino_sd'

    args.mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)

    if args.generate_random_views == 1:
        generate_random_views(args)

    if args.encode_random_views == 1:
        dino_model = DINOWrapper(device=device, small=True, target_size=args.target_image_size).to(device)
        print('loading SD components (first run downloads several GB, cached under $HF_HOME after)...')
        sd_extractor = SDFeatureExtractor(device=device)
        encode_random_views_dino_sd(args, dino_model, sd_extractor)
        del dino_model, sd_extractor
        torch.cuda.empty_cache()

    if args.start_training == 1:
        train_encoder_l2norm(args)
