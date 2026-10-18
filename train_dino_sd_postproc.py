"""
Trains modules/feature_postproc.py's FeaturePostProcessor on top of the frozen, cached
dino_sd_f feature maps, using free correspondence derived from the mesh's own known 3D
geometry (modules/xyz_correspondence.py) -- no manual labels. Fallback plan for
left/right disambiguation (see PROJECT_NOTES.md), used only if the GeoAware-SC
pretrained-network experiment doesn't transfer to the guitar.

Two pair types, both built from modules/xyz_correspondence.py:
- cross-view pairs: query pixel in view A, true target = the real corresponding pixel in
  a different view B. Teaches general correspondence sharpness.
- mirror pairs (the actual disambiguation mechanism): query pixel and its mirror twin,
  both visible in the SAME view. True target = the query's OWN location -- the mirror
  twin sits in the same softmax denominator as a hard negative, so the loss directly
  penalizes the post-processor whenever it can't tell a point apart from its mirror
  image.

Dense soft-argmax loss (adapted from "Telling Left from Right", Zhang et al. CVPR 2024,
Sec. 4.2): similarity map between the query feature and the target view's full feature
map -> softmax -> expected (soft-argmax) location -> L2 against the true location.

New, additive script -- doesn't touch train_encoder_dino_sd.py, train_encoder_dfd_crossview.py,
or any other training script. Trains a small module on top of an existing frozen cache;
the cache itself is never modified.

Run (small-scale sanity check first, per this project's own build-discipline):
cd $DISC_ROOT && python train_dino_sd_postproc.py \
  --encoder_data_dir ./data/guitar/encoder_data_frontcentered_dinosd \
  --out_dir ./experiments/guitar/dino_sd_postproc \
  --n_views 50 --num_steps 500

Full run:
cd $DISC_ROOT && python train_dino_sd_postproc.py \
  --encoder_data_dir ./data/guitar/encoder_data_frontcentered_dinosd \
  --out_dir ./experiments/guitar/dino_sd_postproc \
  --n_views 1000 --num_steps 20000
"""
import argparse
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.barycentric_targets import BACKGROUND_SENTINEL
from modules.feature_postproc import FeaturePostProcessor, save_checkpoint
from modules.utils import device
from modules.xyz_correspondence import build_cross_view_pairs, build_mirror_pairs, load_all_fg_xyz


class ViewCache:
    """Small LRU cache of (feature_map, foreground_mask) per view -- avoids holding all
    1000 views' dino_sd_f tensors (36GB) in RAM at once, since any one training step only
    touches a handful of views."""

    def __init__(self, encoder_data_dir, render_res, feature_model_type, max_size=200):
        self.feat_dir = os.path.join(encoder_data_dir, f'{feature_model_type}_f', str(render_res))
        self.xyz_dir = os.path.join(encoder_data_dir, 'xyz_bary_f', str(render_res))
        self.feature_model_type = feature_model_type
        self.max_size = max_size
        self._cache = OrderedDict()

    def get(self, view_id):
        if view_id in self._cache:
            self._cache.move_to_end(view_id)
            return self._cache[view_id]

        feat = torch.load(os.path.join(
            self.feat_dir, f'target_{self.feature_model_type}_f_{view_id}.pt'),
            map_location='cpu')['model_f'].float()  # cache may be float16 (--cache_dtype)
        if feat.dim() == 4:
            feat = feat.squeeze(0)  # [C, H, W]
        xyz = torch.load(os.path.join(
            self.xyz_dir, f'target_xyz_bary_f_{view_id}.pt'), map_location='cpu')['model_f']  # [3, H, W]
        mask = (xyz != BACKGROUND_SENTINEL).any(dim=0)  # [H, W] bool

        entry = (feat.to(device), mask.to(device))
        self._cache[view_id] = entry
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return entry


def soft_argmax_loss(query_feat, target_map, target_mask, true_row, true_col, temperature):
    """query_feat: [C]. target_map: [C, H, W]. target_mask: [H, W] bool.
    true_row/true_col: ints. Returns a scalar loss."""
    C, H, W = target_map.shape
    sim = F.cosine_similarity(query_feat.view(C, 1, 1), target_map, dim=0)  # [H, W]
    sim = sim.masked_fill(~target_mask, float('-inf'))
    probs = F.softmax(sim.view(-1) / temperature, dim=0).view(H, W)

    rows = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    cols = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
    row_hat = (probs * rows).sum()
    col_hat = (probs * cols).sum()

    loss = ((row_hat - true_row) / H) ** 2 + ((col_hat - true_col) / W) ** 2
    return loss


def build_or_load_pairs(args):
    if args.pairs_cache_path and os.path.exists(args.pairs_cache_path):
        print(f'loading cached pairs from {args.pairs_cache_path}')
        d = torch.load(args.pairs_cache_path)
        return d['xyz'], d['view_id'], d['row'], d['col'], d['cv_a'], d['cv_b'], d['mir_a'], d['mir_b']

    print('loading all foreground xyz...')
    xyz, view_id, row, col = load_all_fg_xyz(args.encoder_data_dir, args.render_res, args.n_views)
    print(f'{len(xyz)} foreground points across {args.n_views} views')

    print('building cross-view pairs...')
    cv_a, cv_b = build_cross_view_pairs(xyz, view_id, max_dist=args.cross_view_max_dist,
                                         n_pairs=args.n_cross_view_pairs, seed=args.seed)
    print(f'{len(cv_a)} cross-view pairs')

    print('building mirror pairs...')
    mir_a, mir_b = build_mirror_pairs(xyz, view_id, mirror_axis=args.mirror_axis,
                                       max_dist=args.mirror_max_dist,
                                       min_axis_offset=args.mirror_min_axis_offset,
                                       same_view_only=bool(args.mirror_same_view_only),
                                       n_pairs=args.n_mirror_pairs, seed=args.seed)
    print(f'{len(mir_a)} mirror pairs')

    if args.pairs_cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.pairs_cache_path)), exist_ok=True)
        torch.save({'xyz': xyz, 'view_id': view_id, 'row': row, 'col': col,
                    'cv_a': cv_a, 'cv_b': cv_b, 'mir_a': mir_a, 'mir_b': mir_b},
                   args.pairs_cache_path)

    return xyz, view_id, row, col, cv_a, cv_b, mir_a, mir_b


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _, view_id, row, col, cv_a, cv_b, mir_a, mir_b = build_or_load_pairs(args)

    postproc = FeaturePostProcessor(in_dim=args.width, hidden_dim=args.hidden_dim,
                                     l2_normalize=bool(args.l2_normalize)).to(device)
    optimizer = torch.optim.Adam(postproc.parameters(), lr=args.learning_rate)
    view_cache = ViewCache(args.encoder_data_dir, args.render_res, args.feature_model_type,
                            max_size=args.lru_cache_size)

    def get_processed(view_id_):
        raw_feat, mask = view_cache.get(view_id_)
        processed = postproc(raw_feat.unsqueeze(0))[0]  # [C, H, W]
        return processed, mask

    for step in tqdm(range(args.num_steps), desc='training postproc'):
        optimizer.zero_grad()

        cv_idx = np.random.choice(len(cv_a), size=min(args.pairs_per_step, len(cv_a)), replace=False)
        mir_idx = np.random.choice(len(mir_a), size=min(args.pairs_per_step, len(mir_a)), replace=False)

        # cache processed maps per unique view touched this step, so a view referenced
        # multiple times (as query or target) only runs through postproc once
        processed_cache = {}

        def processed_for(v):
            if v not in processed_cache:
                processed_cache[v] = get_processed(v)
            return processed_cache[v]

        cv_losses = []
        for i in cv_idx:
            a, b = int(cv_a[i]), int(cv_b[i])
            va, vb = int(view_id[a]), int(view_id[b])
            query_map, _ = processed_for(va)
            target_map, target_mask = processed_for(vb)
            query_feat = query_map[:, int(row[a]), int(col[a])]
            l = soft_argmax_loss(query_feat, target_map, target_mask,
                                  int(row[b]), int(col[b]), args.softmax_temperature)
            cv_losses.append(l)

        mir_losses = []
        for i in mir_idx:
            a, b = int(mir_a[i]), int(mir_b[i])
            va = int(view_id[a])  # same view for both, by construction (same_view_only)
            target_map, target_mask = processed_for(va)
            query_feat = target_map[:, int(row[a]), int(col[a])]
            # true location is the query's OWN location -- the mirror twin at
            # (row[b], col[b]) sits in the same softmax denominator as a hard negative
            l = soft_argmax_loss(query_feat, target_map, target_mask,
                                  int(row[a]), int(col[a]), args.softmax_temperature)
            mir_losses.append(l)

        cv_loss = torch.stack(cv_losses).mean() if cv_losses else torch.zeros((), device=device)
        mir_loss = torch.stack(mir_losses).mean() if mir_losses else torch.zeros((), device=device)
        loss = cv_loss + args.mirror_loss_weight * mir_loss
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == args.num_steps - 1:
            tqdm.write(f'step {step}: total={loss.item():.4f} cross_view={cv_loss.item():.4f} '
                       f'mirror={mir_loss.item():.4f}')

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, 'postproc.pth')
    save_checkpoint(postproc, ckpt_path, in_dim=args.width, hidden_dim=args.hidden_dim,
                     l2_normalize=bool(args.l2_normalize))
    print(f'saved to {ckpt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_data_dir', type=str, required=True)
    parser.add_argument('--render_res', type=int, default=224)
    parser.add_argument('--n_views', type=int, default=1000)
    parser.add_argument('--feature_model_type', type=str, default='dino_sd')
    parser.add_argument('--width', type=int, default=2304)
    parser.add_argument('--out_dir', type=str, required=True)

    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--l2_normalize', type=int, default=1)

    parser.add_argument('--n_cross_view_pairs', type=int, default=200_000)
    parser.add_argument('--n_mirror_pairs', type=int, default=120_000)
    parser.add_argument('--cross_view_max_dist', type=float, default=0.02,
                        help='empirically measured on this mesh: 99th pct match dist ~0.003 at this threshold')
    parser.add_argument('--mirror_max_dist', type=float, default=0.02,
                        help='empirically measured: yields ~120k same-view mirror pairs across 946/1000 views')
    parser.add_argument('--mirror_axis', type=int, default=0)
    parser.add_argument('--mirror_min_axis_offset', type=float, default=0.01)
    parser.add_argument('--mirror_same_view_only', type=int, default=1)
    parser.add_argument('--pairs_cache_path', type=str, default=None)

    parser.add_argument('--softmax_temperature', type=float, default=0.05)
    parser.add_argument('--mirror_loss_weight', type=float, default=1.0)
    parser.add_argument('--pairs_per_step', type=int, default=64,
                        help='cross-view and mirror pairs sampled independently, this many each per step')
    parser.add_argument('--lru_cache_size', type=int, default=200,
                        help='max views held in RAM at once (~35MB/view for width=2304 at 64x64)')

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_steps', type=int, default=20_000)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    main(args)
