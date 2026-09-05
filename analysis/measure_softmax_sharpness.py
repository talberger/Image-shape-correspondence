"""
Measure HOW DIFFERENTIATED a checkpoint's soft-mapping weights actually are.

This is the direct measurement behind the 2026-08-25 softmax-temperature finding: a phase-2
model trained at a low temperature SHRINKS its own within-top-k cosine gaps to restore a
near-uniform softmax, so the temperature you train at is NOT the sharpness you get. The
quantity that matters is the REALIZED sharpness of the trained model read at its own
temperature, which is what this script reports.

Reported per checkpoint (over the 100-point eval set by default, or the full masked set with
--full_mask):
  gap_maxmin   max(sim) - min(sim) within each pixel's top-k, median over pixels. This is the
               "within-top-10 cosine gap" quoted in PROJECT_NOTES.md (baseline 0.0292,
               trained-at-T=0.005 0.0114 -- a 2.5x self-compensation).
  gap_top1top2 the top-1 minus top-2 similarity, median -- the tie-breaking margin specifically,
               which gap_maxmin can hide.
  effective_k  exp(entropy of the weight distribution), i.e. perplexity. k (=10) means perfectly
               uniform / no confidence at all; 1 means all mass on one vertex. This is the number
               to read as "how many vertices is this pixel really averaging over".
  top1_w       the largest softmax weight, median. 1/k (=0.10) is uniform.

Run (a trained checkpoint read at its OWN training temperature -- the meaningful comparison):
python analysis/measure_softmax_sharpness.py \
  --vertex_features_path experiments/guitar/encoder_dino_sd_dfd_crossview_relational_geodesic_mu3dscale1_T0.01_phase2/pred_f.pth \
  --temperature 0.01 --run_label "trained T=0.01, read 0.01"
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pixel_to_vertex_nn import load_features, load_vertex_features
from modules.mapping_loss import soft_map_to_vertices
from modules.utils import loadmesh


def main(args):
    features = load_features(args.features_path)
    vertex_features = load_vertex_features(args.vertex_features_path, features.shape[-1])
    num_vertices = vertex_features.shape[0]
    mesh = loadmesh(dir=args.obj_path, name=args.name)

    if args.full_mask:
        mask = torch.load(args.mask_path, map_location='cpu')
        if isinstance(mask, dict):
            mask = mask['mask'] if 'mask' in mask else list(mask.values())[0]
        mask = mask.squeeze().bool()
        rows, cols = torch.where(mask)
        source = f'{len(rows)} masked pixels from {args.mask_path}'
    else:
        pairs = torch.load(args.eval_pairs_path)
        rows, cols = pairs['row'], pairs['col']
        source = f'{len(rows)} labeled eval pixels from {args.eval_pairs_path}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vert_feats = vertex_features.to(device)
    vert_pos = mesh.vertices.to(device).float()

    # Chunked so --full_mask (~212k pixels) fits: the [n, V] similarity inside
    # soft_map_to_vertices is the dominant allocation.
    gaps_maxmin, gaps_12, eff_ks, top1_ws, rank_ws = [], [], [], [], []
    for start in range(0, len(rows), args.chunk):
        r, c = rows[start:start + args.chunk], cols[start:start + args.chunk]
        pixel_feats = F.normalize(features[r, c].to(device).float().detach(), dim=-1)
        _, topk_idx, weights = soft_map_to_vertices(
            pixel_feats, vert_feats, vert_pos, args.k_vertices_per_pixel,
            return_topk=True, temperature=args.temperature)

        # Recover the top-k cosine similarities themselves (soft_map_to_vertices returns only
        # the post-softmax weights, and at T != 1 those are not invertible to the raw gaps).
        sims = (pixel_feats @ F.normalize(vert_feats, dim=-1).T).gather(1, topk_idx)
        gaps_maxmin.append((sims.max(dim=1).values - sims.min(dim=1).values).cpu())
        s_sorted = sims.sort(dim=1, descending=True).values
        gaps_12.append((s_sorted[:, 0] - s_sorted[:, 1]).cpu())

        w = weights.double().clamp_min(1e-30)
        entropy = -(w * w.log()).sum(dim=1)
        eff_ks.append(entropy.exp().cpu())
        top1_ws.append(w.max(dim=1).values.cpu())
        rank_ws.append(w.sort(dim=1, descending=True).values.cpu())

    gap_maxmin = torch.cat(gaps_maxmin)
    gap_12 = torch.cat(gaps_12)
    eff_k = torch.cat(eff_ks)
    top1_w = torch.cat(top1_ws)
    rank_w = torch.cat(rank_ws)

    print(f'\nrun: {args.run_label}')
    print(f'  {source}')
    print(f'  checkpoint : {args.vertex_features_path}')
    print(f'  read at T  : {args.temperature}   (k={args.k_vertices_per_pixel}, '
          f'{num_vertices} vertices)')
    print(f'  {"":<14}{"median":>10}{"mean":>10}{"min":>10}{"max":>10}')
    for name, v in (('gap_maxmin', gap_maxmin), ('gap_top1top2', gap_12),
                    ('effective_k', eff_k), ('top1_w', top1_w)):
        print(f'  {name:<14}{v.median():>10.4f}{v.float().mean():>10.4f}'
              f'{v.min():>10.4f}{v.max():>10.4f}')
    print(f'  (uniform reference: effective_k = {args.k_vertices_per_pixel}, '
          f'top1_w = {1.0 / args.k_vertices_per_pixel:.4f})')

    # Per-rank weight profile: the median weight the 1st, 2nd, ... k-th candidate receives, i.e.
    # the actual SHAPE of the distribution the scalars above only summarize. Added 2026-08-27
    # (Tal: "how much does top-1 get, and how many of the 10 really carry any weight").
    med_w = rank_w.median(dim=0).values
    cum = med_w.cumsum(0) / med_w.sum()
    print(f'  per-rank median weight (rank 1..{args.k_vertices_per_pixel}):')
    print('    rank  ' + ''.join(f'{i + 1:>8}' for i in range(len(med_w))))
    print('    w     ' + ''.join(f'{v:>8.4f}' for v in med_w))
    print('    cum%  ' + ''.join(f'{v * 100:>8.1f}' for v in cum))

    if args.save_path:
        torch.save({'gap_maxmin': gap_maxmin, 'gap_top1top2': gap_12, 'effective_k': eff_k,
                    'top1_w': top1_w, 'rank_weights': rank_w, 'temperature': args.temperature,
                    'vertex_features_path': args.vertex_features_path,
                    'run_label': args.run_label}, args.save_path)
        print(f'  saved per-pixel values to {args.save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vertex_features_path', type=str, required=True,
                        help='a phase-2 run\'s pred_f.pth (the encoder output being judged)')
    parser.add_argument('--features_path', type=str,
                        default='./demo/guitar/features_analyze/dino_sd_poc/'
                                'guitar_01_pix_dino_sd_features.pth')
    parser.add_argument('--eval_pairs_path', type=str,
                        default='./demo/guitar/eval_labels/pixel_vertex_eval_pairs.pth')
    parser.add_argument('--full_mask', action='store_true',
                        help='measure over every masked pixel instead of the 100 eval points')
    parser.add_argument('--mask_path', type=str,
                        default='./demo/guitar/segment_masks_pixel270_370/'
                                'guitar_01_click_270_370_sam_mask_2.pth')
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')
    parser.add_argument('--k_vertices_per_pixel', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='read-time softmax temperature. To judge a trained model, pass the '
                             'temperature it was TRAINED at.')
    parser.add_argument('--chunk', type=int, default=8192)
    parser.add_argument('--run_label', type=str, default='(unlabeled run)')
    parser.add_argument('--save_path', type=str, default=None)
    main(parser.parse_args())
