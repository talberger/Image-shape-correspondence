"""
Label-free quality measures for a whole correspondence field.

WHY THIS EXISTS (2026-09-08, Tal): *"מבחינה ויזואלית פאזה2 משפרת בהרבה, למרות שזה לא בא לידי ביטוי
במדדים שלנו"*. The n=100 ground-truth eval scores accuracy AT 100 labeled pixels. The
back-projection Tal actually judges runs by shows the mapping of ALL ~212k masked pixels. Those are
different things: a field that is accurate where it was measured but degenerate everywhere else
scores well and looks bad. This measures the everywhere-else, with no labels involved.

Three families, all computed on the top-1 pixel->vertex map:

  SPATIAL COHERENCE -- for each pair of 4-adjacent masked pixels, the geodesic distance between the
  two vertices they map to, as a % of the mesh geodesic diameter. Neighbouring pixels are almost
  always neighbouring surface points, so a coherent field keeps this tiny; a field that scatters
  produces large jumps. `discontinuity rate` is the fraction of adjacent pairs landing further than
  --jump_threshold apart -- the visible speckle in a back-projection.

  MESH USAGE -- how the 212k pixels distribute over the 3886 vertices: how many vertices get used at
  all (`coverage`, the number this project has quoted for months), plus the concentration of the
  load (top-1% share, normalized entropy). Coverage alone cannot tell "spread out evenly" from "one
  vertex swallowing 20k pixels while the rest take a handful"; the entropy can.

  SEGMENT COMPACTNESS -- using the existing 17-segment fine map: for each photo segment, the median
  geodesic spread of the vertices its pixels land on. A correspondence that respects the object maps
  a compact photo region onto a compact surface region.

Run (ZS vs its phase-2, the comparison this was written for):
cd /workspace/bsb && python analysis/correspondence_field_quality.py \
  --knn_paths demo/guitar/runs/isobbox/bsb_isobbox_ZS_sdsquare/analyze/guitar_01_click_270_370_sam_mask_2_pixel_to_vertex_nn.pth \
               demo/guitar/runs/isobbox/bsb_isobbox_sdsquare_olddiag2d_T1_seed0_phase2/analyze/guitar_01_click_270_370_sam_mask_2_pixel_to_vertex_nn.pth \
  --labels "ZS" "phase-2 T=1" \
  --geo_table_path ./experiments/guitar/geodesic_tables/guitar_isobbox_geo_table_3d.pth \
  --fine_segments_path demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_geo_table(path):
    g = torch.load(path, map_location='cpu')
    table = g['table'] if isinstance(g, dict) else g
    return table.float()


def spatial_coherence(knn, geo, diameter, jump_threshold):
    """Geodesic gap between the vertices of 4-adjacent masked pixels."""
    row, col = knn['pixel_row'].long(), knn['pixel_col'].long()
    v = knn['top1_vertex_idx'].long()
    h, w = knn['mask_shape']
    # Dense pixel -> index lookup so neighbours can be found by arithmetic rather than a search.
    lut = torch.full((h * w,), -1, dtype=torch.long)
    lut[row * w + col] = torch.arange(row.numel())

    gaps = []
    for dr, dc in ((0, 1), (1, 0)):
        nr, nc = row + dr, col + dc
        ok = (nr < h) & (nc < w)
        j = torch.full_like(row, -1)
        j[ok] = lut[nr[ok] * w + nc[ok]]
        both = j >= 0
        gaps.append(geo[v[both], v[j[both]]])
    gaps = torch.cat(gaps) / diameter * 100
    return {
        'pairs': gaps.numel(),
        'mean': gaps.mean().item(),
        'median': gaps.median().item(),
        'p90': gaps.quantile(0.9).item(),
        'discontinuity_rate': (gaps > jump_threshold).float().mean().item() * 100,
    }


def mesh_usage(knn):
    v = knn['top1_vertex_idx'].long()
    n = int(knn['num_vertices'])
    counts = torch.bincount(v, minlength=n).float()
    used = counts[counts > 0]
    p = used / used.sum()
    entropy = -(p * p.log()).sum() / torch.log(torch.tensor(float(n)))  # vs perfectly uniform over ALL vertices
    top1pct = int(max(1, round(0.01 * n)))
    share = used.sort(descending=True).values[:top1pct].sum() / used.sum()
    return {
        'coverage': used.numel(),
        'coverage_pct': 100 * used.numel() / n,
        'entropy': entropy.item(),
        'top1pct_share': share.item() * 100,
        'max_pixels_on_one_vertex': int(counts.max().item()),
    }


def segment_compactness(knn, geo, diameter, fine_path):
    """Median pairwise geodesic spread of the vertices each photo segment lands on."""
    fine = torch.load(fine_path, map_location='cpu')
    seg_map = fine['segments'] if isinstance(fine, dict) and 'segments' in fine else fine
    if not torch.is_tensor(seg_map):
        return None
    row, col = knn['pixel_row'].long(), knn['pixel_col'].long()
    v = knn['top1_vertex_idx'].long()
    seg = seg_map[row, col] if seg_map.dim() == 2 else seg_map
    spreads = []
    for s in seg.unique():
        if s < 0:
            continue
        vs = v[seg == s].unique()
        if vs.numel() < 2:
            continue
        sub = geo[vs][:, vs]
        spreads.append(sub[sub > 0].median().item() / diameter * 100)
    if not spreads:
        return None
    t = torch.tensor(spreads)
    return {'segments': t.numel(), 'median_spread': t.median().item(), 'mean_spread': t.mean().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--knn_paths', nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--geo_table_path', required=True)
    p.add_argument('--fine_segments_path', default=None)
    p.add_argument('--jump_threshold', type=float, default=2.0,
                   help='%% of mesh geodesic diameter above which an adjacent-pixel pair counts as '
                        'a discontinuity')
    args = p.parse_args()
    assert len(args.knn_paths) == len(args.labels), '--knn_paths and --labels must match'

    geo = load_geo_table(args.geo_table_path)
    diameter = geo.max()
    print(f'geodesic diameter {diameter:.4f}, jump threshold {args.jump_threshold}% of it\n')

    rows = []
    for path, label in zip(args.knn_paths, args.labels):
        knn = torch.load(path, map_location='cpu')
        rows.append((label, spatial_coherence(knn, geo, diameter, args.jump_threshold),
                     mesh_usage(knn),
                     segment_compactness(knn, geo, diameter, args.fine_segments_path)
                     if args.fine_segments_path else None))

    print('SPATIAL COHERENCE -- geodesic gap between 4-adjacent pixels, % of diameter')
    print(f'  {"run":<18}{"mean":>8}{"median":>9}{"p90":>8}{"discontinuity rate":>21}')
    for label, sc, _, _ in rows:
        print(f'  {label:<18}{sc["mean"]:>7.2f}%{sc["median"]:>8.2f}%{sc["p90"]:>7.2f}%'
              f'{sc["discontinuity_rate"]:>20.2f}%')

    print('\nMESH USAGE -- how 212k pixels spread over the vertices')
    print(f'  {"run":<18}{"coverage":>16}{"entropy":>10}{"top-1% share":>14}{"max on 1 vtx":>14}')
    for label, _, mu, _ in rows:
        print(f'  {label:<18}{mu["coverage"]:>7}/{3886} {mu["coverage_pct"]:>5.1f}%'
              f'{mu["entropy"]:>10.3f}{mu["top1pct_share"]:>13.1f}%{mu["max_pixels_on_one_vertex"]:>14}')

    if rows[0][3]:
        print('\nSEGMENT COMPACTNESS -- median geodesic spread of the vertices a photo segment maps to')
        print(f'  {"run":<18}{"segments":>10}{"median":>10}{"mean":>10}')
        for label, _, _, sg in rows:
            print(f'  {label:<18}{sg["segments"]:>10}{sg["median_spread"]:>9.2f}%{sg["mean_spread"]:>9.2f}%')


if __name__ == '__main__':
    main()
