"""Fixed per-segment mu_3d_scale from a FROZEN reference mapping (Tal, 2026-09-16, bear proportions, idea 2).

For each fine segment k: sample in-segment pixel pairs (centre from the 2D table's centres, partner = any
pixel of the same segment), read d_2D from the 2D geodesic table and d_3D between the two pixels' top1
vertices under the reference mapping (ZS by default). The segment's extent on each side is the p-th
percentile of those SAME pairs (percentile, not max: the ZS mapping is noisy and a max would be an outlier).
    s_k = (P_3D,k / D_3D) / (P_2D,k / D_2D)
so that norm_d_3D / s_k matches norm_d_2D at the segment's extent. Computed ONCE, saved, fed to
train_relational_loss.py --mu_init_path with --mu_lr 0 (i.e. fixed). Segments with < min_pairs get 1.0.
"""
import argparse, os, sys
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--knn_path', required=True)
    p.add_argument('--geo_table_2d_path', required=True)
    p.add_argument('--geo_table_3d_path', required=True)
    p.add_argument('--fine_seg_path', required=True)
    p.add_argument('--geo_diag_2d', type=float, required=True)
    p.add_argument('--geo_diag_3d', type=float, default=None, help='default: the 3D table diagonal')
    p.add_argument('--percentile', type=float, default=0.99)
    p.add_argument('--pairs_per_center', type=int, default=256)
    p.add_argument('--min_pairs', type=int, default=2000)
    p.add_argument('--output_path', required=True)
    a = p.parse_args()
    torch.manual_seed(0)

    knn = torch.load(a.knn_path, map_location='cpu')
    t2 = torch.load(a.geo_table_2d_path, map_location='cpu')
    g3 = torch.load(a.geo_table_3d_path, map_location='cpu')
    G, D3 = g3['table'].float(), (a.geo_diag_3d or float(g3['diagonal']))
    seg = torch.as_tensor(torch.load(a.fine_seg_path, map_location='cpu')).long()
    H, W = seg.shape

    top1 = torch.full((H, W), -1, dtype=torch.long)
    top1[knn['pixel_row'].long(), knn['pixel_col'].long()] = knn['top1_vertex_idx'].long()
    p2n = t2['pixel_to_node'].long()                       # [H, W] -> column of the 2D table, -1 off-graph
    node_rc = torch.nonzero(p2n >= 0)
    node_rc = node_rc[p2n[node_rc[:, 0], node_rc[:, 1]].argsort()]   # row i = pixel of node i
    node_seg = seg[node_rc[:, 0], node_rc[:, 1]]
    node_v = top1[node_rc[:, 0], node_rc[:, 1]]
    cen = t2['coords'].long()
    cen_seg, cen_v = seg[cen[:, 0], cen[:, 1]], top1[cen[:, 0], cen[:, 1]]

    n_groups = int(seg.max()) + 1
    s = torch.ones(n_groups)
    print('segment  centres  pairs   P2D(px)  P2D/D2   P3D    P3D/D3   s_k')
    for k in range(n_groups):
        ci = torch.nonzero(cen_seg == k).flatten()
        nodes = torch.nonzero((node_seg == k) & (node_v >= 0)).flatten()
        if len(ci) == 0 or len(nodes) < 2:
            print('%7d  %7d  (no pairs) -> 1.0' % (k, len(ci)));  continue
        pick = nodes[torch.randint(len(nodes), (len(ci), a.pairs_per_center))]
        d2 = t2['table'][ci[:, None], pick].float()
        d3 = G[cen_v[ci][:, None], node_v[pick]]
        ok = torch.isfinite(d2) & (d2 > 0)
        d2, d3 = d2[ok], d3[ok]
        if d2.numel() < a.min_pairs:
            print('%7d  %7d  %6d  too few -> 1.0' % (k, len(ci), d2.numel()));  continue
        P2, P3 = torch.quantile(d2, a.percentile).item(), torch.quantile(d3, a.percentile).item()
        s[k] = (P3 / D3) / (P2 / a.geo_diag_2d)
        print('%7d  %7d  %6d  %7.1f  %.3f   %.4f  %.3f   %.3f' % (k, len(ci), d2.numel(), P2, P2 / a.geo_diag_2d, P3, P3 / D3, s[k]))
    os.makedirs(os.path.dirname(a.output_path), exist_ok=True)
    torch.save({'mu_scale': s, 'args': vars(a)}, a.output_path)
    print('saved', a.output_path, 's =', [round(x, 3) for x in s.tolist()])


if __name__ == '__main__':
    main()
