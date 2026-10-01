"""Geodesic diameter of the VISIBLE part of a mesh in a photo (Tal, 2026-09-16, bear proportions).

The rkd_spatial_dist loss divides d_3D by the geodesic diameter of the WHOLE mesh, i.e. it assumes the photo
spans the whole object. For a 3/4 bear photo it does not. This picks the phase-1 render view whose silhouette
best matches the photo mask (bbox-aligned IoU, no correspondence used -- SHIC is NOT consulted), takes the
surface points visible in that view (xyz_bary_f maps), snaps them to vertices and measures their geodesic
diameter. Prints max and p99 over visible-vertex pairs, as a fraction of the full diameter too.
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F


def crop_resize(m, size=64):
    ys, xs = torch.nonzero(m, as_tuple=True)
    c = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].float()[None, None]
    return F.interpolate(c, (size, size), mode='bilinear', align_corners=False)[0, 0] > 0.5, \
        (ys.max() - ys.min() + 1).item() / (xs.max() - xs.min() + 1).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mask_path', required=True)
    p.add_argument('--xyz_dir', required=True, help='.../encoder_data/xyz_bary_f/224')
    p.add_argument('--geo_table_3d_path', required=True)
    p.add_argument('--obj_path', required=True)
    p.add_argument('--top_views', type=int, default=5)
    p.add_argument('--aspect_tol', type=float, default=0.25, help='reject views whose bbox aspect differs more')
    a = p.parse_args()

    from modules.utils import loadmesh
    mesh = loadmesh(a.obj_path, 'm')
    V = mesh.vertices.detach().float().cpu()
    g = torch.load(a.geo_table_3d_path, map_location='cpu')
    G, D = g['table'].float(), float(g['diagonal'])

    mask = torch.load(a.mask_path, map_location='cpu').bool()
    pm, p_aspect = crop_resize(mask)
    scores = []
    files = sorted(glob.glob(os.path.join(a.xyz_dir, 'target_xyz_bary_f_*.pt')))
    for f in files:
        x = torch.load(f, map_location='cpu')['model_f']
        sil = (x != 2).any(0)          # background is filled with 2.0
        if sil.sum() < 20:
            continue
        rm, r_aspect = crop_resize(sil)
        if abs(r_aspect - p_aspect) / p_aspect > a.aspect_tol:
            continue
        for flip in (False, True):          # the photo may show the mirror side
            r = rm.flip(1) if flip else rm
            iou = (r & pm).sum().item() / max((r | pm).sum().item(), 1)
            scores.append((iou, f, flip))
    scores.sort(reverse=True)
    print('photo bbox aspect %.3f, %d candidate (view, flip) pairs' % (p_aspect, len(scores)))
    rows = []
    for iou, f, flip in scores[:a.top_views]:
        x = torch.load(f, map_location='cpu')
        pts = x['model_f'].reshape(3, -1).t()
        pts = pts[(pts != 2).any(1)]
        vis = torch.cdist(pts, V).argmin(1).unique()
        S = G[vis][:, vis]
        mx, p99 = S.max().item(), torch.quantile(S.flatten()[torch.randperm(S.numel())[:4000000]], 0.99).item()
        rows.append((mx, p99))
        print('IoU %.3f  view %s flip=%s elev=%.2f azim=%.2f  visible v=%d  max %.4f (%.3f D)  p99 %.4f (%.3f D)'
              % (iou, os.path.basename(f), flip, float(x['elev']), float(x['azim']), len(vis), mx, mx / D, p99, p99 / D))
    mx = sum(r[0] for r in rows) / len(rows)
    print('\nfull diameter D = %.5f' % D)
    print('VISIBLE geodesic diameter (mean of max over top %d views): %.5f  = %.3f D' % (len(rows), mx, mx / D))


if __name__ == '__main__':
    main()
