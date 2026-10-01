"""The learned mu, per direction, drawn on the fine-segment map (Tal, 2026-09-21).

For every mu group of a phase-2 run -- a segment (segment / segment_aniso), a grid node (grid_aniso) or
the whole object (global_aniso) -- draws at that group's location:
  white  the mu = 1 reference circle
  cyan   (optional, --iso_run_dir) the ISOTROPIC run's mu for the same segment, as a circle
  red    mu(theta): the distance from the centre to the red curve in image direction theta is the mu a
         centre->neighbour pair in that direction gets (a polar plot of sqrt(u^T M u) -- close to, but
         not exactly, an ellipse)
and prints a table: iso mu | mu vertical | mu horizontal | major/minor | major-axis angle (0 = vertical).

Usage:
  python analysis/plot_mu_ellipses.py --run_dir experiments/shic_bear/encoder_..._phase2 \
    --fine_seg_path demo/shic/bear3/bear3_fine_segments.pth --img_path demo/shic/images/bear3.png \
    [--iso_run_dir experiments/shic_bear/encoder_..._rerun0921b_phase2] --output_path out.png
"""
import argparse
import math
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def mu_dir(p, th_deg):
    """mu for image direction th (0 = along rows / vertical, 90 = along columns / horizontal)."""
    u0, u1 = math.cos(math.radians(th_deg)), math.sin(math.radians(th_deg))
    a, b, c = p
    return math.hypot(math.exp(a) * u0 + c * u1, math.exp(b) * u1)


def rep_pixel(mask):
    """A pixel INSIDE the region, nearest its median -- so the marker never lands outside a
    non-convex segment."""
    ys, xs = np.nonzero(mask)
    cy, cx = np.median(ys), np.median(xs)
    k = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
    return ys[k], xs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--fine_seg_path', required=True)
    ap.add_argument('--img_path', required=True)
    ap.add_argument('--iso_run_dir', default=None, help='an isotropic per-segment run on the SAME segments')
    ap.add_argument('--output_path', required=True)
    ap.add_argument('--radius', type=float, default=45.0, help='pixels for mu = 1')
    ap.add_argument('--title', default=None)
    ap.add_argument('--min_frac', type=float, default=0.0,
                    help='2026-09-22 (Tal): skip segments smaller than this fraction of the object (slivers that are '
                         'invisible in the map but still got an ellipse). Skipped ones are listed, not drawn.')
    ap.add_argument('--no_legend', action='store_true',
                    help='do not write the title/legend INTO the image (it gets cut on narrow photos); the caller '
                         'writes it under the figure instead')
    args = ap.parse_args()

    seg = torch.load(args.fine_seg_path, map_location='cpu').numpy()
    img = Image.open(args.img_path).convert('RGB').resize((seg.shape[1], seg.shape[0]))
    rng = np.random.RandomState(3)
    pal = (rng.rand(int(seg.max()) + 1, 3) * 200 + 40).astype(np.uint8)
    ov = np.asarray(img).copy()
    m = seg > 0
    ov[m] = (0.45 * ov[m] + 0.55 * pal[seg[m]]).astype(np.uint8)
    im = Image.fromarray(ov)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 15)
    except OSError:
        font = ImageFont.load_default()

    iso = None
    if args.iso_run_dir:
        iso = torch.load(os.path.join(args.iso_run_dir, 'mu_scale.pth'), map_location='cpu')['mu_scale']

    aniso_path = os.path.join(args.run_dir, 'mu_aniso.pth')
    if os.path.exists(aniso_path):
        A = torch.load(aniso_path, map_location='cpu')
        P = A['mu_aniso'].tolist()
        scope = A.get('scope', 'segment_aniso')     # files written before the key existed are per segment
    else:                                           # an isotropic run: a = b = log s, c = 0
        s = torch.load(os.path.join(args.run_dir, 'mu_scale.pth'), map_location='cpu')['mu_scale']
        P = [[math.log(v), math.log(v), 0.0] for v in s.tolist()]
        scope = 'segment' if len(P) > 1 else 'global'

    # (label, pixel (row, col), params, iso or None)
    groups = []
    if scope in ('segment', 'segment_aniso', 'segment_aniso_xglobal', 'global_then_segment_aniso'):
        for g in range(1, int(seg.max()) + 1):
            frac = (seg == g).sum() / max(m.sum(), 1)
            if (seg == g).any() and frac < args.min_frac:
                print('seg %d skipped: %.2f%% of the object < --min_frac' % (g, 100 * frac))
            elif (seg == g).any():
                groups.append(('seg %d' % g, rep_pixel(seg == g), P[g], None if iso is None else iso[g].item()))
    elif scope in ('global_aniso', 'global'):
        groups.append(('all', rep_pixel(m), P[0], None))
    elif scope == 'grid_aniso':
        G = A['grid']
        rmin, rmax, cmin, cmax = A['grid_bbox']
        for i in range(G):
            for j in range(G):
                r = int(round(rmin + (rmax - rmin) * i / (G - 1)))
                c = int(round(cmin + (cmax - cmin) * j / (G - 1)))
                if 0 <= r < seg.shape[0] and 0 <= c < seg.shape[1] and m[r, c]:   # nodes on the object only
                    groups.append(('node %d,%d' % (i, j), (r, c), P[i * G + j], None))
    else:
        raise SystemExit('unknown scope %r' % scope)
    if scope in ('segment_aniso_xglobal', 'global_then_segment_aniso'):
        # the hybrid's CROSS-segment ellipse = the last row; drawn in magenta in the image's top-right corner
        groups.append(('xglobal', (int(1.6 * args.radius), seg.shape[1] - int(1.6 * args.radius)), P[-1], None))

    R = args.radius * (2.0 if scope in ('global_aniso', 'global') else 1.0)
    print('%-10s %8s %8s %8s %8s %8s' % ('group', 'iso', 'vert', 'horiz', 'ratio', 'major'))
    for label, (cy, cx), p, iso_v in groups:
        circ = lambda rad: [(cx + rad * math.sin(math.radians(t)), cy + rad * math.cos(math.radians(t)))
                            for t in range(0, 361, 5)]
        d.line(circ(R), fill=(255, 255, 255), width=1)
        if iso_v is not None:
            d.line(circ(R * iso_v), fill=(0, 230, 255), width=2)
        curve = [(cx + R * mu_dir(p, t) * math.sin(math.radians(t)),
                  cy + R * mu_dir(p, t) * math.cos(math.radians(t))) for t in range(0, 361, 5)]
        d.line(curve, fill=(255, 0, 255) if label == 'xglobal' else (255, 0, 0), width=3)
        if scope != 'grid_aniso':
            d.text((cx + 4, cy - 8), label.replace('seg ', ''), fill=(255, 255, 0), font=font,
                   stroke_width=2, stroke_fill=(0, 0, 0))
        vals = [mu_dir(p, t) for t in range(0, 180)]
        th_max = int(np.argmax(vals))
        print('%-10s %8s %8.3f %8.3f %8.2f %6d deg'
              % (label, '-' if iso_v is None else '%.3f' % iso_v, mu_dir(p, 0), mu_dir(p, 90),
                 max(vals) / min(vals), th_max))
    if args.no_legend:
        args.title = None
    legend = '' if args.no_legend else 'red: mu per direction   white: mu = 1' + ('   cyan: isotropic run' if iso is not None else '')
    if legend: d.text((8, 8), (args.title + '  |  ' if args.title else '') + legend, fill=(255, 255, 255), font=font,
           stroke_width=2, stroke_fill=(0, 0, 0))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    im.save(args.output_path)
    print('saved', args.output_path)


if __name__ == '__main__':
    main()
