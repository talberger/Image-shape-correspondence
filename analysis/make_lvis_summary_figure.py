"""One summary figure per DensePose-LVIS instance: ours vs SHIC, everything in one frame.

Tal's layout request, 2026-09-20. Per instance:
  row 1  the original crop with the GT points marked | OUR back-projection (soft-snap) | SHIC's (argmax)
  row 2  the mesh from several azimuths, with GT / ours / SHIC markers
  side   a legend and the geodesic errors that were actually computed

The two back-projections use DIFFERENT decoding rules on purpose, because that is each method's own
rule: ours is soft-snap (read T=0.01, k=10), SHIC's is argmax (top-1). The per-point table reports
both of ours so the argmax line is directly comparable to theirs.

Errors are the 228-normalized geodesic(GT vertex, predicted vertex) -- the same units as SHIC
Table 1 (bear: SHIC 33.0, CSE 35.7, zero-shot SD-DINO 48.2).

Inputs are the files the other scripts already wrote; this only composes them.

Run:
cd $DISC_ROOT && python analysis/make_lvis_summary_figure.py --ann_id 117683 \
  --base demo/shic/lvis_vs_shic --run_dir data/densepose_lvis/runs/lvis117683 \
  --output_path demo/shic/lvis_vs_shic/SUMMARY_lvis117683.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image

ROLE_GT, ROLE_OURS, ROLE_SHIC = (60 / 255, 180 / 255, 75 / 255), (0, 110 / 255, 230 / 255), (230 / 255, 30 / 255, 30 / 255)


def split_side_by_side(path):
    """color_image_from_knn.py writes [SQUARE coloured-mesh render | back-projection]. The render is h x h, the
    back-projection has the photo's own width, so the seam is at x = h, NOT at w // 2 (2026-10-05: cutting at w // 2
    left a sliver of the render inside the back-projection panel and hid the coloured mesh)."""
    im = Image.open(path).convert('RGB')
    w, h = im.size
    return im.crop((0, 0, h, h)), im.crop((h, 0, w, h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann_id', type=int, required=True)
    ap.add_argument('--base', default='demo/shic/lvis_vs_shic')
    ap.add_argument('--run_dir', default=None)
    ap.add_argument('--output_path', required=True)
    ap.add_argument('--class_name', default='bear',
                    help='2026-09-22: non-bear classes print their own SHIC reproduction from '
                         'data/densepose_lvis/eval_shic_<class>/results.json instead of the bear Table-1 block')
    args = ap.parse_args()
    tag = f'lvis{args.ann_id}'
    base = args.base

    res = json.load(open(os.path.join(base, f'{tag}_vs_shic.json')))
    photo = Image.open(os.path.join(base, f'{tag}_photo_gt_points.png')).convert('RGB')
    mesh_col, ours_bp = split_side_by_side(os.path.join(base, f'backproj_{tag}', f'{tag}_OURS_soft_backproj_side_by_side.png'))
    _, shic_bp = split_side_by_side(os.path.join(base, f'backproj_{tag}', 'shic', f'{tag}_SHIC_argmax_backproj_side_by_side.png'))
    w, h = mesh_col.size                       # drop the render's own "elev/azim" caption strip and the empty margins
    # cropped to the animal itself (non-white pixels below the caption strip), not to fixed fractions: those were
    # tuned on the bear render and cut the horse's legs and back (2026-10-05)
    arr = np.asarray(mesh_col)[int(0.12 * h):]
    ys, xs = np.nonzero((arr < 245).any(-1))
    pad = int(0.03 * w)
    mesh_col = mesh_col.crop((max(xs.min() - pad, 0), max(ys.min() + int(0.12 * h) - pad, 0),
                              min(xs.max() + pad, w), min(ys.max() + int(0.12 * h) + pad, h)))
    # padded with white to the photo's panel shape (this column is 0.9x as wide), so its axes is as tall as its
    # neighbours' and the title lines up with theirs (Tal, 2026-10-05)
    mw, mh = mesh_col.size
    asp = (photo.size[1] / photo.size[0]) / 0.9               # the panel shape to match, height / width
    tw, th = max(mw, int(round(mh / asp))), max(mh, int(round(mw * asp)))   # pad whichever side is short
    canvas = Image.new('RGB', (tw, th), 'white'); canvas.paste(mesh_col, ((tw - mw) // 2, (th - mh) // 2)); mesh_col = canvas
    # 2026-10-05: markers drawn in 2D over the render (analysis/render_lvis_markers_2d.py) when that file exists
    mesh2d = os.path.join(base, f'{tag}_mesh_views_2d.png')
    mesh = Image.open(mesh2d if os.path.exists(mesh2d) else os.path.join(base, f'{tag}_mesh_views.png')).convert('RGB')

    fig = plt.figure(figsize=(21, 9))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1.0], width_ratios=[1, 0.9, 1, 1, 0.95],
                          hspace=0.16, wspace=0.06)

    for ax, im, title in ((fig.add_subplot(gs[0, 0]), photo, f'original crop + GT points  ({tag})'),
                          (fig.add_subplot(gs[0, 1]), mesh_col, 'coloured mesh (the colour key)'),
                          (fig.add_subplot(gs[0, 2]), ours_bp, 'OURS — back-projection (soft-snap, T=0.01)'),
                          (fig.add_subplot(gs[0, 3]), shic_bp, 'SHIC — back-projection (argmax)')):
        ax.imshow(im); ax.set_title(title, fontsize=10); ax.axis('off')

    ax_mesh = fig.add_subplot(gs[1, 0:4])
    ax_mesh.imshow(mesh); ax_mesh.axis('off')
    ax_mesh.set_title('mesh: GT label vs each method\'s predicted vertex', fontsize=10)

    # 2026-10-05 (Tal): the error table sits beside the top row; the marker legend sits beside the MESH row, because
    # that is the row it explains. Legend swatches are real circles (scatter markers), not axes-space patches, which
    # came out as ellipses on this non-square panel.
    ax_l = fig.add_subplot(gs[1, 4]); ax_l.axis('off'); ax_l.set_xlim(0, 1); ax_l.set_ylim(0, 1)
    yl = 0.80
    ax_l.text(0.0, yl, 'markers on the mesh', fontsize=10, weight='bold', va='center'); yl -= 0.10
    for col, lab, size in ((ROLE_GT, 'GT label', 260), (ROLE_OURS, 'OURS  (soft-snap)', 190), (ROLE_SHIC, 'SHIC  (argmax)', 190)):
        ax_l.scatter([0.05], [yl], s=size, color=[col], edgecolors='0.2', linewidths=0.6, clip_on=False)
        ax_l.text(0.13, yl, lab, fontsize=10, va='center'); yl -= 0.095
    ax_l.text(0.0, yl - 0.01, 'number = GT point\nfaded sphere = behind the mesh in that view', fontsize=8.5, va='top', color='0.25')

    ax_t = fig.add_subplot(gs[0, 4]); ax_t.axis('off')
    y = 0.98
    ax_t.text(0.0, y, f'ann {args.ann_id}', fontsize=13, weight='bold', va='top'); y -= 0.13
    ax_t.text(0.0, y, 'geodesic error  (228-normalized)', fontsize=10, weight='bold', va='top'); y -= 0.100
    hdr = f'{"pt":>3}  {"GT v":>6} {"ours":>6} {"SHIC":>6}   {"ours":>6} {"SHIC":>6}'
    ax_t.text(0.0, y, hdr, fontsize=9, family='monospace', va='top'); y -= 0.076
    for i in range(res['n_points']):
        eo, es = res['err_ours_soft'][i], res['err_shic'][i]
        line = (f'{i+1:>3}  {res["gt_vertex"][i]:>6} {res["ours_soft_vertex"][i]:>6} '
                f'{res["shic_vertex"][i]:>6}   {eo:>6.1f} {es:>6.1f}')
        ax_t.text(0.0, y, line, fontsize=9, family='monospace', va='top',
                  color=('tab:green' if eo < es else 'tab:red'))
        y -= 0.072
    mo, ms = res['mean_ours_soft'], res['mean_shic']
    ax_t.text(0.0, y, '-' * len(hdr), fontsize=9, family='monospace', va='top'); y -= 0.060
    # same font size and NOT bold-width-shifted, so the two means sit under their own columns (Tal, 2026-10-05)
    ax_t.text(0.0, y, f'{"mean error":<25}   {mo:>6.1f} {ms:>6.1f}', fontsize=9, family='monospace', weight='bold',
              va='top', color=('tab:green' if mo < ms else 'tab:red')); y -= 0.120
    ax_t.text(0.0, y, 'green = ours closer, red = SHIC closer', fontsize=8.5, va='top', color='0.25')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    fig.savefig(args.output_path, dpi=130, bbox_inches='tight', facecolor='white')
    print(f'saved {args.output_path}')


if __name__ == '__main__':
    main()
