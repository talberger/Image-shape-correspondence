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
cd /workspace/bsb && python analysis/make_lvis_summary_figure.py --ann_id 117683 \
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


def right_half(path):
    """The back-projection tools save a 'mesh render | back-projected photo' side-by-side; the
    photo is the right panel and the mesh render is redundant here (row 2 shows the mesh)."""
    im = Image.open(path).convert('RGB')
    w, h = im.size
    return im.crop((w // 2, 0, w, h))


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
    ours_bp = right_half(os.path.join(base, f'backproj_{tag}', f'{tag}_OURS_soft_backproj_side_by_side.png'))
    shic_bp = right_half(os.path.join(base, f'backproj_{tag}', 'shic', f'{tag}_SHIC_argmax_backproj_side_by_side.png'))
    mesh = Image.open(os.path.join(base, f'{tag}_mesh_views.png')).convert('RGB')

    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], width_ratios=[1, 1, 1, 0.95],
                          hspace=0.16, wspace=0.06)

    for ax, im, title in ((fig.add_subplot(gs[0, 0]), photo, f'original crop + GT points  ({tag})'),
                          (fig.add_subplot(gs[0, 1]), ours_bp, 'OURS — back-projection (soft-snap, T=0.01)'),
                          (fig.add_subplot(gs[0, 2]), shic_bp, 'SHIC — back-projection (argmax)')):
        ax.imshow(im); ax.set_title(title, fontsize=10); ax.axis('off')

    ax_mesh = fig.add_subplot(gs[1, 0:3])
    ax_mesh.imshow(mesh); ax_mesh.axis('off')
    ax_mesh.set_title('mesh: GT label vs each method\'s predicted vertex', fontsize=10)

    ax_t = fig.add_subplot(gs[:, 3]); ax_t.axis('off')
    y = 0.98
    ax_t.text(0.0, y, f'ann {args.ann_id}', fontsize=13, weight='bold', va='top'); y -= 0.06
    ax_t.text(0.0, y, 'markers on the mesh', fontsize=10, weight='bold', va='top'); y -= 0.045
    for col, lab in ((ROLE_GT, 'GT label  (largest sphere)'), (ROLE_OURS, 'OURS  (soft-snap)'),
                     (ROLE_SHIC, 'SHIC  (argmax)')):
        ax_t.add_patch(Circle((0.035, y - 0.012), 0.022, color=col, transform=ax_t.transAxes, clip_on=False))
        ax_t.text(0.10, y, lab, fontsize=10, va='top'); y -= 0.05
    y -= 0.02

    ax_t.text(0.0, y, 'geodesic error  (228-normalized)', fontsize=10, weight='bold', va='top'); y -= 0.05
    hdr = f'{"pt":>3}  {"GT v":>6} {"ours":>6} {"SHIC":>6}   {"ours":>6} {"SHIC":>6}'
    ax_t.text(0.0, y, hdr, fontsize=9, family='monospace', va='top'); y -= 0.038
    for i in range(res['n_points']):
        eo, es = res['err_ours_soft'][i], res['err_shic'][i]
        line = (f'{i+1:>3}  {res["gt_vertex"][i]:>6} {res["ours_soft_vertex"][i]:>6} '
                f'{res["shic_vertex"][i]:>6}   {eo:>6.1f} {es:>6.1f}')
        ax_t.text(0.0, y, line, fontsize=9, family='monospace', va='top',
                  color=('tab:green' if eo < es else 'tab:red'))
        y -= 0.036
    y -= 0.015
    mo, ms, ma = res['mean_ours_soft'], res['mean_shic'], res['mean_ours_argmax']
    ax_t.text(0.0, y, f'{"MEAN":>3}  {"":>6} {"":>6} {"":>6}   {mo:>6.1f} {ms:>6.1f}',
              fontsize=10, family='monospace', weight='bold', va='top'); y -= 0.05
    ax_t.text(0.0, y, f'ours argmax (same rule as SHIC): {ma:.1f}', fontsize=9, va='top'); y -= 0.055

    ax_t.text(0.0, y, 'context', fontsize=10, weight='bold', va='top'); y -= 0.045
    if args.class_name == 'bear':
        ctx = ('SHIC Table 1, bear (132 inst):', '   SHIC 33.0 | CSE sup. 35.7', '   zero-shot SD-DINO 48.2',
               'reproduced here: 33.09')
    else:
        import json as _json
        r = _json.load(open(f'data/densepose_lvis/eval_shic_{args.class_name}/results.json'))['bbox']
        ctx = (f'SHIC, {args.class_name} ({len(r["per_instance"])} inst):', f'   reproduced here: {r["mean_point"]:.2f}')
    for line in ctx + ('',
                 'green = we are closer, red = SHIC is.',
                 'NOTE: 2-3 points on ONE instance --',
                 'a single instance is not',
                 'evidence either way.'):
        ax_t.text(0.0, y, line, fontsize=8.5, va='top', color='0.25')
        y -= 0.032

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    fig.savefig(args.output_path, dpi=130, bbox_inches='tight', facecolor='white')
    print(f'saved {args.output_path}')


if __name__ == '__main__':
    main()
