"""
Render the 100 hand-labeled GROUND-TRUTH vertices on the mesh, numbered, and put that
side by side with the numbered pixel labels on the photo -- so the same label number can be
checked in both domains at a glance (Tal's request, 2026-08-26).

Front-view orthographic projection (screen x = mesh x, screen y = mesh y): guitar.obj is
essentially flat in z (-0.062..0.059) and long in y (-1.0..0.804), so the front view is the
one that matches the photo's framing.

NOTE, verified 2026-08-26: do NOT classify front/back by the sign of z. The headstock is
angled back from the body plane exactly as on a real guitar -- every one of its 1106 vertices
sits at z < 0 (-0.0617..-0.0087) -- so a z >= 0 test wrongly calls the headstock labels
(0-5, 9) "back-facing". All 100 labels are on the front surface.

Run:
python analysis/render_labeled_mesh_numbers.py \
  --save_path demo/guitar/eval_labels/labeled_mesh_numbered.png \
  --side_by_side_path demo/guitar/eval_labels/labels_image_vs_mesh.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils import loadmesh


def main(args):
    mesh = loadmesh(dir=args.obj_path, name=args.name)
    V = mesh.vertices.cpu().numpy()
    F = mesh.faces.cpu().numpy()
    pairs = torch.load(args.eval_pairs_path)
    label_id = pairs['label_id'].numpy()
    gt = pairs['vertex_idx'].numpy().astype(int)

    tri = V[F]                                     # [F, 3, 3]
    order = np.argsort(tri[:, :, 2].mean(axis=1))  # painter's algorithm, back to front

    fig, ax = plt.subplots(figsize=(args.width_in, args.height_in), dpi=args.dpi)
    ax.add_collection(PolyCollection(tri[order][:, :, :2], facecolors='#e2e6ea',
                                     edgecolors='#b9c0c7', linewidths=0.18, zorder=1))

    ax.scatter(V[gt][:, 0], V[gt][:, 1], s=args.marker_size, c='tab:red', zorder=3,
               linewidths=0.6, edgecolors='white')

    for i in range(len(gt)):
        ax.annotate(str(int(label_id[i])), (V[gt][i, 0], V[gt][i, 1]),
                    textcoords='offset points', xytext=(5.5, 4.0), fontsize=args.fontsize,
                    color='#0b2545', zorder=4,
                    weight='bold',
                    path_effects=[pe.withStroke(linewidth=3.0, foreground='white')])

    pad = 0.03
    ax.set_xlim(V[:, 0].min() - pad, V[:, 0].max() + pad)
    ax.set_ylim(V[:, 1].min() - pad, V[:, 1].max() + pad)
    ax.set_aspect('equal'); ax.axis('off')
    ax.invert_yaxis() if args.invert_y else None
    ax.set_title(f'mesh: {len(gt)} ground-truth vertices (front view)', fontsize=16)
    fig.tight_layout(pad=0.2)
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    fig.savefig(args.save_path, bbox_inches='tight', facecolor='white')
    print(f'saved numbered mesh render to {args.save_path}')

    if args.side_by_side_path:
        from PIL import Image
        img = Image.open(args.pixel_labels_png).convert('RGB')
        msh = Image.open(args.save_path).convert('RGB')
        h = max(img.height, msh.height)
        img = img.resize((round(img.width * h / img.height), h), Image.LANCZOS)
        msh = msh.resize((round(msh.width * h / msh.height), h), Image.LANCZOS)
        gap = 16
        out = Image.new('RGB', (img.width + gap + msh.width, h), 'white')
        out.paste(img, (0, 0)); out.paste(msh, (img.width + gap, 0))
        out.save(args.side_by_side_path)
        print(f'saved side-by-side to {args.side_by_side_path}  ({out.size[0]}x{out.size[1]})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    ap.add_argument('--name', type=str, default='guitar')
    ap.add_argument('--eval_pairs_path', type=str,
                    default='./demo/guitar/eval_labels/pixel_vertex_eval_pairs.pth')
    ap.add_argument('--pixel_labels_png', type=str,
                    default='./demo/guitar/eval_labels/pixel_labels_numbered.png')
    ap.add_argument('--save_path', type=str,
                    default='./demo/guitar/eval_labels/labeled_mesh_numbered.png')
    ap.add_argument('--side_by_side_path', type=str,
                    default='./demo/guitar/eval_labels/labels_image_vs_mesh.png')
    ap.add_argument('--fontsize', type=float, default=15.0)
    ap.add_argument('--marker_size', type=float, default=34)
    ap.add_argument('--width_in', type=float, default=7.6)
    ap.add_argument('--height_in', type=float, default=19.0)
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--invert_y', action='store_true')
    main(ap.parse_args())
