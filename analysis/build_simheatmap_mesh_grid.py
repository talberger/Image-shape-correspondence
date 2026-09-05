"""Mesh-side-only comparison of similarity heatmaps across runs (2026-09-03, Tal's request).

`analysis/feature_similarity_heatmap.py` writes one `simheatmap_grid.png` per run: a 4x2 figure
whose LEFT half is the 2D (pixel-query -> image) side and whose RIGHT half is the 3D
(vertex-query -> mesh) side. When the question is about mesh-side distinctiveness the 2D half is
noise, and comparing runs means opening N files side by side.

This script crops the mesh half out of each run's grid and lays the panels out as one figure:
one ROW per run, one COLUMN per query vertex. Panels are trimmed to their content bounding box
first, so the small mesh renders fill the cell instead of sitting in whitespace.

It reads the ALREADY-RENDERED grid PNGs -- no mesh, no GPU, no re-render, and therefore no risk
of the crop disagreeing with what the run actually produced.

Usage:
  python analysis/build_simheatmap_mesh_grid.py \
    --grids <run1>/simheatmap_grid.png,<run2>/... --labels "T=1,T=0.5" \
    --output_path demo/guitar/.../simheatmap_mesh_side.png
"""
import argparse, os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# feature_similarity_heatmap.py's layout: 4 columns x 2 rows, the 2 rightmost columns being the
# mesh side. Hardcoded because it is that script's own fixed layout, not a property of the image.
GRID_COLS, GRID_ROWS = 4, 2
MESH_COL_START = 2


def trim(im, bg_tol=250):
    """Crop to the non-background bounding box (panels are mostly white margin)."""
    rgb = im.convert('RGB')
    mask = rgb.point(lambda v: 255 if v < bg_tol else 0).convert('L')
    box = mask.getbbox()
    return im.crop(box) if box else im


def mesh_panels(path):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    cw, ch = w // GRID_COLS, h // GRID_ROWS
    out = []
    for row in range(GRID_ROWS):
        for col in range(MESH_COL_START, GRID_COLS):
            out.append(trim(im.crop((col * cw, row * ch, (col + 1) * cw, (row + 1) * ch))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', required=True, help='comma-separated simheatmap_grid.png paths')
    ap.add_argument('--labels', required=True, help='comma-separated row labels, one per grid')
    ap.add_argument('--titles', default='vertex_idx535,vertex_idx549,vertex_idx1017,vertex_idx2885',
                    help='comma-separated column titles, in panel order')
    ap.add_argument('--title', default='similarity heatmap -- mesh side only')
    ap.add_argument('--output_path', required=True)
    args = ap.parse_args()

    grids = [g.strip() for g in args.grids.split(',') if g.strip()]
    labels = [l.strip() for l in args.labels.split(',')]
    titles = [t.strip() for t in args.titles.split(',')]
    if len(labels) != len(grids):
        raise ValueError(f'{len(grids)} grids but {len(labels)} labels')
    for g in grids:
        if not os.path.isfile(g):
            raise FileNotFoundError(g)

    rows = [mesh_panels(g) for g in grids]
    ncols = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.0 * ncols, 3.4 * len(rows)),
                             squeeze=False)
    for r, (panels, label) in enumerate(zip(rows, labels)):
        for c in range(ncols):
            ax = axes[r][c]
            ax.axis('off')
            if c < len(panels):
                ax.imshow(panels[c])
            if r == 0 and c < len(titles):
                ax.set_title(titles[c], fontsize=11)
        axes[r][0].axis('on')
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for side in ('top', 'right', 'bottom', 'left'):
            axes[r][0].spines[side].set_visible(False)
        axes[r][0].set_ylabel(label, fontsize=13, fontweight='bold', rotation=0,
                              ha='right', va='center', labelpad=14)
    fig.suptitle(args.title, fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    fig.savefig(args.output_path, dpi=130, bbox_inches='tight')
    print(f'saved {args.output_path}  ({len(rows)} rows x {ncols} cols)')


if __name__ == '__main__':
    main()
