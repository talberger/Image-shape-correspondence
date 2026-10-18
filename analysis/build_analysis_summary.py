"""
One-glance summary grid for a phase-2 run's full_analyze output: 10 panels (2 rows x 5 cols),
each labeled by where it came from, built by cropping the shared "left" half out of every
side_by_side.png instead of repeating it -- e.g. color_mesh_from_knn_k1/k10/k100 and
color_mesh_from_soft_mapping all render the SAME source photo on the left (same POSCOLOR input,
verified byte-identical across all four for the SD-DINO L2norm run), so it's shown once,
followed by the four render-only panels. Same idea for back_projection's two outputs (both use
the same checker16 mesh render on the left) and hard-snap-spheres (kept as its own left+right
pair since its "left" -- the clicked-pixels photo -- is a different image, not identical to the
top row's plain position-colored source).

Tal's layout (2026-08-20), assuming every side_by_side.png this project's render_colored_mesh.py/
color_image_from_knn.py produces is [left panel | right panel] pasted at native heights
(see build_step_grid.py's crop_rendered_view for the same assumption, used there for the
source-photo/rendered-view split specifically):

  top row (5):    source | knn k=1 | knn k=10 | knn k=100 | soft mapping
  bottom row (5):  checker mesh | back-proj k=1 | back-proj soft-snap | hard-snap source | hard-snap render

Run (standard full_analyze layout, defaults match every *_full_analyze.sh in scripts/):
cd $DISC_ROOT && python analysis/build_analysis_summary.py \
  --analyze_dir demo/guitar/runs/legacy/bsb_sd_dino_l2norm_relational_geodesic_mu3dscale1_phase2/analyze \
  --output_path demo/guitar/runs/legacy/bsb_sd_dino_l2norm_relational_geodesic_mu3dscale1_phase2/analysis_summary.png
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont


def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert('RGB')


def split(img, split_x):
    """Same [left | right] side_by_side assumption as build_step_grid.py's
    crop_rendered_view -- panels pasted left-to-right at native heights, split_x is the
    known, fixed width of the left panel."""
    left = img.crop((0, 0, split_x, img.height))
    right = img.crop((split_x, 0, img.width, img.height))
    return left, right


def main(args):
    mb = args.mask_base
    pt = args.pixtag
    a = args.analyze_dir

    knn_k1 = load(f'{a}/color_mesh_from_knn_k1/render/{mb}_mesh_color_position_k1_side_by_side.png')
    knn_k10 = load(f'{a}/color_mesh_from_knn_k10/render/{mb}_mesh_color_position_k10_side_by_side.png')
    knn_k100 = load(f'{a}/color_mesh_from_knn_k100/render/{mb}_mesh_color_position_k100_side_by_side.png')
    sd = args.soft_dir_name
    soft = load(f'{a}/{sd}/render/{mb}_mesh_color_soft_snap_position_k10_side_by_side.png')
    bp_k1 = load(args.bp_k1_path or f'{a}/back_projection/{mb}_backproj_{args.checker_stem}_k1_side_by_side.png')
    bp_soft = load(args.bp_soft_path or f'{a}/back_projection/{mb}_backproj_{args.checker_stem}_soft_snap_k1_side_by_side.png')
    hard_snap = load(f'{a}/{sd}/render/{mb}_hard_snap_spheres_pix_{pt}_side_by_side.png')

    source, knn_k1_r = split(knn_k1, args.knn_split_x)
    _, knn_k10_r = split(knn_k10, args.knn_split_x)
    _, knn_k100_r = split(knn_k100, args.knn_split_x)
    _, soft_r = split(soft, args.knn_split_x)

    checker_mesh, bp_k1_r = split(bp_k1, args.backproj_split_x)
    _, bp_soft_r = split(bp_soft, args.backproj_split_x)

    hard_snap_l, hard_snap_r = split(hard_snap, args.hard_snap_split_x)

    top_row = [('source', source), ('knn k=1', knn_k1_r), ('knn k=10', knn_k10_r),
               ('knn k=100', knn_k100_r), ('soft mapping', soft_r)]
    bottom_row = [('checker mesh', checker_mesh), ('back-proj k=1', bp_k1_r),
                  ('back-proj soft-snap', bp_soft_r), ('hard-snap source', hard_snap_l),
                  ('hard-snap render', hard_snap_r)]

    cell_h = args.cell_size
    label_h = 28
    cols = 5
    resized = []
    for row in (top_row, bottom_row):
        resized_row = []
        for label, panel in row:
            scale = cell_h / panel.height
            panel = panel.resize((max(1, int(panel.width * scale)), cell_h))
            resized_row.append((label, panel))
        resized.append(resized_row)

    cell_w = max(p.width for row in resized for _, p in row)
    grid = Image.new('RGB', (cols * cell_w, 2 * (cell_h + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
    except OSError:
        font = ImageFont.load_default()

    for r, row in enumerate(resized):
        for c, (label, panel) in enumerate(row):
            x = c * cell_w + (cell_w - panel.width) // 2
            y = r * (cell_h + label_h) + label_h
            draw.text((c * cell_w + 4, r * (cell_h + label_h) + 4), label, fill=(0, 0, 0), font=font)
            grid.paste(panel, (x, y))

    grid.save(args.output_path)
    print(f'saved 10-panel summary (5x2) to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze_dir', type=str, required=True,
                         help='a run\'s .../analyze directory, as produced by any *_full_analyze.sh')
    parser.add_argument('--mask_base', type=str, default='guitar_01_click_270_370_sam_mask_2')
    parser.add_argument('--pixtag', type=str, default='explicit9')
    parser.add_argument('--checker_stem', type=str, default='guitar_mesh_checker16_color',
                        help='basename (no .pth) of the vertex-colour file the back-projection was '
                             'built from -- color_image_from_knn.py names its outputs after THAT, '
                             'not after the mesh, so a different mesh\'s checker lands under a '
                             'different filename (2026-09-02). Default is the original mesh\'s.')
    parser.add_argument('--knn_split_x', type=int, default=810,
                         help='left-panel (source photo) width in the knn_k1/k10/k100/'
                              'soft_mapping side_by_side.png files')
    parser.add_argument('--backproj_split_x', type=int, default=1080,
                         help='left-panel (checker16 mesh render) width in the back_projection '
                              'side_by_side.png files')
    parser.add_argument('--hard_snap_split_x', type=int, default=810,
                         help='left-panel (clicked-pixels photo) width in the hard-snap-spheres '
                              'side_by_side.png')
    parser.add_argument('--cell_size', type=int, default=320,
                         help='height each panel is scaled to before tiling')
    # SHIC-bear layout (2026-09-17): soft mapping lives in analyze/soft_mapping, back-projections are
    # named bear3_OURS_{top1,soft}_backproj_... Defaults keep the guitar layout unchanged.
    parser.add_argument('--soft_dir_name', type=str, default='color_mesh_from_soft_mapping')
    parser.add_argument('--bp_k1_path', type=str, default=None)
    parser.add_argument('--bp_soft_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    main(args)
