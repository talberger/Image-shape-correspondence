"""
Contact-sheet grid: tile the per-step rendered-view panel (cropped out of each step's
color_mesh_from_knn.py + render_colored_mesh.py side-by-side PNG) into one grid image,
labeled by step number, so a dense step-by-step sweep can be scanned at a glance instead
of opening each render individually.

Run:
cd $DISC_ROOT && python analysis/build_step_grid.py \
  --side_by_side_glob 'demo/guitar/bsb_mapping_loss_dense50/step{step}/color_mesh_from_knn_k10/render/guitar_01_click_130_140_sam_mask_2_mesh_color_position_k10_pix_random7_seed152_side_by_side.png' \
  --steps 1-50 --cols 10 \
  --output_path demo/guitar/bsb_mapping_loss_dense50/step_progression_grid.png
"""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont


def parse_steps(steps_str):
    steps = []
    for part in steps_str.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            steps.extend(range(int(lo), int(hi) + 1))
        else:
            steps.append(int(part))
    return steps


def crop_rendered_view(side_by_side_path, source_panel_width):
    """side_by_side.png is [source photo | rendered view(s)] pasted left-to-right at
    native heights (render_colored_mesh.py:220-236) -- with exactly one rendered view
    (our --azim_deg 90 single-view setup) the view panel is everything right of the
    known, fixed source-photo width."""
    img = Image.open(side_by_side_path).convert('RGB')
    return img.crop((source_panel_width, 0, img.width, img.height))


def main(args):
    steps = parse_steps(args.steps)
    cell_h = args.cell_size
    panels = []

    if args.reference_image:
        # The source photo with the 7 click markers -- the left panel of any
        # step's side_by_side.png (all identical: same seed, same mask), shown
        # once so the grid reads against the real object, not just the mesh.
        ref = Image.open(args.reference_image).convert('RGB')
        scale = cell_h / ref.height
        ref = ref.resize((max(1, int(ref.width * scale)), cell_h))
        panels.append((args.reference_label, ref))

    for step in steps:
        path = args.side_by_side_glob.format(step=step)
        if not os.path.exists(path):
            raise FileNotFoundError(f'missing render for step {step}: {path}')
        panel = crop_rendered_view(path, args.source_panel_width)
        scale = cell_h / panel.height
        panel = panel.resize((max(1, int(panel.width * scale)), cell_h))
        panels.append((f'step {step}', panel))

    cell_w = max(p.width for _, p in panels)
    cols = args.cols
    rows = (len(panels) + cols - 1) // cols
    label_h = 28

    grid = Image.new('RGB', (cols * cell_w, rows * (cell_h + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
    except OSError:
        font = ImageFont.load_default()

    for i, (label, panel) in enumerate(panels):
        r, c = divmod(i, cols)
        x = c * cell_w + (cell_w - panel.width) // 2
        y = r * (cell_h + label_h) + label_h
        draw.text((c * cell_w + 4, r * (cell_h + label_h) + 4), label, fill=(0, 0, 0), font=font)
        grid.paste(panel, (x, y))

    grid.save(args.output_path)
    print(f'saved {len(panels)}-panel grid ({cols}x{rows}) to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--side_by_side_glob', type=str, required=True,
                         help='path template with a {step} placeholder')
    parser.add_argument('--steps', type=str, required=True,
                         help='e.g. "1-50" or "1,5,10-20"')
    parser.add_argument('--source_panel_width', type=int, default=810,
                         help='width of the source-photo panel in each side_by_side.png, '
                              'to know where the rendered-view panel starts')
    parser.add_argument('--cols', type=int, default=10)
    parser.add_argument('--cell_size', type=int, default=220,
                         help='height each rendered-view panel is scaled to before tiling')
    parser.add_argument('--reference_image', type=str, default=None,
                         help='optional source-photo-with-click-markers image (any step\'s '
                              '..._clicked_pixels.png -- identical across steps, same seed/mask), '
                              'prepended as the first panel so the grid reads against the real object')
    parser.add_argument('--reference_label', type=str, default='source (7px)')
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    main(args)
