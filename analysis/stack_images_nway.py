"""
Stack N images vertically with the same title / label / source bands the 2-way stacker uses.

stack_images_labeled.py is hardwired to exactly two panels (top/bottom), which was enough while
every comparison in this project was a head-to-head. The fuse 2x2 closed with THREE arms that only
make sense read together (2026-09-08: the three ZS arms span 2.7%-6.3% mean error), so this takes
an arbitrary number of --panel PATH LABEL pairs instead.

Panel header colours cycle through the 2-way tool's red/green plus a blue, so a 2-panel call here
looks the same as the old tool's output.

Run:
python analysis/stack_images_nway.py \
  --panel <A>/analyze/back_projection/..._side_by_side.png "sdsquare_fuse16 (ZS)" \
  --panel <B>/analyze/back_projection/..._side_by_side.png "sdsquare_fuse32 (ZS)" \
  --panel <C>/analyze/back_projection/..._side_by_side.png "sdsquash_fuse16 (ZS)" \
  --title "back-projection soft-snap k=1, read T=1 -- three ZS arms" \
  --output_path <out>/1_back_projection_soft_readT1.png
"""
import argparse
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.make_run_vs_baseline_comparison import (
    HEADER_H, SOURCE_H, TITLE_H, is_reference_label, label_band, shared_tokens, source_band)

PANEL_COLORS = [(120, 30, 30), (20, 90, 20), (30, 40, 120), (110, 70, 10), (80, 25, 90)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', nargs=2, action='append', metavar=('PATH', 'LABEL'), required=True,
                   help='repeat once per panel, in the order they should be stacked')
    p.add_argument('--title', required=True)
    p.add_argument('--output_path', required=True)
    args = p.parse_args()

    missing = [path for path, _ in args.panel if not os.path.exists(path)]
    if missing:
        raise SystemExit('missing input image(s):\n  ' + '\n  '.join(missing))

    images = [Image.open(path).convert('RGB') for path, _ in args.panel]
    width = max(im.width for im in images)

    def pad(im):
        if im.width == width:
            return im
        canvas = Image.new('RGB', (width, im.height), (255, 255, 255))
        canvas.paste(im, ((width - im.width) // 2, 0))
        return canvas

    images = [pad(im) for im in images]

    title = label_band(width, args.title, font_size=26, bg=(10, 10, 60)).resize((width, TITLE_H))
    total_h = TITLE_H + sum(HEADER_H + im.height + SOURCE_H for im in images)
    canvas = Image.new('RGB', (width, total_h), (255, 255, 255))

    # 2026-09-14 (Tal): dim what every row's label has in common, highlight what differs
    common = shared_tokens([label for _, label in args.panel])

    y = 0
    canvas.paste(title, (0, y)); y += TITLE_H
    for i, (im, (path, label)) in enumerate(zip(images, args.panel)):
        hl = None if is_reference_label(label) else common
        canvas.paste(label_band(width, label, bg=PANEL_COLORS[i % len(PANEL_COLORS)], shared=hl), (0, y))
        y += HEADER_H
        canvas.paste(im, (0, y)); y += im.height
        canvas.paste(source_band(width, path), (0, y)); y += SOURCE_H

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    canvas.save(args.output_path)
    print(f'saved {args.output_path}  ({width}x{total_h}, {len(images)} panels)')


if __name__ == '__main__':
    main()
