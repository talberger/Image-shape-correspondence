"""Lay out N labelled images in a row / column / grid, for cross-arm comparison figures.

Why this exists: analysis/stack_images_labeled.py takes exactly TWO images (top/bottom). Comparing
more than two arms at once -- e.g. the three feature-RKD gradient paths against their control --
needed a manual PIL block every time, which is how the SHIC comparison grid ended up inlined in a
shell script. This is the reusable version.

--crop_right_frac is the piece worth knowing about: this project's back-projection outputs are
`<mesh render> | <back-projected photo>` concatenated at a common height, and in a multi-arm figure
the mesh render is IDENTICAL in every panel, so repeating it wastes most of the width. Passing the
photo's share of the width keeps only the half that actually differs between arms.
"""
import argparse, os
from PIL import Image, ImageDraw


def _panel_split_x(im, white=250):
    """x of the panel boundary: the midpoint of the widest all-white column run, ignoring the
    image's own left/right margins. Returns 0 if no interior gap is found, i.e. crop nothing."""
    px = im.load()
    w, h = im.size
    step = max(1, h // 200)                       # sampling rows is enough and much faster
    is_white = []
    for x in range(w):
        col_white = True
        for y in range(0, h, step):
            r, g, b = px[x, y]
            if r < white or g < white or b < white:
                col_white = False
                break
        is_white.append(col_white)
    best = (0, -1)                                # (width, start)
    run_start = None
    for x in range(w + 1):
        if x < w and is_white[x]:
            if run_start is None:
                run_start = x
        elif run_start is not None:
            if run_start > 0 and x < w:           # interior run only, not a page margin
                if x - run_start > best[0]:
                    best = (x - run_start, run_start)
            run_start = None
    return 0 if best[1] < 0 else best[1] + best[0] // 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--paths', required=True, help='image paths, joined by --sep')
    p.add_argument('--labels', required=True, help='labels, joined by --sep, same order/count')
    p.add_argument('--sep', default=',',
                   help="separator for --paths/--labels (default ','). Labels routinely contain "
                        "commas, which silently mis-splits them -- pass e.g. --sep '|' then.")
    p.add_argument('--title', default='')
    p.add_argument('--output_path', required=True)
    p.add_argument('--cols', type=int, default=0, help='0 = one row (default); N = wrap into N columns')
    p.add_argument('--panel_height', type=int, default=520)
    p.add_argument('--crop_right_frac', type=float, default=0.0,
                   help='keep only the rightmost F of each source image (0 = whole image). Use for '
                        'this project\'s "<mesh render> | <photo>" side-by-side outputs, where the '
                        'mesh render repeats identically across arms.')
    p.add_argument('--crop_right_auto', action='store_true',
                   help='instead of a hand-tuned --crop_right_frac, FIND the panel boundary: the '
                        'widest run of all-white columns, cropping from its midpoint. Built because '
                        'guessing the fraction cut INTO the mesh panel -- the two panels are '
                        'concatenated at their own widths, which depend on each render\'s aspect, '
                        'so there is no fixed fraction that works across figures.')
    p.add_argument('--band', type=int, default=34, help='label strip height, px')
    a = p.parse_args()

    paths = [s.strip() for s in a.paths.split(a.sep)]
    labels = [s.strip() for s in a.labels.split(a.sep)]
    if len(paths) != len(labels):
        raise SystemExit('--paths and --labels must have the same count '
                         f'({len(paths)} vs {len(labels)})')
    for q in paths:
        if not os.path.isfile(q):
            raise SystemExit(f'missing input: {q}')

    panels = []
    for q in paths:
        im = Image.open(q).convert('RGB')
        if a.crop_right_auto:
            im = im.crop((_panel_split_x(im), 0, im.width, im.height))
        elif a.crop_right_frac > 0:
            im = im.crop((int(im.width * (1.0 - a.crop_right_frac)), 0, im.width, im.height))
        scale = a.panel_height / im.height
        panels.append(im.resize((max(1, int(im.width * scale)), a.panel_height)))

    cols = a.cols if a.cols > 0 else len(panels)
    rows = (len(panels) + cols - 1) // cols
    cell_w = max(p_.width for p_ in panels)
    title_h = 40 if a.title else 0
    canvas = Image.new('RGB', (cell_w * cols, title_h + rows * (a.band + a.panel_height)), 'white')
    d = ImageDraw.Draw(canvas)
    if a.title:
        d.text((12, 14), a.title, fill=(0, 0, 0))
    for i, (im, lab) in enumerate(zip(panels, labels)):
        r, c = divmod(i, cols)
        x = c * cell_w
        y = title_h + r * (a.band + a.panel_height)
        d.text((x + 8, y + 10), lab, fill=(0, 0, 0))
        canvas.paste(im, (x + (cell_w - im.width) // 2, y + a.band))
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    canvas.save(a.output_path)
    print(f'saved {len(panels)}-panel grid ({rows}x{cols}) to {a.output_path}')


if __name__ == '__main__':
    main()
