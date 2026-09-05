"""One-off comparison grid: DFD phase-1 vs axisfix phase-1 (current best phase-1), PCA + sim_heatmap
side by side. Ad hoc per Tal's 2026-08-22 request, not a general-purpose analysis tool -- paths are
hardcoded to the two specific ZS runs being compared, doesn't touch any existing analysis script.

Mesh-domain (3D) panels only -- Tal's follow-up request: axisfix and DFD differ only in how the
vertex/3D encoder was trained, the 2D real-photo pixel features (dino_sd_poc) are identical inputs
to both, so the pixel-domain panels in each source image add no comparison signal. Crops both source
images down to their mesh-render half instead of re-running either analysis script with different
args, since the underlying PCA/sim-heatmap data is unchanged -- only which half of the existing
images gets shown. PCA side-by-side is [source photo (810px wide) | rendered mesh view (1080px
wide)] at a fixed 1890x1080 -- confirmed by inspecting images/guitar_01.png's own size (810x1080)
before writing the crop -- so mesh panel = x in [810, 1890]. Sim-heatmap grid is
[4 pixel-domain panels | 4 mesh-domain panels] side by side at a fixed 1600x900 (2 rows x 4 cols,
confirmed by viewing the actual image, not just the docstring's top/bottom description, which
doesn't match feature_similarity_heatmap.py's real column-block layout) -- mesh block = right half,
x in [800, 1600].
"""
import os

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

AXISFIX_DIR = 'demo/guitar/runs/legacy/bsb_sd_dino_l2norm_frontcentered_axisfix_ZS/features_analyze'
DFD_DIR = 'demo/guitar/runs/legacy/bsb_sd_dino_dfd_frontcentered_ZS/features_analyze'
OUT_PATH = 'demo/guitar/dfd_vs_axisfix_phase1_comparison_3donly.png'

PCA_MESH_CROP = (810, 0, 1890, 1080)
HEATMAP_MESH_CROP = (800, 0, 1600, 900)

PAD = 20
LABEL_H = 60
ROW_LABEL_W = 220
BG = (255, 255, 255)
FG = (20, 20, 20)


def load(path, target_w=None, crop=None):
    im = Image.open(path).convert('RGB')
    if crop is not None:
        im = im.crop(crop)
    if target_w is not None and im.width != target_w:
        h = int(im.height * target_w / im.width)
        im = im.resize((target_w, h), Image.LANCZOS)
    return im


def main():
    pca_axisfix = load(os.path.join(AXISFIX_DIR, 'pca/guitar_mesh_pca_shared_color_side_by_side.png'),
                        crop=PCA_MESH_CROP)
    pca_dfd = load(os.path.join(DFD_DIR, 'pca/guitar_mesh_pca_shared_color_side_by_side.png'),
                    crop=PCA_MESH_CROP)
    heat_axisfix = load(os.path.join(AXISFIX_DIR, 'sim_heatmap/simheatmap_grid.png'),
                         target_w=pca_axisfix.width, crop=HEATMAP_MESH_CROP)
    heat_dfd = load(os.path.join(DFD_DIR, 'sim_heatmap/simheatmap_grid.png'),
                     target_w=pca_dfd.width, crop=HEATMAP_MESH_CROP)

    col_w = max(pca_axisfix.width, pca_dfd.width)
    row1_h = max(pca_axisfix.height, pca_dfd.height)
    row2_h = max(heat_axisfix.height, heat_dfd.height)

    n_cols = 2
    total_w = ROW_LABEL_W + n_cols * col_w + (n_cols + 1) * PAD
    total_h = LABEL_H + row1_h + row2_h + 4 * PAD

    canvas = Image.new('RGB', (total_w, total_h), BG)
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(FONT_PATH, 28)
    label_font = ImageFont.truetype(FONT_PATH, 22)

    col_headers = ['axisfix (current best) -- phase 1, 3D/mesh only', 'DFD -- phase 1, 3D/mesh only']
    for c, header in enumerate(col_headers):
        x = ROW_LABEL_W + PAD + c * (col_w + PAD)
        draw.text((x, PAD), header, fill=FG, font=title_font)

    row_headers = ['PCA\n(shared color)', 'Sim-heatmap\n(vertex query)']
    y0 = LABEL_H
    for r, (header, h) in enumerate(zip(row_headers, [row1_h, row2_h])):
        draw.text((PAD, y0 + h // 2 - 20), header, fill=FG, font=label_font)
        y0 += h + PAD

    y = LABEL_H
    canvas.paste(pca_axisfix, (ROW_LABEL_W + PAD, y))
    canvas.paste(pca_dfd, (ROW_LABEL_W + PAD * 2 + col_w, y))
    y += row1_h + PAD
    canvas.paste(heat_axisfix, (ROW_LABEL_W + PAD, y))
    canvas.paste(heat_dfd, (ROW_LABEL_W + PAD * 2 + col_w, y))

    canvas.save(OUT_PATH)
    print(f'Saved {OUT_PATH} ({canvas.size})')


if __name__ == '__main__':
    main()
