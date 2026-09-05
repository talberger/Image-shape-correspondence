"""PCA->RGB of DINO, SD and the FUSED map, for several framing variants, side by side.

One row per variant, columns: DINO / SD / FUSED (= what phase-2 actually consumes).

Everything is cropped to the VALID region first. That is not cosmetic: measured 2026-09-03, the
padded 16x16 view makes sd_square look far better aligned than squash (corr 0.81 vs 0.52), but the
entire gap is padding-cells agreeing with padding-cells -- on real content the two are identical
(0.78 vs 0.78). Comparing anything on a padded grid here produces a wrong answer.
"""
import argparse, os, sys
import torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.dino_sd_overlay_check import valid_cells
from analysis.dino_sd_pca_check import pca_rgb


def pca_gray(x):
    """[C, h, w] -> [h, w] in 0..1 from the FIRST principal component only.

    Added 2026-09-03 (Tal). The RGB view gives each map its own 3-component basis, so colours are
    not comparable between panels and only structure is. One component removes that ambiguity --
    what is left is a single scalar field per map, so panels can be read against each other
    directly. The sign is pinned by the same convention as pca_rgb (top-left quadrant negative),
    otherwise two identical maps can come out inverted.
    """
    c, h, w = x.shape
    flat = (x.reshape(c, -1) - x.reshape(c, -1).mean(1, keepdim=True)).T
    u, sv, v = torch.pca_lowrank(flat, q=1, center=False)
    comp = (flat @ v)[:, 0].reshape(h, w)
    if comp[:h // 2, :w // 2].mean() > 0:
        comp = -comp
    lo, hi = comp.flatten().quantile(0.02), comp.flatten().quantile(0.98)
    return ((comp - lo) / (hi - lo).clamp(min=1e-8)).clamp(0, 1)


def load_valid(path, sd_no_padding, sd_input_res=512):
    d = torch.load(path, map_location='cpu')
    dino, sd, fused = d['dino_raw'].float(), d['sd_raw'].float(), d['fused_raw'].float()
    in_h, in_w = d['input_size']; oh, ow = d['original_size']
    tgt = d.get('dino_target_size', max(in_h, in_w))
    dh, dw = valid_cells(dino.shape[-2:], (in_h, in_w), tgt)
    dino = dino[:, :dh, :dw]
    if not sd_no_padding:
        s = sd_input_res / max(oh, ow)
        sh, sw = valid_cells(sd.shape[-2:], (round(oh * s), round(ow * s)), sd_input_res)
        sd = sd[:, :sh, :sw]
    # The fused map lives on whichever grid its fuse_mode produced. common_grid fuses ALREADY
    # cropped inputs, so it is valid everywhere; dino_grid inherits DINO's padding and must be
    # cropped with DINO's own ratio.
    if fused.shape[-2:] == d['dino_raw'].shape[-2:]:
        fused = fused[:, :dh, :dw]
    return dino, sd, fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', required=True,
                    help='name:pregrid_path:nopad(0|1), comma-separated')
    ap.add_argument('--output_path', required=True)
    ap.add_argument('--resample_hw', type=int, nargs=2, default=None,
                    help='resample EVERY panel to this (h, w) before the PCA, so all variants are '
                         'shown at one resolution. 16 12 is DINO\'s valid grid, i.e. the '
                         'resolution fuse_dino_sd actually fuses at (2026-09-03, Tal).')
    ap.add_argument('--gray', action='store_true',
                    help='show the FIRST principal component as grayscale instead of 3 components '
                         'as RGB -- one scalar field per map, directly comparable between panels.')
    ap.add_argument('--scale', type=float, default=1.0, help='figure size multiplier')
    ap.add_argument('--title', default='DINO / SD / FUSED -- PCA->RGB, valid region only')
    args = ap.parse_args()

    rows = []
    for spec in args.variants.split(','):
        name, path, nopad = spec.split(':')
        maps = load_valid(path, nopad == '1')
        if args.resample_hw:
            hw = tuple(args.resample_hw)
            maps = tuple(F.interpolate(m[None], size=hw, mode='bilinear',
                                       align_corners=False)[0] for m in maps)
        rows.append((name, *maps))

    fig, axes = plt.subplots(len(rows), 3, figsize=(11 * args.scale, 4.3 * args.scale * len(rows)), squeeze=False)
    for r, (name, dino, sd, fused) in enumerate(rows):
        for c, (m, t) in enumerate(((dino, 'DINO'), (sd, 'SD'), (fused, 'FUSED (what phase-2 sees)'))):
            ax = axes[r][c]
            if args.gray:
                ax.imshow(pca_gray(m).numpy(), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            else:
                ax.imshow(pca_rgb(m).numpy(), interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(t, fontsize=12, fontweight='bold')
            if c == 0:
                ax.set_ylabel(name, fontsize=12, fontweight='bold', rotation=0,
                              ha='right', va='center', labelpad=16)
            ax.set_xlabel(f'{m.shape[-2]}x{m.shape[-1]}', fontsize=9)
    fig.suptitle(args.title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    fig.savefig(args.output_path, dpi=140, bbox_inches='tight', facecolor='white')
    print('saved', args.output_path)


if __name__ == '__main__':
    main()
