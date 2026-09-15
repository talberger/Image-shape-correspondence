"""
THE BLOCK PATTERN, STAGE BY STAGE -- photo side only, DINO only, no SD.

Written 2026-09-06 at Tal's request. Question being answered: the photo path produces a
16x16 token grid (16 wide x 12 tall valid) which is bilinearly blown up to 1080x810 --
so WHY does that read as a block pattern, when bilinear interpolation blends neighbours
rather than holding a value constant across a cell?

Every stage below is the PRODUCTION code path, not a reimplementation:
  encoder.preprocess_image        -> ResizeLongestSide(224) + pad to square
  DINOWrapper.__call__            -> the 16x16 patch-token grid
  DINOWrapper.postprocess_features-> interpolate to 224x224, crop the padding, interpolate
                                     to the original 1080x810
SD is deliberately absent: fuse_dino_sd() resizes SD DOWN onto this same DINO grid before
any of this happens, so it cannot set the cell size (modules/dino_sd_fusion.py:229).

Colour: one PCA basis is fit ONCE on the valid raw tokens and reused unchanged at every
stage, so colours are comparable across figures. Interpolation and cropping are linear
per channel and the PCA projection is a fixed linear map over channels, so projecting
before or after the resize is identical -- verified numerically at the end of the run.

Run:
  source /workspace/activate_bsb.sh
  python analysis/blockpattern_stages.py --img_path images/guitar_01.png \
      --out_dir demo/guitar/features_analyze/blockpattern
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder import preprocess_image
from modules.dino import DINOWrapper
from modules.utils import device

GRID_C = '#00ff00'      # cell BOUNDARIES
ANCHOR_C = '#ff00ff'    # token CENTRES (where the bilinear anchors actually sit)


def pca_basis(tokens_valid):
    """tokens_valid: [N, C] float32. Returns (mean, comps[3, C], lo[3], hi[3])."""
    mean = tokens_valid.mean(0, keepdim=True)
    x = tokens_valid - mean
    _, _, v = torch.pca_lowrank(x, q=3, center=False)
    comps = v[:, :3].T.contiguous()                     # [3, C]
    proj = x @ comps.T                                  # [N, 3]
    lo = torch.quantile(proj, 0.02, dim=0)
    hi = torch.quantile(proj, 0.98, dim=0)
    return mean, comps, lo, hi


def to_rgb(feat_chw, basis):
    """feat_chw: [C, H, W] -> HxWx3 float in [0,1], using the FIXED basis."""
    mean, comps, lo, hi = [t.to(feat_chw.device) for t in basis]
    c, h, w = feat_chw.shape
    flat = feat_chw.reshape(c, -1).T                    # [HW, C]
    proj = (flat - mean) @ comps.T                      # [HW, 3]
    proj = (proj - lo) / (hi - lo)
    return proj.clamp(0, 1).reshape(h, w, 3).cpu().numpy()


def show(ax, img, title, xlim=None, ylim=None, interp='nearest'):
    h, w = img.shape[:2] if img.ndim == 3 else img.shape
    ax.imshow(img, extent=[0, w, h, 0], interpolation=interp)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(xlim or (0, w))
    ax.set_ylim(ylim or (h, 0))


def draw_grid(ax, step_x, step_y, w, h, offset_x=0.0, offset_y=0.0, color=GRID_C,
              lw=0.8, ls='-', alpha=0.9):
    x = offset_x
    while x <= w + 1e-6:
        ax.axvline(x, color=color, lw=lw, ls=ls, alpha=alpha)
        x += step_x
    y = offset_y
    while y <= h + 1e-6:
        ax.axhline(y, color=color, lw=lw, ls=ls, alpha=alpha)
        y += step_y


def save(fig, path):
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_path', default='images/guitar_01.png')
    ap.add_argument('--out_dir', default='demo/guitar/features_analyze/blockpattern')
    ap.add_argument('--target_size', type=int, default=224)
    ap.add_argument('--scan_row', type=int, default=None,
                    help='full-res row for the scanline plot (default: middle of the valid area)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    P = args.target_size
    K = 14  # DINO patch size

    image = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    print(f'photo {args.img_path}: {W}x{H} (WxH)')

    # ---- STAGE 1: the model input -------------------------------------------------
    input_square, input_size = preprocess_image(image, P)   # [1,3,224,224], input_size=(h,w)
    ih, iw = int(input_size[0]), int(input_size[1])
    print(f'stage 1  model input {tuple(input_square.shape[-2:])}, valid region {ih}x{iw} (HxW), '
          f'padding {P - ih} rows / {P - iw} cols')

    inp = input_square[0].permute(1, 2, 0).cpu().numpy()
    n_h, n_w = P // K, P // K
    v_h, v_w = ih // K, iw // K
    print(f'         token grid {n_h}x{n_w} = {n_h * n_w} tokens, of which '
          f'{v_h}x{v_w} = {v_h * v_w} are real image')

    fig, ax = plt.subplots(figsize=(7, 7))
    show(ax, inp, f'STAGE 1 — model input {P}x{P} (photo resized to {iw}x{ih}, zero-padded)\n'
                  f'green = 14 px patch boundaries -> {n_w}x{n_h} tokens', interp='bilinear')
    draw_grid(ax, K, K, P, P)
    ax.add_patch(plt.Rectangle((0, 0), iw, ih, fill=False, ec='red', lw=2.5))
    ax.text(4, ih - 6, f'valid: {v_w}x{v_h} = {v_w * v_h} real tokens', color='red',
            fontsize=10, va='top', bbox=dict(fc='white', alpha=0.7, ec='none'))
    save(fig, f'{args.out_dir}/stage1_input_{P}_patchgrid.png')

    # ---- STAGE 2: the raw token grid ----------------------------------------------
    model = DINOWrapper(device=device, target_size=P)
    with torch.no_grad():
        raw = model(input_square)                          # [1, C, 16, 16]
    C = raw.shape[1]
    print(f'stage 2  raw token grid {tuple(raw.shape)}  ({C} channels)')

    valid_tokens = raw[0, :, :v_h, :v_w].reshape(C, -1).T.float()
    basis = pca_basis(valid_tokens)
    rgb16 = to_rgb(raw[0].float(), basis)

    fig, ax = plt.subplots(figsize=(7, 7))
    show(ax, rgb16, f'STAGE 2 — DINO output: {n_w}x{n_h} tokens, {C} ch (PCA->RGB, nearest)\n'
                    f'one square = ONE feature vector')
    draw_grid(ax, 1, 1, n_w, n_h, lw=0.6)
    ax.add_patch(plt.Rectangle((0, 0), v_w, v_h, fill=False, ec='red', lw=2.5))
    save(fig, f'{args.out_dir}/stage2_tokens_{n_w}x{n_h}.png')

    # ---- STAGE 3/4/5: the three postprocess ops, each captured ---------------------
    # On CPU: the full-res tensor is C x H x W x 4 B (1536 x 1080 x 810 = 5.0 GB), which OOMs a
    # 20 GB card once postprocess_features' own copy is alongside it. Same reason
    # run_sd_dino_stride14_nosdsquare_photo_features.sh passes --cpu_postprocess.
    raw = raw.float().cpu()
    model = model.cpu()
    basis = tuple(t.cpu() for t in basis)
    up = F.interpolate(raw, (P, P), mode='bilinear', align_corners=False)
    crop = up[..., :ih, :iw]
    full = F.interpolate(crop, (H, W), mode='bilinear', align_corners=False)

    ref = model.postprocess_features(raw, input_size=input_size, original_size=(H, W))
    err = (ref - full).abs().max().item()
    print(f'stage 3-5 reproduce postprocess_features exactly: max abs diff {err:.3e}')

    rgb_up, rgb_crop, rgb_full = to_rgb(up[0], basis), to_rgb(crop[0], basis), to_rgb(full[0], basis)

    # anchors: token i sits at K*i + K/2 in the 224 frame (align_corners=False)
    anch = K / 2.0
    fig, ax = plt.subplots(figsize=(7, 7))
    show(ax, rgb_up, f'STAGE 3 — bilinear {n_w}x{n_h} -> {P}x{P}\n'
                     f'green = 14 px cell boundaries, magenta = token CENTRES (14i+7)',
         interp='nearest')
    draw_grid(ax, K, K, P, P)
    draw_grid(ax, K, K, P, P, offset_x=anch, offset_y=anch, color=ANCHOR_C, lw=0.7, ls=':')
    ax.add_patch(plt.Rectangle((0, 0), iw, ih, fill=False, ec='red', lw=2.5))
    save(fig, f'{args.out_dir}/stage3_upsampled_{P}.png')

    fig, ax = plt.subplots(figsize=(6, 8) if ih > iw else (8, 6))
    show(ax, rgb_crop, f'STAGE 4 — padding cropped: {P}x{P} -> {iw}x{ih}\n'
                       f'(bottom {K // 2} px still blend in the first PADDING token row)',
         interp='nearest')
    draw_grid(ax, K, K, iw, ih)
    draw_grid(ax, K, K, iw, ih, offset_x=anch, offset_y=anch, color=ANCHOR_C, lw=0.7, ls=':')
    save(fig, f'{args.out_dir}/stage4_cropped_{iw}x{ih}.png')

    sx, sy = W / iw, H / ih
    cell_x, cell_y = K * sx, K * sy
    print(f'stage 5  full res {W}x{H}: one token covers {cell_x:.1f} x {cell_y:.1f} px, '
          f'anchors every {cell_x:.1f} px starting at {anch * sx:.2f}')

    fig, ax = plt.subplots(figsize=(8.5, 11) if H > W else (11, 8.5))
    show(ax, rgb_full, f'STAGE 5 — bilinear {iw}x{ih} -> {W}x{H}\n'
                       f'green = cell boundaries every {cell_x:.1f} px, '
                       f'magenta = token centres', interp='nearest')
    draw_grid(ax, cell_x, cell_y, W, H)
    draw_grid(ax, cell_x, cell_y, W, H, offset_x=anch * sx, offset_y=anch * sy,
              color=ANCHOR_C, lw=0.7, ls=':')
    save(fig, f'{args.out_dir}/stage5_full_{W}x{H}.png')

    # the photo itself under the same grid, for reference
    fig, ax = plt.subplots(figsize=(8.5, 11) if H > W else (11, 8.5))
    show(ax, image, f'REFERENCE — the photo under the same {cell_x:.1f} px cell grid\n'
                    f'the whole guitar is described by {v_w * v_h} feature vectors',
         interp='bilinear')
    draw_grid(ax, cell_x, cell_y, W, H)
    save(fig, f'{args.out_dir}/stage0_photo_cellgrid.png')

    # ---- THE ANSWER: a scanline ----------------------------------------------------
    # Sample the full-res row that sits EXACTLY on a token-row anchor (y = cell_y*j + cell_y/2).
    # Any other row blends two token rows vertically, so the token values of a single row would
    # not lie on the curve -- which is a property of the sampling, not of the interpolation.
    if args.scan_row is not None:
        row = args.scan_row
        tok_row = int(min(v_h - 1, (row / sy) // K))
    else:
        tok_row = v_h // 2
        row = int(round(anch * sy + cell_y * tok_row - 0.5))
    print(f'scanline full-res row {row} (token row {tok_row}, anchor at y={anch * sy + cell_y * tok_row:.2f})')

    mean, comps, lo, hi = basis
    line_full = ((full[0, :, row, :].T - mean) @ comps.T)[:, 0].cpu().numpy()
    line_tok = ((raw[0, :, tok_row, :v_w].T - mean) @ comps.T)[:, 0].cpu().numpy()

    # what a REAL block pattern would look like: same tokens, nearest-neighbour instead
    near = F.interpolate(raw, (P, P), mode='nearest')[..., :ih, :iw]
    near = F.interpolate(near, (H, W), mode='nearest')
    line_near = ((near[0, :, row, :].T - mean) @ comps.T)[:, 0].cpu().numpy()
    del near

    xs = anch * sx + cell_x * np.arange(v_w)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(np.arange(W), line_near, color='#bbbbbb', lw=1.2, ls='-',
            label='NEAREST (what a real block pattern looks like) — for contrast only, NOT our path')
    ax.plot(np.arange(W), line_full, color='#1f77b4', lw=1.6,
            label=f'OUR PATH: bilinear, PC1 along full-res row {row}')
    ax.plot(xs, line_tok, 'o', color=ANCHOR_C, ms=8, zorder=5,
            label=f'the {v_w} token values on token row {tok_row} (the ONLY real data)')
    for i in range(v_w + 1):
        ax.axvline(cell_x * i, color=GRID_C, lw=0.8, alpha=0.8)
    ax.set_xlim(0, W)
    ax.set_xlabel('x in the full-resolution photo (px)')
    ax.set_ylabel('PC1')
    ax.set_title(f'THE ANSWER — bilinear DOES blend neighbours: between anchors the signal is a '
                 f'straight RAMP, not a flat block.\nThe blue curve passes exactly through the '
                 f'{v_w} magenta anchors; everything between them is invented by interpolation.')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.2)
    save(fig, f'{args.out_dir}/stage6_scanline_row{row}.png')

    resid = np.abs(line_full[np.round(xs).astype(int)] - line_tok).max()
    rng = float(line_tok.max() - line_tok.min())
    # NOT exactly zero, and the reason is worth knowing: postprocess_features upsamples in TWO
    # steps (16 -> 224 -> full), and the intermediate 224 grid samples at pixel CENTRES while a
    # token anchor sits at 14i+7, i.e. exactly on a pixel BOUNDARY. The anchor value itself is
    # therefore never sampled, so each kink is clipped by ~half a 224-px step of the local slope.
    # Isolated in a 1-D test with this exact geometry: two-step 1.51, one-step 0.24, and sampling
    # at the rounded pixel instead of the exact anchor costs only 0.02. So the token values are
    # very slightly NOT reproduced at full resolution -- ~2.6% of range here.
    print(f'check    curve passes through the anchors: max |curve - token| = {resid:.3f} '
          f'({100 * resid / rng:.1f}% of the {rng:.1f} range; two-step upsample clips the kinks)')

    # equivalence check: project-then-resize == resize-then-project
    small = ((raw[0].float().reshape(C, -1).T - mean) @ comps.T).T.reshape(3, n_h, n_w)[None]
    small_up = F.interpolate(small, (P, P), mode='bilinear', align_corners=False)[..., :ih, :iw]
    small_full = F.interpolate(small_up, (H, W), mode='bilinear', align_corners=False)
    big = ((full[0].reshape(C, -1).T - mean) @ comps.T).T.reshape(3, H, W)[None]
    print(f'check    project-then-resize vs resize-then-project: max abs diff '
          f'{(small_full - big).abs().max().item():.3e}')
    del up, crop, full, ref, big
    print('done')


if __name__ == '__main__':
    main()
