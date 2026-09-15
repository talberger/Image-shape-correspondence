"""
SELF-SIMILARITY of the photo feature map, at every stage of the upsample -- DINO only, no SD.

Written 2026-09-06 at Tal's request, follow-up to analysis/blockpattern_stages.py: PCA->RGB
makes the cell structure hard to see, so this looks at it the way it is actually used --
cosine similarity between one query feature and every other position.

Four resolutions, same queries, same colour scale, so they can be compared directly:
  A  16x16   raw token grid (includes the 4 padding columns)
  B  12x16   the valid tokens only (padding dropped)  [W x H]
  C  224x224 after the first bilinear upsample
  D  810x1080 the full-resolution map the network actually consumes

Queries are the project's 9 tracked pixels (adopted 2026-08-17), given in full-resolution
(row, col) and mapped into each frame, so these figures line up with every other tracked-pixel
figure in the project.

Run:
  source /workspace/activate_bsb.sh
  python analysis/blockpattern_selfsim.py            # A, B, C
  python analysis/blockpattern_selfsim.py --full     # + D (needs ~11 GB RAM)
"""
import argparse
import os
import sys
import time

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

# the project's 9 tracked pixels, (row, col) in images/guitar_01.png -- see PROJECT_NOTES.md
TRACKED = [(151, 94), (338, 240), (458, 304), (571, 206), (700, 650),
           (853, 736), (922, 382), (930, 520), (655, 455)]
TRACKED_NAMES = ['headstock', 'neck', 'neck lower', 'left horn', 'body/knobs',
                 'body right edge', 'lower body', 'lower bout', 'upper bout']
K = 14


def cos_sim_maps(feat_chw, qs):
    """feat_chw [C,H,W]; qs list of (row, col). Returns list of [H,W] float32 numpy."""
    c, h, w = feat_chw.shape
    flat = feat_chw.reshape(c, -1)
    flat = F.normalize(flat, dim=0)
    out = []
    for (r, col) in qs:
        r = int(np.clip(r, 0, h - 1)); col = int(np.clip(col, 0, w - 1))
        q = flat[:, r * w + col]
        out.append((q @ flat).reshape(h, w).cpu().numpy())
    return out


def panel_fig(maps, qs, shape, cell, title, path, vmin, vmax, names):
    h, w = shape
    fig, axes = plt.subplots(3, 3, figsize=(13, 13 * h / w * 1.05))
    for k, ax in enumerate(axes.ravel()):
        im = ax.imshow(maps[k], extent=[0, w, h, 0], cmap='inferno', vmin=vmin, vmax=vmax,
                       interpolation='nearest')
        if cell is not None:
            cx, cy = cell
            for i in range(1, int(np.ceil(w / cx))):
                ax.axvline(cx * i, color='#00ff00', lw=0.4, alpha=0.55)
            for j in range(1, int(np.ceil(h / cy))):
                ax.axhline(cy * j, color='#00ff00', lw=0.4, alpha=0.55)
        r, c = qs[k]
        ax.plot([c + 0.5], [r + 0.5], 'o', mfc='none', mec='#00ffff', ms=11, mew=2)
        ax.set_title(f'{k}: {names[k]}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(right=0.9, top=0.94)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label='cosine similarity')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_path', default='images/guitar_01.png')
    ap.add_argument('--out_dir', default='demo/guitar/features_analyze/blockpattern/selfsim')
    ap.add_argument('--target_size', type=int, default=224)
    ap.add_argument('--full', action='store_true', help='also do the full-resolution map (~11 GB RAM)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    P = args.target_size

    image = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    input_square, input_size = preprocess_image(image, P)
    ih, iw = int(input_size[0]), int(input_size[1])
    v_h, v_w = ih // K, iw // K

    model = DINOWrapper(device=device, target_size=P)
    t0 = time.time()
    with torch.no_grad():
        raw = model(input_square).float().cpu()
    print(f'raw token grid {tuple(raw.shape)}   ({time.time() - t0:.1f}s)')

    # queries, mapped into each frame (align_corners=False convention)
    q_full = TRACKED
    q_224 = [(int((r + 0.5) * ih / H - 0.5), int((c + 0.5) * iw / W - 0.5)) for r, c in q_full]
    q_tok = [(min(v_h - 1, r224 // K), min(v_w - 1, c224 // K)) for r224, c224 in q_224]
    print('query mapping (full -> 224 -> token):')
    for k, (a, b, c) in enumerate(zip(q_full, q_224, q_tok)):
        print(f'  {k} {TRACKED_NAMES[k]:16s} {a} -> {b} -> {c}')

    stages = []
    A = raw[0]
    stages.append(('A_tokens_16x16', A, q_tok, (1, 1),
                   f'A — RAW TOKEN GRID {A.shape[2]}x{A.shape[1]} (includes the padding columns)'))
    B = raw[0][:, :v_h, :v_w].contiguous()
    stages.append(('B_tokens_valid', B, q_tok, (1, 1),
                   f'B — VALID TOKENS ONLY {v_w}x{v_h} = {v_w * v_h} (padding dropped)'))
    up = F.interpolate(raw, (P, P), mode='bilinear', align_corners=False)
    C = up[0]
    q_224_sq = q_224
    stages.append(('C_upsampled_224', C, q_224_sq, (K, K),
                   f'C — AFTER BILINEAR {A.shape[2]}x{A.shape[1]} -> {P}x{P} (green = 14 px cells)'))

    if args.full:
        t0 = time.time()
        D = F.interpolate(up[..., :ih, :iw], (H, W), mode='bilinear', align_corners=False)[0]
        print(f'full-res map {tuple(D.shape)} built in {time.time() - t0:.1f}s '
              f'({D.numel() * 4 / 2**30:.1f} GB)')
        stages.append(('D_full_res', D, q_full, (W / v_w, H / v_h),
                       f'D — FULL RESOLUTION {W}x{H} (green = {W / v_w:.1f} px cells)'))

    # one colour scale for every figure, from the actual similarity values
    all_maps = {}
    lo, hi = [], []
    for name, feat, qs, cell, title in stages:
        t0 = time.time()
        maps = cos_sim_maps(feat, qs)
        all_maps[name] = maps
        st = np.stack(maps)
        lo.append(np.percentile(st, 1)); hi.append(st.max())
        print(f'{name}: {len(maps)} sim maps of {maps[0].shape} in {time.time() - t0:.1f}s   '
              f'range [{st.min():.3f}, {st.max():.3f}]')
    vmin, vmax = float(min(lo)), float(max(hi))
    print(f'shared colour scale: [{vmin:.3f}, {vmax:.3f}]')

    for name, feat, qs, cell, title in stages:
        panel_fig(all_maps[name], qs, feat.shape[1:], cell,
                  f'SELF-SIMILARITY (cosine) — {title}\nqueries = the 9 tracked pixels, '
                  f'cyan circle = query; shared scale [{vmin:.2f}, {vmax:.2f}]',
                  f'{args.out_dir}/selfsim_{name}.png', vmin, vmax, TRACKED_NAMES)
    print('done')


if __name__ == '__main__':
    main()
