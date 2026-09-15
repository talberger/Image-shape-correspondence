"""
THE RESOLUTION EXPERIMENT, VISUALLY -- self-similarity at 224 / 448 / 840, photo side, DINO only.

Companion to analysis/blockpattern_resolution.py, which measured that adjacent-token cosine
rises 0.570 -> 0.678 -> 0.739 with the input size. This shows what that does to the two things
we can actually see: the 67.5 px LATTICE in the similarity maps, and the detail the map can
resolve at all.

Every resolution is taken all the way to the full 810x1080 map through the production
postprocess_features path, so the three rows are directly comparable -- same queries, same
colour scale, same output size. What differs is only DINOWrapper's target_size.

Run:
  source /workspace/activate_bsb.sh
  python analysis/blockpattern_resolution_viz.py --sizes 224 448 840
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

K = 14
TRACKED = [(151, 94), (338, 240), (458, 304), (571, 206), (700, 650),
           (853, 736), (922, 382), (930, 520), (655, 455)]
NAMES = ['headstock', 'neck', 'neck lower', 'left horn', 'body/knobs',
         'body right edge', 'lower body', 'lower bout', 'upper bout']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_path', default='images/guitar_01.png')
    ap.add_argument('--mask_path',
                    default='demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth')
    ap.add_argument('--out_dir', default='demo/guitar/features_analyze/blockpattern/resolution')
    ap.add_argument('--sizes', type=int, nargs='+', default=[224, 448, 840])
    ap.add_argument('--queries', type=int, nargs='+', default=[0, 1, 4, 7])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    image = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    mask = torch.load(args.mask_path).squeeze().cpu().numpy().astype(bool)

    sims, profiles, meta = {}, {}, {}
    for P in args.sizes:
        sq, isz = preprocess_image(image, P)
        ih, iw = int(isz[0]), int(isz[1])
        vh, vw = ih // K, iw // K
        model = DINOWrapper(device=device, target_size=P)
        with torch.no_grad():
            raw = model(sq).float().cpu()
        del model
        torch.cuda.empty_cache()

        # --- the cell profile, in this resolution's own 14 px cell ---
        up_in = F.interpolate(raw, (P, P), mode='bilinear', align_corners=False)[0][:, :ih, :iw]
        mk = F.interpolate(torch.from_numpy(mask[None, None].astype(np.float32)), (ih, iw),
                           mode='nearest')[0, 0].numpy() > .5
        nrm = up_in.norm(dim=0).numpy()
        flat_in = F.normalize(up_in.reshape(up_in.shape[0], -1), dim=0)

        def phase(x):
            acc, cnt = np.zeros((K, K)), np.zeros((K, K))
            for j in range(ih // K):
                for i in range(iw // K):
                    c, cm = x[j*K:(j+1)*K, i*K:(i+1)*K], mk[j*K:(j+1)*K, i*K:(i+1)*K]
                    acc += np.where(cm, c, 0); cnt += cm
            return acc / np.maximum(cnt, 1)

        q_in = [(int((r + .5) * ih / H - .5), int((c + .5) * iw / W - .5)) for r, c in TRACKED]
        pn = phase(nrm)
        ps = np.mean([phase((flat_in[:, r*iw+c] @ flat_in).reshape(ih, iw).numpy())
                      for r, c in q_in], 0)
        profiles[P] = (pn, ps)
        del up_in, flat_in

        # --- full-resolution similarity, through the production path ---
        t0 = time.time()
        full = F.interpolate(F.interpolate(raw, (P, P), mode='bilinear',
                                           align_corners=False)[..., :ih, :iw],
                             (H, W), mode='bilinear', align_corners=False)[0]
        flat = F.normalize(full.reshape(full.shape[0], -1), dim=0)
        sims[P] = {k: (flat[:, TRACKED[k][0]*W + TRACKED[k][1]] @ flat).reshape(H, W).numpy()
                   for k in args.queries}
        meta[P] = dict(grid=f'{vw}x{vh}', ntok=vw*vh, cell=W/vw,
                       swing_n=100*(pn.max()/pn.min()-1), swing_s=100*(ps.max()/ps.min()-1))
        print(f'{P}: grid {vw}x{vh} = {vw*vh} tokens, cell {W/vw:.1f} px, '
              f'norm swing {meta[P]["swing_n"]:.1f}%, sim swing {meta[P]["swing_s"]:.1f}%  '
              f'({time.time()-t0:.1f}s)')
        del full, flat

    # ---- figure 1: rows = resolution, cols = query ----
    nq = len(args.queries)
    fig, axes = plt.subplots(len(args.sizes), nq, figsize=(3.1*nq, 4.1*len(args.sizes)))
    axes = np.atleast_2d(axes)
    for r, P in enumerate(args.sizes):
        for c, k in enumerate(args.queries):
            ax = axes[r, c]
            im = ax.imshow(sims[P][k], cmap='inferno', vmin=0, vmax=1, extent=[0, W, H, 0],
                           interpolation='nearest')
            cell = meta[P]['cell']
            for i in range(1, int(np.ceil(W/cell))):
                ax.axvline(cell*i, color='#00ff00', lw=0.35, alpha=0.45)
            for j in range(1, int(np.ceil(H/cell))):
                ax.axhline(cell*j, color='#00ff00', lw=0.35, alpha=0.45)
            ax.plot([TRACKED[k][1]], [TRACKED[k][0]], 'o', mfc='none', mec='#00ffff', ms=10, mew=2)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'{k}: {NAMES[k]}', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'input {P}\n{meta[P]["grid"]} = {meta[P]["ntok"]} tokens\n'
                              f'cell {meta[P]["cell"]:.0f} px', fontsize=9)
    fig.suptitle('SELF-SIMILARITY AT THE FULL 810x1080, BY INPUT RESOLUTION\n'
                 'same queries, same colour scale; green = that row\'s own cell grid', fontsize=13)
    fig.subplots_adjust(right=0.9, top=0.93)
    fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.012, 0.7]), label='cosine similarity')
    p = f'{args.out_dir}/selfsim_by_resolution.png'
    fig.savefig(p, dpi=120, bbox_inches='tight'); plt.close(fig); print(f'  wrote {p}')

    # ---- figure 2: the cell profile shrinking ----
    fig, axes = plt.subplots(2, len(args.sizes), figsize=(3.7*len(args.sizes), 7.4))
    axes = np.atleast_2d(axes)
    for c, P in enumerate(args.sizes):
        pn, ps = profiles[P]
        for r, (d, lab, sw) in enumerate([(pn, 'norm ||f||', meta[P]['swing_n']),
                                          (ps, 'self-similarity', meta[P]['swing_s'])]):
            ax = axes[r, c]
            im = ax.imshow(d, cmap='inferno', extent=[0, K, K, 0], interpolation='nearest')
            ax.plot([K/2], [K/2], 'o', mfc='none', mec='#00ffff', ms=11, mew=2)
            ax.set_title(f'input {P} — {lab}\nswing {sw:.1f}%', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('THE LATTICE, ONE CELL, BY INPUT RESOLUTION — averaged over every guitar cell.\n'
                 'Cyan = token anchor. Smaller swing = weaker block pattern.', fontsize=12)
    fig.tight_layout()
    p = f'{args.out_dir}/lattice_by_resolution.png'
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); print(f'  wrote {p}')

    print('\n  input |   grid   | tokens | cell px | norm swing | sim swing')
    for P in args.sizes:
        m = meta[P]
        print(f'  {P:>5} | {m["grid"]:>8} | {m["ntok"]:>6} | {m["cell"]:>7.1f} | '
              f'{m["swing_n"]:>9.1f}% | {m["swing_s"]:>8.1f}%')


if __name__ == '__main__':
    main()
