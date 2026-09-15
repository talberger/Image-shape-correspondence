"""
PHASE MARGINALIZATION at test time -- does averaging over patch-grid phases kill the lattice?

Written 2026-09-06. We measured that the photo feature map carries a lattice locked to DINO's
14 px patch grid: feature norm +17.2% at the token anchor, 16.5% swing in cosine similarity
inside a single cell (analysis/blockpattern_selfsim.py). The literature says that artifact is
positional, not image-driven (Denoising Vision Transformers, arXiv 2401.02957, which finds it
in DINOv2-reg too -- the model we use), and the training-free remedy is to marginalize over the
grid phase at inference (arXiv 2606.08132).

The idea: the artifact is nailed to the patch grid, so shifting the image by a sub-patch offset
moves the artifact but not the content. Re-align and average over offsets and the artifact
cancels while the content survives.

HOW THIS IS DONE HERE, exactly:
  * the aspect-resized photo is placed at (K, K) in a zero canvas of P+2K, so every crop window
    below contains the whole image -- no content enters or leaves between phases.
  * for each phase (dy, dx) a (P+K)x(P+K) window is cropped at that offset and run through DINO,
    giving a 17x17 token grid whose token (j,i) sits at canvas position (dy+14j+7, dx+14i+7).
  * each phase's tokens are upsampled exactly the way postprocess_features does, written back
    into the canvas frame at its own offset, and only then averaged -- so the phases are aligned
    in IMAGE space, not in token space.
  * the metric is the same cell profile as before, always measured against the phase-0 grid.

Run:
  source /workspace/activate_bsb.sh
  python analysis/blockpattern_phase_marginalize.py --grids 1 2 3 4
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


def phase_profile(x, mk, ih, iw):
    """average x over whole cells of the PHASE-0 grid -> [K,K]; (7,7) is the token anchor."""
    acc, cnt = np.zeros((K, K)), np.zeros((K, K))
    for j in range(ih // K):
        for i in range(iw // K):
            c, cm = x[j*K:(j+1)*K, i*K:(i+1)*K], mk[j*K:(j+1)*K, i*K:(i+1)*K]
            acc += np.where(cm, c, 0)
            cnt += cm
    return acc / np.maximum(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_path', default='images/guitar_01.png')
    ap.add_argument('--mask_path',
                    default='demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth')
    ap.add_argument('--out_dir', default='demo/guitar/features_analyze/blockpattern/phase_marg')
    ap.add_argument('--target_size', type=int, default=224)
    ap.add_argument('--grids', type=int, nargs='+', default=[1, 2, 3, 4],
                    help='n -> n x n phases, offsets round(k*14/n)')
    ap.add_argument('--queries', type=int, nargs='+', default=[0, 1, 4, 7])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    P = args.target_size

    image = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    mask = torch.load(args.mask_path).squeeze().cpu().numpy().astype(bool)
    sq, isz = preprocess_image(image, P)
    ih, iw = int(isz[0]), int(isz[1])
    img = sq[0, :, :ih, :iw]                                  # aspect-resized, no padding

    Pc = P + K                                                # crop window, 17 patches
    canvas = torch.zeros(3, P + 2 * K, P + 2 * K, device=img.device, dtype=img.dtype)
    canvas[:, K:K + ih, K:K + iw] = img
    assert K - 1 + Pc <= canvas.shape[-1], 'canvas too small for the phase sweep'

    model = DINOWrapper(device=device, target_size=Pc)
    mk = F.interpolate(torch.from_numpy(mask[None, None].astype(np.float32)), (ih, iw),
                       mode='nearest')[0, 0].numpy() > .5

    def run_phase(dy, dx):
        win = canvas[:, dy:dy + Pc, dx:dx + Pc][None]
        with torch.no_grad():
            tok = model(win)                                   # [1,C,17,17]
        up = F.interpolate(tok.float(), (Pc, Pc), mode='bilinear', align_corners=False)[0].cpu()
        out = torch.zeros(up.shape[0], canvas.shape[-2], canvas.shape[-1])
        out[:, dy:dy + Pc, dx:dx + Pc] = up
        return out[:, K:K + ih, K:K + iw]                      # back to the image frame

    results = {}
    for n in args.grids:
        offs = [int(round(k * K / n)) for k in range(n)]
        t0 = time.time()
        acc = None
        for dy in offs:
            for dx in offs:
                f = run_phase(dy, dx)
                acc = f if acc is None else acc + f
        acc /= (n * n)
        dt = time.time() - t0

        nrm = acc.norm(dim=0).numpy()
        flat = F.normalize(acc.reshape(acc.shape[0], -1), dim=0)
        q_in = [(int((r + .5) * ih / H - .5), int((c + .5) * iw / W - .5)) for r, c in TRACKED]
        pn = phase_profile(nrm, mk, ih, iw)
        ps = np.mean([phase_profile((flat[:, r*iw+c] @ flat).reshape(ih, iw).numpy(), mk, ih, iw)
                      for r, c in q_in], 0)
        sw_n = 100 * (pn.max() / pn.min() - 1)
        sw_s = 100 * (ps.max() / ps.min() - 1)
        results[n] = dict(acc=acc, pn=pn, ps=ps, sw_n=sw_n, sw_s=sw_s, dt=dt, nph=n * n,
                          offs=offs)
        print(f'n={n} ({n*n:2d} phases, offsets {offs}): norm swing {sw_n:5.1f}%, '
              f'sim swing {sw_s:5.1f}%,  {dt:.1f}s')

    # does the averaging destroy the signal? compare against the single-phase map
    base = results[args.grids[0]]['acc']
    bflat = F.normalize(base.reshape(base.shape[0], -1), dim=0)
    print('\nagreement with the single-phase map (per-pixel cosine, guitar only):')
    gm = torch.from_numpy(mk.reshape(-1))
    for n in args.grids:
        a = results[n]['acc']
        af = F.normalize(a.reshape(a.shape[0], -1), dim=0)
        c = (af * bflat).sum(0)[gm]
        print(f'  n={n}: mean {c.mean():.4f}   min {c.min():.4f}')

    # ---- figure 1: the cell profile, before vs after ----
    ns = args.grids
    fig, axes = plt.subplots(2, len(ns), figsize=(3.7 * len(ns), 7.4))
    axes = np.atleast_2d(axes)
    for c, n in enumerate(ns):
        r_ = results[n]
        for r, (d, lab, sw) in enumerate([(r_['pn'], 'norm ||f||', r_['sw_n']),
                                          (r_['ps'], 'self-similarity', r_['sw_s'])]):
            ax = axes[r, c]
            im = ax.imshow(d, cmap='inferno', extent=[0, K, K, 0], interpolation='nearest')
            ax.plot([K / 2], [K / 2], 'o', mfc='none', mec='#00ffff', ms=11, mew=2)
            ax.set_title(f'{r_["nph"]} phase' + ('s' if r_['nph'] > 1 else '') +
                         f' — {lab}\nswing {sw:.1f}%', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('PHASE MARGINALIZATION — the same cell, averaged over every guitar cell.\n'
                 'Cyan = the phase-0 token anchor. Flatter = the lattice is gone.', fontsize=12)
    fig.tight_layout()
    p = f'{args.out_dir}/lattice_by_phases.png'
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); print(f'\n  wrote {p}')

    # ---- figure 2: full-res self-similarity, EVERY phase count ----
    show = list(ns)
    fig, axes = plt.subplots(len(show), len(args.queries),
                             figsize=(3.1 * len(args.queries), 4.1 * len(show)))
    axes = np.atleast_2d(axes)
    for r, n in enumerate(show):
        full = F.interpolate(results[n]['acc'][None], (H, W), mode='bilinear',
                             align_corners=False)[0]
        ff = F.normalize(full.reshape(full.shape[0], -1), dim=0)
        for c, k in enumerate(args.queries):
            s = (ff[:, TRACKED[k][0] * W + TRACKED[k][1]] @ ff).reshape(H, W).numpy()
            ax = axes[r, c]
            im = ax.imshow(s, cmap='inferno', vmin=0, vmax=1, extent=[0, W, H, 0],
                           interpolation='nearest')
            cell = W / (iw // K)
            for i in range(1, int(np.ceil(W / cell))):
                ax.axvline(cell * i, color='#00ff00', lw=0.35, alpha=0.45)
            for j in range(1, int(np.ceil(H / cell))):
                ax.axhline(cell * j, color='#00ff00', lw=0.35, alpha=0.45)
            ax.plot([TRACKED[k][1]], [TRACKED[k][0]], 'o', mfc='none', mec='#00ffff', ms=10, mew=2)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'{k}: {NAMES[k]}', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{results[n]["nph"]} phase'
                              f'{"s" if results[n]["nph"] > 1 else ""}\n'
                              f'swing {results[n]["sw_s"]:.1f}%', fontsize=10)
        del full, ff
    fig.suptitle('SELF-SIMILARITY AT FULL RESOLUTION — by number of grid phases averaged\n'
                 f'input {P}, same queries and colour scale, no training', fontsize=13)
    fig.subplots_adjust(right=0.9, top=0.92)
    fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.012, 0.7]), label='cosine similarity')
    p = f'{args.out_dir}/selfsim_phase_marginalized.png'
    fig.savefig(p, dpi=120, bbox_inches='tight'); plt.close(fig); print(f'  wrote {p}')


if __name__ == '__main__':
    main()
