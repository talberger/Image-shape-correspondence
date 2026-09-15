"""
DOES RAISING THE INPUT RESOLUTION MAKE DINOv2's TOKENS SPATIALLY SMOOTH? -- photo side, DINO only.

Written 2026-09-06. The 224 input gives a 12x16 valid grid whose ADJACENT tokens have cosine
0.570 (random far pairs: 0.264; the mesh side's 1-ring: 0.986), and that dissimilarity is what
produces the 67.5 px lattice in the similarity maps (analysis/blockpattern_selfsim.py).

Two competing explanations, and this script separates them:
  (a) RESOLUTION -- at 224 one cell covers 67.5 px of the photo, so adjacent cells genuinely
      contain different things and 0.57 is honest. Raising the resolution should raise it.
  (b) THE MODEL -- ViT tokens carry position-dependent noise regardless of scale, so adjacent
      tokens stay dissimilar however fine the grid gets.
If (a), the cosine climbs with target_size. If (b), it stays flat.

The control that makes this a real test: `--match_content` also reports the cosine between
tokens that are FURTHER APART at high resolution but cover the SAME photo distance as an
adjacent pair at 224 (67.5 px). If the model is fine and only the sampling was coarse, that
number should match 224's 0.570.

Run:
  source /workspace/activate_bsb.sh
  python analysis/blockpattern_resolution.py --sizes 224 448 840
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder import preprocess_image
from modules.dino import DINOWrapper
from modules.utils import device

K = 14


def adjacent_stats(f, gt, cell_rgb, step=1):
    """f [C,h,w] L2-normalized; gt [h,w] guitar fraction; cell_rgb [h,w,3]."""
    h, w = f.shape[1:]
    cos, drgb, gg, bb = [], [], [], []
    for j in range(h):
        for i in range(w):
            for dj, di in ((0, step), (step, 0)):
                if j + dj < h and i + di < w:
                    cos.append(float((f[:, j, i] * f[:, j + dj, i + di]).sum()))
                    drgb.append(float(np.linalg.norm(cell_rgb[j, i] - cell_rgb[j + dj, i + di])))
                    gg.append(gt[j, i] > .8 and gt[j + dj, i + di] > .8)
                    bb.append(gt[j, i] < .2 and gt[j + dj, i + di] < .2)
    return (np.array(cos), np.array(drgb), np.array(gg), np.array(bb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_path', default='images/guitar_01.png')
    ap.add_argument('--mask_path',
                    default='demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth')
    ap.add_argument('--sizes', type=int, nargs='+', default=[224, 448, 840])
    args = ap.parse_args()

    image = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    mask = torch.load(args.mask_path).squeeze().cpu().numpy().astype(bool)
    rng = np.random.default_rng(0)
    rows = []

    for P in args.sizes:
        assert P % K == 0, f'{P} is not a multiple of the 14 px patch'
        sq, isz = preprocess_image(image, P)
        ih, iw = int(isz[0]), int(isz[1])
        vh, vw = ih // K, iw // K
        model = DINOWrapper(device=device, target_size=P)
        torch.cuda.empty_cache()
        t0 = time.time()
        with torch.no_grad():
            raw = model(sq)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0
        torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
        raw = raw.float().cpu()[0][:, :vh, :vw]
        f = F.normalize(raw, dim=0)
        del model
        torch.cuda.empty_cache()

        rgb = sq[0, :, :ih, :iw].cpu().numpy().transpose(1, 2, 0)
        mk = F.interpolate(torch.from_numpy(mask[None, None].astype(np.float32)), (ih, iw),
                           mode='nearest')[0, 0].numpy()
        cell_rgb = rgb.reshape(vh, K, vw, K, 3).mean((1, 3))
        gt = mk.reshape(vh, K, vw, K).mean((1, 3))

        cos, drgb, gg, bb = adjacent_stats(f, gt, cell_rgb)
        # the floor: random pairs at least 3 cells apart
        far = []
        for _ in range(6000):
            j1, i1, j2, i2 = rng.integers(0, vh), rng.integers(0, vw), rng.integers(0, vh), rng.integers(0, vw)
            if abs(j1 - j2) + abs(i1 - i2) > 2:
                far.append(float((f[:, j1, i1] * f[:, j2, i2]).sum()))
        cell_px = W / vw
        # content-matched control: how many cells apart is 67.5 px at this resolution?
        step = max(1, int(round(67.5 / cell_px)))
        cos_m = adjacent_stats(f, gt, cell_rgb, step=step)[0] if step > 1 else cos

        rows.append(dict(P=P, grid=f'{vw}x{vh}', ntok=vw * vh, cell=cell_px, dt=dt, peak=peak,
                         adj=cos.mean(), adj_g=cos[gg].mean() if gg.sum() else np.nan,
                         adj_b=cos[bb].mean() if bb.sum() else np.nan,
                         floor=np.mean(far), corr=np.corrcoef(cos, drgb)[0, 1],
                         step=step, matched=cos_m.mean(), ng=int(gg.sum())))
        print(f'{P:4d}: grid {vw}x{vh} = {vw*vh:5d} tok, cell {cell_px:5.1f} px, '
              f'{dt:5.2f}s, peak {peak:4.1f} GB, adj {cos.mean():.3f}, floor {np.mean(far):.3f}')

    print('\n' + '=' * 108)
    print(f'{"input":>6} {"grid":>8} {"tokens":>7} {"cell px":>8} | {"ADJACENT":>9} {"guitar":>8} '
          f'{"bkgd":>7} {"floor":>7} | {"corr img":>9} | {"67.5px-matched":>15}')
    print('-' * 108)
    for r in rows:
        print(f'{r["P"]:>6} {r["grid"]:>8} {r["ntok"]:>7} {r["cell"]:>8.1f} | {r["adj"]:>9.3f} '
              f'{r["adj_g"]:>8.3f} {r["adj_b"]:>7.3f} {r["floor"]:>7.3f} | {r["corr"]:>+9.3f} | '
              f'{r["matched"]:>10.3f} (step {r["step"]})')
    print('=' * 108)
    print('ADJACENT = mean cosine between neighbouring tokens. floor = random far pairs.')
    print('corr img = corr(token cosine, mean-RGB distance of the two cells) -- how much the')
    print('           feature similarity tracks what the image actually shows.')
    print('67.5px-matched = tokens spaced the same PHOTO distance as an adjacent pair at 224;')
    print('           if only the sampling was coarse, this should stay near 224\'s value.')
    print(f'reference: mesh-side 1-ring cosine = 0.986 (PROJECT_NOTES.md)')


if __name__ == '__main__':
    main()
