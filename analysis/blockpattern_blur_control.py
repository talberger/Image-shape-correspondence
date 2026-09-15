"""
THE BLUR CONTROL — is phase marginalization CANCELLING the artifact, or just smoothing?

Written 2026-09-06 at Tal's request, before acting on the phase-marginalization result.
Averaging four shifted runs is also, by itself, a smoothing operation. Two explanations for
the 16.8% -> 3.5% drop in lattice swing:
  (A) the artifact cancels, because it is locked to the patch grid and the phases move it;
  (B) nothing special -- any smoothing of that extent would flatten the profile too.

THE CONTROL: apply the SAME 4-tap kernel to the SINGLE-phase map -- the identical averaging
geometry (offsets 0 and 7 on both axes), but with no new DINO runs, so no new information.
  * if the control also lands near 3.5%, (B) holds and the extra forward passes bought nothing;
  * if the control stays high, the cancellation is real.

Reported alongside a SMEAR metric, because a method that kills the lattice by destroying detail
is not a win: mean cosine between a pixel and the pixel d px away, over the guitar. Higher =
more smeared. Both arms must be judged on the pair (swing, smear), not on swing alone.

Run:
  source /workspace/activate_bsb.sh
  python analysis/blockpattern_blur_control.py
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

K = 14
TRACKED = [(151, 94), (338, 240), (458, 304), (571, 206), (700, 650),
           (853, 736), (922, 382), (930, 520), (655, 455)]


def phase_profile(x, mk, ih, iw):
    acc, cnt = np.zeros((K, K)), np.zeros((K, K))
    for j in range(ih // K):
        for i in range(iw // K):
            c, cm = x[j*K:(j+1)*K, i*K:(i+1)*K], mk[j*K:(j+1)*K, i*K:(i+1)*K]
            acc += np.where(cm, c, 0); cnt += cm
    return acc / np.maximum(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_path', default='images/guitar_01.png')
    ap.add_argument('--mask_path',
                    default='demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth')
    ap.add_argument('--out_dir', default='demo/guitar/features_analyze/blockpattern/phase_marg')
    ap.add_argument('--target_size', type=int, default=224)
    args = ap.parse_args()
    P = args.target_size
    os.makedirs(args.out_dir, exist_ok=True)

    image = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    mask = torch.load(args.mask_path).squeeze().cpu().numpy().astype(bool)
    sq, isz = preprocess_image(image, P)
    ih, iw = int(isz[0]), int(isz[1])
    img = sq[0, :, :ih, :iw]

    Pc = P + K
    canvas = torch.zeros(3, P + 2*K, P + 2*K, device=img.device, dtype=img.dtype)
    canvas[:, K:K+ih, K:K+iw] = img
    model = DINOWrapper(device=device, target_size=Pc)
    mk = F.interpolate(torch.from_numpy(mask[None, None].astype(np.float32)), (ih, iw),
                       mode='nearest')[0, 0].numpy() > .5
    mkt = torch.from_numpy(mk)

    def run_phase(dy, dx):
        with torch.no_grad():
            tok = model(canvas[:, dy:dy+Pc, dx:dx+Pc][None])
        up = F.interpolate(tok.float(), (Pc, Pc), mode='bilinear', align_corners=False)[0].cpu()
        out = torch.zeros(up.shape[0], canvas.shape[-2], canvas.shape[-1])
        out[:, dy:dy+Pc, dx:dx+Pc] = up
        return out[:, K:K+ih, K:K+iw]

    OFFS = [0, 7]
    single = run_phase(0, 0)
    marg = torch.stack([run_phase(dy, dx) for dy in OFFS for dx in OFFS]).mean(0)

    # the control: identical 4-tap geometry applied to the SINGLE-phase map, no new runs
    pad = F.pad(single[None], (0, max(OFFS), 0, max(OFFS)), mode='replicate')[0]
    blur = torch.stack([pad[:, dy:dy+ih, dx:dx+iw] for dy in OFFS for dx in OFFS]).mean(0)
    # and a plain box filter of the same extent, as a second reference
    box = F.avg_pool2d(F.pad(single[None], (3, 4, 3, 4), mode='replicate'), 8, stride=1)[0]

    q_in = [(int((r + .5) * ih / H - .5), int((c + .5) * iw / W - .5)) for r, c in TRACKED]

    def smear(f, d):
        """mean cosine between a guitar pixel and the pixel d px to the right."""
        a = F.normalize(f, dim=0)
        v = (a[:, :, :-d] * a[:, :, d:]).sum(0)
        m = mkt[:, :-d] & mkt[:, d:]
        return float(v[m].mean())

    arms = [('1 phase (baseline)', single), ('4 phases MARGINALIZED', marg),
            ('CONTROL: same kernel, 1 run', blur), ('CONTROL: 8x8 box, 1 run', box)]
    print(f'{"arm":<30} {"norm swing":>11} {"sim swing":>10} | {"smear d=5":>10} '
          f'{"d=20":>7} {"d=40":>7}')
    print('-' * 84)
    rows = []
    for name, f in arms:
        nrm = f.norm(dim=0).numpy()
        flat = F.normalize(f.reshape(f.shape[0], -1), dim=0)
        pn = phase_profile(nrm, mk, ih, iw)
        ps = np.mean([phase_profile((flat[:, r*iw+c] @ flat).reshape(ih, iw).numpy(), mk, ih, iw)
                      for r, c in q_in], 0)
        sw_n = 100*(pn.max()/pn.min()-1); sw_s = 100*(ps.max()/ps.min()-1)
        s5, s20, s40 = smear(f, 5), smear(f, 20), smear(f, 40)
        rows.append((name, pn, ps, sw_n, sw_s, s5, s20, s40))
        print(f'{name:<30} {sw_n:>10.1f}% {sw_s:>9.1f}% | {s5:>10.4f} {s20:>7.4f} {s40:>7.4f}')
    print('-' * 84)
    print('smear = mean cosine to the pixel d px away, over the guitar. HIGHER = more blurred.')

    fig, axes = plt.subplots(2, len(rows), figsize=(3.6*len(rows), 7.4))
    for c, (name, pn, ps, sw_n, sw_s, *_ ) in enumerate(rows):
        for r, (d, lab, sw) in enumerate([(pn, 'norm', sw_n), (ps, 'self-similarity', sw_s)]):
            ax = axes[r, c]
            im = ax.imshow(d, cmap='inferno', extent=[0, K, K, 0], interpolation='nearest')
            ax.plot([K/2], [K/2], 'o', mfc='none', mec='#00ffff', ms=11, mew=2)
            ax.set_title(f'{name}\n{lab} — swing {sw:.1f}%', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('THE BLUR CONTROL — does the lattice die from CANCELLATION or from SMOOTHING?\n'
                 'The two right-hand arms use the same averaging geometry but only ONE DINO run.',
                 fontsize=12)
    fig.tight_layout()
    p = f'{args.out_dir}/blur_control.png'
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'\n  wrote {p}')

    # ---- the actual similarity maps, full resolution, arm by arm ----
    qs = [0, 1, 4, 7]
    names = ['headstock', 'neck', 'body/knobs', 'lower bout']
    fig, axes = plt.subplots(len(arms), len(qs), figsize=(3.1*len(qs), 4.1*len(arms)))
    for r, (name, f) in enumerate(arms):
        full = F.interpolate(f[None], (H, W), mode='bilinear', align_corners=False)[0]
        ff = F.normalize(full.reshape(full.shape[0], -1), dim=0)
        for c, k in enumerate(qs):
            sm = (ff[:, TRACKED[k][0]*W + TRACKED[k][1]] @ ff).reshape(H, W).numpy()
            ax = axes[r, c]
            im = ax.imshow(sm, cmap='inferno', vmin=0, vmax=1, extent=[0, W, H, 0],
                           interpolation='nearest')
            cell = W / (iw // K)
            for i in range(1, int(np.ceil(W/cell))):
                ax.axvline(cell*i, color='#00ff00', lw=0.35, alpha=0.4)
            for j in range(1, int(np.ceil(H/cell))):
                ax.axhline(cell*j, color='#00ff00', lw=0.35, alpha=0.4)
            ax.plot([TRACKED[k][1]], [TRACKED[k][0]], 'o', mfc='none', mec='#00ffff',
                    ms=10, mew=2)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(names[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{name}\nswing {rows[r][4]:.1f}%  smear {rows[r][6]:.3f}',
                              fontsize=9)
        del full, ff
    fig.suptitle('THE SAME SIMILARITY MAPS, ARM BY ARM — 4 real runs vs the 1-run controls\n'
                 'same queries and colour scale', fontsize=13)
    fig.subplots_adjust(right=0.9, top=0.94)
    fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.012, 0.7]), label='cosine similarity')
    p = f'{args.out_dir}/blur_control_selfsim.png'
    fig.savefig(p, dpi=120, bbox_inches='tight'); plt.close(fig)
    print(f'  wrote {p}')


if __name__ == '__main__':
    main()
