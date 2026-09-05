"""Do the DINO and SD feature maps actually cover the same thing?

Reads a *_RAW_pregrid.pth written by analysis/extract_dino_sd_features.py (which stores dino_raw,
sd_raw and the frame metadata) and renders a two-colour overlay of the two saliency maps on a common
grid: GREEN = DINO, RED = SD, YELLOW = both. Plus the numbers -- centroid offset, IoU and
correlation -- so agreement is a measurement and not an impression.

Saliency per cell = L2 norm of the channel-centred feature vector, i.e. how far that cell is from the
map's own mean. On these features that tracks "object" vs "background/padding" well, and needs no
segmentation mask, so the check stays independent of anything downstream.

Both sides are cropped to their VALID (non-padding) region first -- otherwise the comparison is
dominated by padding that gets thrown away later anyway.

Run:
python analysis/dino_sd_overlay_check.py \
  --raw_pregrid_path demo/guitar/features_analyze/dino_sd_aligned/guitar_01_pix_dino_sd_features_aligned_RAW_pregrid.pth \
  --label "WITH sd_square" --save_path demo/guitar/features_analyze/overlay_sd_square.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def saliency(x):
    """[C, h, w] -> [h, w] in 0..1. Distance of each cell from the map's own channel mean."""
    s = (x - x.mean(dim=(1, 2), keepdim=True)).norm(dim=0)
    s = s - s.min()
    return s / s.max().clamp(min=1e-8)


def valid_cells(grid_hw, frame_hw, res):
    """How many cells of a res x res padded square are real image, given the content size."""
    gh, gw = grid_hw
    fh, fw = frame_hw
    return max(1, round(gh * fh / res)), max(1, round(gw * fw / res))


def centroid(s):
    h, w = s.shape
    ys = torch.arange(h, dtype=torch.float32)[:, None].expand(h, w)
    xs = torch.arange(w, dtype=torch.float32)[None, :].expand(h, w)
    t = s.sum().clamp(min=1e-8)
    return (ys * s).sum().item() / t, (xs * s).sum().item() / t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_pregrid_path', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--save_path', required=True)
    ap.add_argument('--sd_input_res', type=int, default=512,
                    help='the square SD was padded to; ignored when --sd_no_padding is set')
    ap.add_argument('--sd_no_padding', action='store_true',
                    help='SD was fed a non-square frame with no padding, so every cell is valid')
    ap.add_argument('--grid', type=int, default=64, help='common grid, long side')
    ap.add_argument('--single_panel', action='store_true',
                    help='one large overlay instead of the 4-panel strip (2026-09-03, Tal): the '
                         'two saliency maps on top of each other, GREEN=DINO, RED=SD, YELLOW=both, '
                         'with each map\'s >0.5 region outlined so what overlaps and what does not '
                         'is readable at a glance rather than inferred from a colour blend.')
    args = ap.parse_args()

    d = torch.load(args.raw_pregrid_path, map_location='cpu')
    dino, sd = d['dino_raw'].float(), d['sd_raw'].float()
    in_h, in_w = d['input_size']          # DINO's content size inside its padded square
    orig_h, orig_w = d['original_size']
    dino_target = d.get('dino_target_size', max(in_h, in_w))

    # crop each side to its own valid region
    dh, dw = valid_cells(dino.shape[-2:], (in_h, in_w), dino_target)
    dino_v = dino[:, :dh, :dw]
    if args.sd_no_padding:
        sd_v = sd
    else:
        sd_scale = args.sd_input_res / max(orig_h, orig_w)
        sh, sw = valid_cells(sd.shape[-2:], (round(orig_h * sd_scale), round(orig_w * sd_scale)),
                             args.sd_input_res)
        sd_v = sd[:, :sh, :sw]

    # both onto one grid, sized to the image's aspect
    gh = args.grid if orig_h >= orig_w else max(1, round(args.grid * orig_h / orig_w))
    gw = args.grid if orig_w > orig_h else max(1, round(args.grid * orig_w / orig_h))
    D = saliency(F.interpolate(dino_v[None], size=(gh, gw), mode='bilinear', align_corners=False)[0])
    S = saliency(F.interpolate(sd_v[None], size=(gh, gw), mode='bilinear', align_corners=False)[0])

    dcy, dcx = centroid(D)
    scy, scx = centroid(S)
    off_y, off_x = scy - dcy, scx - dcx
    px_y, px_x = off_y * orig_h / gh, off_x * orig_w / gw
    thr = 0.5
    Db, Sb = D > thr, S > thr
    iou = (Db & Sb).sum().item() / max((Db | Sb).sum().item(), 1)
    corr = float(np.corrcoef(D.flatten().numpy(), S.flatten().numpy())[0, 1])

    rgb = torch.zeros(gh, gw, 3)
    rgb[..., 0] = S      # red   = SD
    rgb[..., 1] = D      # green = DINO   (both high -> yellow)

    if args.single_panel:
        import matplotlib.patches as mpatches
        fig, a = plt.subplots(figsize=(9, 11), dpi=160)
        a.imshow(rgb.numpy(), interpolation='nearest')
        # Outlines, not just the colour blend: a dim yellow and a dim green are hard to tell apart.
        a.contour(Db.numpy().astype(float), levels=[0.5], colors='lime', linewidths=2.0)
        a.contour(Sb.numpy().astype(float), levels=[0.5], colors='red', linewidths=2.0)
        a.plot([dcx], [dcy], 'o', mfc='none', mec='lime', ms=15, mew=2.5)
        a.plot([scx], [scy], 'x', color='red', ms=15, mew=2.5)
        a.set_xticks([]); a.set_yticks([])
        a.legend(handles=[mpatches.Patch(color='#00ff00', label='DINO only'),
                          mpatches.Patch(color='#ff0000', label='SD only'),
                          mpatches.Patch(color='#ffff00', label='BOTH (agree)'),
                          mpatches.Patch(color='black', label='neither')],
                 loc='lower left', fontsize=11, framealpha=0.9)
        a.set_title(f'{args.label}\nDINO vs SD saliency, immediately BEFORE fuse_dino_sd\n'
                    f'IoU@0.5 {iou:.3f}   corr {corr:.3f}   '
                    f'centroid offset {px_y:+.0f}, {px_x:+.0f} px',
                    fontsize=12)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        fig.savefig(args.save_path, bbox_inches='tight', facecolor='white')
        print(f'--- {args.label}')
        print(f'  IoU@0.5 {iou:.4f}   correlation {corr:.4f}')
        print(f'  saved {args.save_path}')
        return

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6), dpi=150)
    ax[0].imshow(D.numpy(), cmap='Greens'); ax[0].set_title('DINO saliency', fontsize=10)
    ax[1].imshow(S.numpy(), cmap='Reds'); ax[1].set_title('SD saliency', fontsize=10)
    ax[2].imshow(rgb.numpy()); ax[2].set_title('overlay: green DINO / red SD / YELLOW = agree',
                                               fontsize=10)
    ax[2].plot([dcx], [dcy], 'o', mfc='none', mec='lime', ms=11, mew=2)
    ax[2].plot([scx], [scy], 'x', color='red', ms=11, mew=2)
    ax[3].imshow((D - S).numpy(), cmap='bwr', vmin=-1, vmax=1)
    ax[3].set_title('DINO - SD  (blue = SD stronger)', fontsize=10)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f'{args.label}   |   centroid offset {off_y:+.2f}, {off_x:+.2f} cells '
                 f'= {px_y:+.0f}, {px_x:+.0f} px   |   IoU@0.5 {iou:.3f}   |   corr {corr:.3f}',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    fig.savefig(args.save_path, bbox_inches='tight', facecolor='white')

    print(f'--- {args.label}')
    print(f'  dino grid {tuple(dino.shape[-2:])} -> valid {dh}x{dw}   '
          f'sd grid {tuple(sd.shape[-2:])} -> valid {tuple(sd_v.shape[-2:])}')
    print(f'  centroid offset (SD - DINO): {off_y:+.3f}, {off_x:+.3f} cells  '
          f'= {px_y:+.1f}, {px_x:+.1f} px of {orig_h}x{orig_w}')
    print(f'  IoU@0.5 {iou:.4f}   correlation {corr:.4f}')
    print(f'  saved {args.save_path}')


if __name__ == '__main__':
    main()
