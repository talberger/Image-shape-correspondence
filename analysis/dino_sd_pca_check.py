"""What the DINO and SD feature maps actually look like, immediately BEFORE fuse_dino_sd.

`analysis/dino_sd_overlay_check.py` reduces each map to a SALIENCY scalar (L2 norm of the
channel-centred vector). That answers "is the object in the same place" -- a framing question --
and throws away all 2304 channels of structure. Tal, 2026-09-03: the question is what the FEATURE
map looks like, so reduce with PCA to 3 components and show it as RGB instead.

IMPORTANT, and it limits what the figure can prove: DINO and SD live in different channel spaces,
so each map gets its OWN PCA basis. Colours are therefore NOT comparable between the two panels --
only STRUCTURE is (which regions are distinguished from which). Component signs are pinned by a
fixed convention (make each component's mean over the top-left quadrant negative) so the colours do
not flip between runs, which would otherwise make two identical maps look different.

The third panel is the one that IS directly comparable: for a chosen query cell, the cosine
similarity of every cell to that cell, computed separately in each map's own space and shown as a
difference. Same query location, same units (cosine), so disagreement there is real disagreement
about what that point looks like -- not a basis artefact.
"""
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.dino_sd_overlay_check import valid_cells


def pca_rgb(x):
    """[C, h, w] -> [h, w, 3] in 0..1 via PCA to 3 components, signs pinned."""
    c, h, w = x.shape
    flat = (x.reshape(c, -1) - x.reshape(c, -1).mean(1, keepdim=True)).T   # [hw, C]
    u, s, v = torch.pca_lowrank(flat, q=3, center=False)
    comp = (flat @ v).T.reshape(3, h, w)
    for i in range(3):                      # sign convention: top-left quadrant goes negative
        if comp[i, :h // 2, :w // 2].mean() > 0:
            comp[i] = -comp[i]
    lo = comp.reshape(3, -1).quantile(0.02, dim=1)[:, None, None]
    hi = comp.reshape(3, -1).quantile(0.98, dim=1)[:, None, None]
    return ((comp - lo) / (hi - lo).clamp(min=1e-8)).clamp(0, 1).permute(1, 2, 0)


def simmap(x, qy, qx):
    c, h, w = x.shape
    f = F.normalize(x.reshape(c, -1).T, dim=1)
    return (f @ f[qy * w + qx]).reshape(h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_pregrid_path', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--save_path', required=True)
    ap.add_argument('--sd_input_res', type=int, default=512)
    ap.add_argument('--grid', type=int, default=48)
    ap.add_argument('--sd_no_padding', action='store_true',
                    help='SD was fed a non-square frame with no padding (this is the case for '
                         'squash: the photo is STRETCHED to fill 512x512, so every SD cell is real '
                         'image). Without this the valid-region crop throws away real content.')
    ap.add_argument('--as_fused', action='store_true',
                    help='Reproduce fuse_dino_sd EXACTLY: no valid-region cropping on either side, '
                         'SD bilinearly resized onto DINO\'s raw padded grid. This is what the '
                         'production fusion actually concatenates -- the cropped view is more '
                         'forgiving than anything the model ever sees.')
    ap.add_argument('--query_frac', type=float, nargs=2, default=[0.30, 0.42],
                    help='query cell as (row, col) fraction of the grid; default is on the neck')
    args = ap.parse_args()

    d = torch.load(args.raw_pregrid_path, map_location='cpu')
    dino, sd = d['dino_raw'].float(), d['sd_raw'].float()
    in_h, in_w = d['input_size']; orig_h, orig_w = d['original_size']
    dino_target = d.get('dino_target_size', max(in_h, in_w))

    if args.as_fused:
        # fuse_dino_sd(): SD -> DINO's RAW grid, padding included, nothing cropped.
        gh, gw = dino.shape[-2:]
        D = dino
        S = F.interpolate(sd[None], size=(gh, gw), mode='bilinear', align_corners=False)[0]
    else:
        dh, dw = valid_cells(dino.shape[-2:], (in_h, in_w), dino_target)
        dino_v = dino[:, :dh, :dw]
        if args.sd_no_padding:
            sd_v = sd
        else:
            sd_scale = args.sd_input_res / max(orig_h, orig_w)
            sh, sw = valid_cells(sd.shape[-2:], (round(orig_h * sd_scale), round(orig_w * sd_scale)),
                                 args.sd_input_res)
            sd_v = sd[:, :sh, :sw]
        gh = args.grid if orig_h >= orig_w else max(1, round(args.grid * orig_h / orig_w))
        gw = args.grid if orig_w > orig_h else max(1, round(args.grid * orig_w / orig_h))
        D = F.interpolate(dino_v[None], size=(gh, gw), mode='bilinear', align_corners=False)[0]
        S = F.interpolate(sd_v[None], size=(gh, gw), mode='bilinear', align_corners=False)[0]

    qy, qx = int(args.query_frac[0] * gh), int(args.query_frac[1] * gw)
    sD, sS = simmap(D, qy, qx), simmap(S, qy, qx)
    agree = float(np.corrcoef(sD.flatten().numpy(), sS.flatten().numpy())[0, 1])

    fig, ax = plt.subplots(1, 4, figsize=(19, 6.2), dpi=150)
    ax[0].imshow(pca_rgb(D).numpy(), interpolation='nearest'); ax[0].set_title('DINO features, PCA->RGB', fontsize=11)
    ax[1].imshow(pca_rgb(S).numpy(), interpolation='nearest'); ax[1].set_title('SD features, PCA->RGB', fontsize=11)
    for a, m, t in ((ax[2], sD, 'DINO'), (ax[3], sS, 'SD')):
        im = a.imshow(m.numpy(), cmap='magma', vmin=-1, vmax=1, interpolation='nearest')
        a.set_title(f'{t}: cosine to the query cell', fontsize=11)
        a.plot([qx], [qy], 'o', mfc='none', mec='cyan', ms=13, mew=2.5)
        plt.colorbar(im, ax=a, fraction=0.046)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f'{args.label}   |   BEFORE fuse_dino_sd   |   grid {gh}x{gw}   |   '
                 f'corr between the two cosine maps = {agree:.3f}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    fig.savefig(args.save_path, bbox_inches='tight', facecolor='white')
    print(f'--- {args.label}\n  query cell ({qy},{qx}) of {gh}x{gw}\n'
          f'  corr between DINO and SD cosine-to-query maps: {agree:.4f}\n  saved {args.save_path}')


if __name__ == '__main__':
    main()
