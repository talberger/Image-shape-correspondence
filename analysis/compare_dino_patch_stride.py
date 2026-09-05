"""
Overlapping-patch ("stride trick") DINO feature extraction, and a side-by-side against the
production stride=patch_size setting.

Why: the block/square pattern in the 2D features is the ViT patch grid (2026-08-27 diagnosis --
16x12 valid cells for a 1080x810 photo, one cell covering ~67.5x67.5 px, then bilinearly
upsampled). Following Amir et al. 2021 ("Deep ViT Features as Dense Visual Descriptors"), the
patch-embedding conv's stride can be set BELOW its kernel size at test time, with no retraining:
every 14x14 window the model sees is still a patch it was trained on, they are simply taken more
densely. Adjacent tokens then share input, so the field varies smoothly instead of stepping at
cell boundaries -- which is what removes blockiness, as opposed to merely making the blocks
smaller (raising the input resolution does the latter).

DINO ONLY -- no SD, no fusion (Tal's call: isolate the problem to DINO first).

Run:
cd /workspace/bsb && python analysis/compare_dino_patch_stride.py \
  --img_path ./images/guitar_01.png --strides 14 7 \
  --save_dir ./demo/guitar/features_analyze/dino_stride_trick/
"""
import argparse
import math
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


def _interpolate_pos_encoding_any_stride(self, x, w, h):
    """Drop-in replacement for dinov2's DinoVisionTransformer.interpolate_pos_encoding.

    The original derives the token grid as `w // self.patch_size`, which is only correct when
    stride == patch_size, and then asserts that against the real grid -- so it raises as soon as
    the stride is lowered. This version reads the grid off the conv geometry instead, and
    interpolates the position embeddings to it with an explicit size= (no scale_factor rounding,
    hence no need for the original's interpolate_offset fudge).
    """
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed
    pos_embed = self.pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]

    k = self.patch_embed.proj.kernel_size[0]
    s = self.patch_embed.proj.stride[0]
    h0 = (h - k) // s + 1
    w0 = (w - k) // s + 1

    sqrt_N = int(math.sqrt(N))
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, sqrt_N, sqrt_N, dim).permute(0, 3, 1, 2),
        size=(h0, w0), mode='bicubic', align_corners=False,
        antialias=getattr(self, 'interpolate_antialias', False))
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)


def set_patch_stride(dino_model, stride):
    """Set the patch-embedding conv's stride (test-time only, no weights touched)."""
    vit = dino_model.model
    vit.patch_embed.proj.stride = (stride, stride)
    vit.interpolate_pos_encoding = _interpolate_pos_encoding_any_stride.__get__(vit, type(vit))


def extract_raw_grid(dino_model, image, target_size, stride):
    """Returns (features [C, gh, gw] on the padded square grid, input_size, original_size)."""
    set_patch_stride(dino_model, stride)
    square, input_size = preprocess_image(image, target_size)
    with torch.no_grad():
        feats = dino_model(square)
    return feats[0].float().cpu(), tuple(input_size), image.shape[:2]


def valid_crop(feats, input_size, target_size, kernel, stride):
    """Drop the columns/rows that only ever saw zero padding."""
    vh = (input_size[0] - kernel) // stride + 1
    vw = (input_size[1] - kernel) // stride + 1
    return feats[:, :vh, :vw]


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    image = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    print(f'photo: {image.shape[:2]}')

    dino = DINOWrapper(device=device, small=True, target_size=args.dino_target_size).to(device)
    kernel = dino.patch_size()

    grids = {}
    for s in args.strides:
        feats, input_size, orig = extract_raw_grid(dino, image, args.dino_target_size, s)
        valid = valid_crop(feats, input_size, args.dino_target_size, kernel, s)
        grids[s] = valid
        cell = orig[0] / valid.shape[1]
        print(f'stride {s:>2}: padded grid {tuple(feats.shape[1:])}, valid {tuple(valid.shape[1:])}, '
              f'{valid.shape[1] * valid.shape[2]:>5} tokens, one cell ~{cell:.1f} px, '
              f'support {orig[0] * kernel / input_size[0]:.1f} px')
        torch.save({'features': valid, 'stride': s, 'kernel': kernel,
                    'input_size': input_size, 'original_size': orig},
                   os.path.join(args.save_dir, f'guitar_01_dino_raw_grid_stride{s}.pth'))

    # Shared PCA basis, fit on the finest grid so colours are comparable across strides.
    finest = grids[min(args.strides)]
    C = finest.shape[0]
    flat = finest.reshape(C, -1).T
    mu = flat.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(flat - mu, q=3)

    def to_rgb(x):
        p = ((x.reshape(C, -1).T - mu) @ V).T.reshape(3, x.shape[1], x.shape[2])
        lo = p.amin(dim=(1, 2), keepdim=True); hi = p.amax(dim=(1, 2), keepdim=True)
        return ((p - lo) / (hi - lo)).permute(1, 2, 0).numpy()

    n = len(args.strides)
    fig, ax = plt.subplots(2, n + 1, figsize=(5.0 * (n + 1), 9.5))
    ax[0, 0].imshow(image); ax[0, 0].set_title(f'photo {image.shape[0]}x{image.shape[1]}')
    ax[1, 0].axis('off')
    for i, s in enumerate(args.strides):
        g = grids[s]
        up = F.interpolate(g[None], size=image.shape[:2], mode='bilinear', align_corners=False)[0]
        ax[0, i + 1].imshow(to_rgb(g), interpolation='nearest')
        ax[0, i + 1].set_title(f'stride {s}  ->  raw grid {g.shape[1]}x{g.shape[2]}\n'
                               f'{g.shape[1] * g.shape[2]} tokens')
        ax[1, i + 1].imshow(to_rgb(up))
        ax[1, i + 1].set_title(f'stride {s}, bilinear -> full res\n'
                               f'{"production setting" if s == kernel else "overlapping patches"}')
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle('DINO patch stride: lowering it below the patch size makes neighbouring tokens '
                 'overlap,\nso the feature field varies smoothly instead of stepping at cell boundaries',
                 fontsize=14)
    fig.tight_layout()
    out = os.path.join(args.save_dir, 'dino_stride_trick_comparison.png')
    fig.savefig(out, dpi=105, bbox_inches='tight')
    print('saved', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--img_path', type=str, default='./images/guitar_01.png')
    p.add_argument('--save_dir', type=str, required=True)
    p.add_argument('--strides', type=int, nargs='+', default=[14, 7])
    p.add_argument('--dino_target_size', type=int, default=224)
    main(p.parse_args())
