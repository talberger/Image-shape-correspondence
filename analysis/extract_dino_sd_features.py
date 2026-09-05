"""
Extract fused DINO+SD per-pixel features for one image -- drop-in
[H, W, C] float32 .pth, same convention as
modules/best_segmentation_buddies.py's get_dino_features output (which this
does NOT import from or modify -- see modules/dino_sd_fusion.py's docstring
for the full design). Used both for the real photo (this script, run
directly) and for every synthetic phase-1 render target
(train_encoder_dino_sd.py's encode_random_views_dino_sd(), which calls the
same extraction logic).

Run (Stage 1 POC, real photo):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/extract_dino_sd_features.py \
  --img_path ./images/guitar_01.png \
  --save_dir ./demo/guitar/features_analyze/dino_sd_poc/ \
  --output_path ./demo/guitar/features_analyze/dino_sd_poc/guitar_01_pix_dino_sd_features.pth
"""
import argparse
import os
import sys

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder import preprocess_image
from analysis.compare_dino_patch_stride import set_patch_stride
from modules.dino import DINOWrapper
from modules.dino_sd_fusion import (SD_INPUT_RES, SDFeatureExtractor, crop_valid, dino_only,
                                    fuse_dino_sd, fuse_dino_sd_common_grid,
                                    sd_frame_aspect_nopad, sd_square, sd_squash,
                                    sd_square_replicate)
from modules.utils import device


def extract_fused_features(image, dino_model, sd_extractor, dino_target_size,
                           save_raw_path=None, cpu_postprocess=False,
                           fix_grid_alignment=False, patch_stride=None,
                           sd_frame='square_pad', fuse_mode='dino_grid', common_grid=64,
                           dino_only_mode=False):
    """image: HxWx3 uint8 RGB numpy array. Returns [H, W, C_fused] float32
    CPU tensor, same convention as best_segmentation_buddies.get_dino_features.

    save_raw_path: if given, also dump the PRE-UPSCALE fused map -- the tensor exactly as
    fuse_dino_sd() returns it, on DINO's raw patch grid, before postprocess_features()
    bilinearly blows it up to the full image size. Added 2026-08-27 to investigate the
    block/square pattern visible in the 2D features (that pattern is this grid).
    """
    original_size = image.shape[:2]

    input_image_square, input_size = preprocess_image(image, dino_target_size)
    with torch.no_grad():
        dino_raw = dino_model(input_image_square)  # [1, 1024, h_p, w_p], patch grid

    if dino_only_mode:
        # The DINO half alone (1024 ch). SD is never run, so sd_frame/fuse_mode are irrelevant
        # here by construction -- the caller is responsible for not implying otherwise.
        fused_raw = dino_only(dino_raw)
        if save_raw_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_raw_path)), exist_ok=True)
            torch.save(fused_raw[0].permute(1, 2, 0).contiguous().cpu(), save_raw_path)
            print(f'saved PRE-UPSCALE DINO-only map {tuple(fused_raw.shape)} to {save_raw_path}')
        if cpu_postprocess:
            dino_model = dino_model.cpu()
            fused_raw = fused_raw.cpu()
        fused = dino_model.postprocess_features(fused_raw, input_size=input_size,
                                                original_size=original_size,
                                                fix_grid_alignment=fix_grid_alignment,
                                                patch_stride=patch_stride)
        return fused[0].permute(1, 2, 0).contiguous().cpu()

    image_t = torch.as_tensor(image, device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    # --- Tal edit, 2026-08-30: SD's framing and the fusion grid are now independent knobs.
    # Defaults reproduce the production path exactly (square_pad + dino_grid).
    if sd_frame == 'aspect_nopad':
        image_sd = sd_frame_aspect_nopad(image_t)
    elif sd_frame == 'squash':
        image_sd = sd_squash(image_t)
    elif sd_frame == 'square_pad_replicate':
        image_sd = sd_square_replicate(image_t)
    else:
        image_sd = sd_square(image_t)
    sd_raw = sd_extractor(image_sd)  # [1, 1280, h_sd, w_sd]

    if fuse_mode == 'common_grid':
        # Each branch is reduced to its own VALID region, then both go onto one grid big enough to
        # keep SD's detail (fuse_dino_sd would downsample SD onto DINO's much coarser grid).
        if sd_frame == 'squash':
            raise ValueError("--fuse_mode common_grid does not support --sd_frame squash: the "
                             "squashed frame has no padding to crop away, and its cells do not "
                             "correspond to image regions at a constant scale.")
        if sd_frame == 'aspect_nopad':
            sd_valid = sd_raw                      # no padding was added, every cell is real
        else:
            sd_scale = SD_INPUT_RES / max(original_size)
            sd_valid = crop_valid(sd_raw, (round(original_size[0] * sd_scale),
                                           round(original_size[1] * sd_scale)),
                                  (SD_INPUT_RES, SD_INPUT_RES))
        oh = common_grid if original_size[0] >= original_size[1] else \
            max(1, round(common_grid * original_size[0] / original_size[1]))
        ow = common_grid if original_size[1] > original_size[0] else \
            max(1, round(common_grid * original_size[1] / original_size[0]))
        # DINO's half goes through Itai's own postprocess_features, so the padding crop AND the
        # gridalign correction are reused rather than reimplemented -- that code is untouched.
        dino_at_grid = dino_model.postprocess_features(dino_raw, input_size=input_size,
                                                       original_size=(oh, ow),
                                                       fix_grid_alignment=fix_grid_alignment,
                                                       patch_stride=patch_stride)
        fused_raw = fuse_dino_sd_common_grid(sd_valid, dino_at_grid, (oh, ow))
    else:
        fused_raw = fuse_dino_sd(sd_raw, dino_raw)  # [1, C_fused, h_p, w_p], on DINO's grid
    if save_raw_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_raw_path)), exist_ok=True)
        torch.save({'fused_raw': fused_raw[0].cpu(), 'dino_raw': dino_raw[0].cpu(),
                    'sd_raw': sd_raw[0].float().cpu(), 'input_size': tuple(input_size),
                    'original_size': tuple(original_size),
                    'dino_target_size': dino_target_size}, save_raw_path)
        print(f'  dino patch grid  : {tuple(dino_raw.shape[-2:])}  ({dino_raw.shape[1]} ch)')
        print(f'  sd raw grid      : {tuple(sd_raw.shape[-2:])}  ({sd_raw.shape[1]} ch, '
              f'downsampled onto DINO\'s grid by fuse_dino_sd)')
        print(f'  fused raw grid   : {tuple(fused_raw.shape[-2:])}  ({fused_raw.shape[1]} ch)')
        print(f'  -> upscaled to   : {original_size}')
        print(f'  saved pre-upscale features to {save_raw_path}')
    # The full-res [2304, H, W] tensor is ~8GB -- on a 16GB card that OOMs with DINO+SD still
    # resident, so allow doing the (purely elementwise-in-space) upsample on CPU instead. Off by
    # default so the production path stays bit-identical.
    if cpu_postprocess:
        fused_raw = fused_raw.cpu()
    if fuse_mode == 'common_grid':
        # already cropped and aligned above; this is a plain upsample to full resolution.
        fused = torch.nn.functional.interpolate(fused_raw, size=original_size, mode='bilinear',
                                                align_corners=False)
    else:
        fused = dino_model.postprocess_features(fused_raw, input_size=input_size,
                                               original_size=original_size,
                                               fix_grid_alignment=fix_grid_alignment,
                                               patch_stride=patch_stride)
    return fused[0].permute(1, 2, 0).contiguous().cpu()  # [H, W, C_fused]


def main(args):
    image = cv2.imread(args.img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print('loading DINOWrapper...')
    dino_model = DINOWrapper(device=device, small=True, target_size=args.dino_target_size).to(device)
    if args.dino_patch_stride is not None:
        # Overlapping-patch extraction (Amir et al. 2021) -- test-time only, no weights touched.
        # See analysis/compare_dino_patch_stride.py for the mechanism and the grid arithmetic.
        set_patch_stride(dino_model, args.dino_patch_stride)
        print(f'  DINO patch stride overridden: {dino_model.patch_size()} -> {args.dino_patch_stride}')
    if args.dino_only:
        print('--dino_only: skipping Stable Diffusion entirely (1024-ch DINO features)')
        sd_extractor = None
    else:
        print('loading SD components (first run downloads several GB, cached under $HF_HOME after)...')
        sd_extractor = SDFeatureExtractor(device=device)

    fused = extract_fused_features(image, dino_model, sd_extractor, args.dino_target_size,
                                   sd_frame=args.sd_frame, fuse_mode=args.fuse_mode,
                                   common_grid=args.common_grid,
                                   save_raw_path=args.save_raw_path,
                                   cpu_postprocess=args.cpu_postprocess,
                                   fix_grid_alignment=args.fix_grid_alignment,
                                   patch_stride=args.dino_patch_stride,
                                   dino_only_mode=args.dino_only)
    print(f'fused features: shape={tuple(fused.shape)}, dtype={fused.dtype}')

    output_path = args.output_path or os.path.join(
        args.save_dir, f'{os.path.splitext(os.path.basename(args.img_path))[0]}_pix_dino_sd_features.pth')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(fused, output_path)
    print(f'saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None,
                        help='defaults to <save_dir>/<img_name>_pix_dino_sd_features.pth')
    parser.add_argument('--save_raw_path', type=str, default=None,
                        help='also dump the pre-upscale fused map (on DINO\'s raw patch grid)')
    parser.add_argument('--dino_patch_stride', type=int, default=None,
                        help='lower the patch-embedding stride below the patch size (overlapping '
                             'patches, denser grid, no retraining); default None = unchanged')
    parser.add_argument('--fix_grid_alignment', action='store_true',
                        help='place tokens at their true receptive-field centres when upsampling '
                             '(required for any stride != patch size; see modules/dino.py)')
    parser.add_argument('--cpu_postprocess', action='store_true',
                        help='do the final upsample on CPU (the full-res tensor is ~8GB)')
    parser.add_argument('--sd_frame', choices=['square_pad', 'square_pad_replicate', 'aspect_nopad', 'squash'],
                        default='square_pad',
                        help="square_pad = sd_square (production). square_pad_replicate = same "
                             "frame, edge-replicated pad instead of zeros (2026-09-03). "
                             "aspect_nopad = true aspect, no "
                             "padding at all, dims rounded to a multiple of 8. squash = the "
                             "pre-2026-08-27 framing, image stretched to a square.")
    parser.add_argument('--dino_only', action='store_true',
                        help='drop the SD branch entirely: emit the DINO half of the fusion '
                             '(1024 ch, not 2304). SD is never loaded or run, and '
                             '--sd_frame/--fuse_mode become no-ops. Added 2026-09-01 for the '
                             'DINO vs DINO+SD ablation on the current recipe.')
    parser.add_argument('--fuse_mode', choices=['dino_grid', 'common_grid'], default='dino_grid',
                        help="dino_grid = fuse_dino_sd (production; SD is downsampled onto DINO's "
                             "grid). common_grid = crop each side to its valid region, then fuse on "
                             "a shared grid that keeps SD's resolution.")
    parser.add_argument('--common_grid', type=int, default=64,
                        help='long side of the shared fusion grid, --fuse_mode common_grid only')
    parser.add_argument('--dino_target_size', type=int, default=224,
                        help='matches best_segmentation_buddies.get_dino_features\'s own default')

    args = parser.parse_args()
    main(args)
