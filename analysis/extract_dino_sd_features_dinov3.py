"""
Stage C (bsb env) of the DINOv2->DINOv3 swap, real-photo/query side -- fork of
extract_dino_sd_features.py with dino_raw coming from
dinov3_repo/extract_dinov3_photo_features.py's cached output (Stage B, isolated
`dinov3` env) instead of a live DINOWrapper call. SD extraction, fusion, and the
[H, W, C_fused] output schema are all unchanged -- drop-in replacement for the
plain-DINOv2 pix-features file wherever a script takes --pix_dino_features_path.

Prerequisite: extract_dinov3_photo_features.py already run (dinov3 env).

Run:
cd $DISC_ROOT && python analysis/extract_dino_sd_features_dinov3.py \
  --dinov3_raw_path ./demo/guitar/features_analyze/dino_sd_dinov3_poc/guitar_01_dinov3_raw.pth \
  --img_path ./images/guitar_01.png \
  --output_path ./demo/guitar/features_analyze/dino_sd_dinov3_poc/guitar_01_pix_dino_sd_dinov3_features.pth
"""
import argparse
import os
import sys

import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.dino_sd_fusion import SD_INPUT_RES, SDFeatureExtractor, fuse_dino_sd, sd_square
from modules.utils import device


def postprocess_features(features, input_size, original_size, target_size=224):
    features = F.interpolate(features, (target_size, target_size), mode='bilinear', align_corners=False)
    features = features[..., :input_size[0], :input_size[1]]
    features = F.interpolate(features, original_size, mode='bilinear', align_corners=False)
    return features


def main(args):
    raw = torch.load(args.dinov3_raw_path, map_location='cpu')
    dino_raw = raw['dino_raw'].to(device)
    input_size = raw['input_size']
    original_size = raw['original_size']

    image = cv2.imread(args.img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert tuple(image.shape[:2]) == tuple(original_size), \
        f'{args.img_path} shape {image.shape[:2]} != dinov3_raw original_size {original_size}'

    print('loading SD components...')
    sd_extractor = SDFeatureExtractor(device=device)

    image_t = torch.as_tensor(image, device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    image_sd = sd_square(image_t)
    sd_raw = sd_extractor(image_sd)

    fused_raw = fuse_dino_sd(sd_raw, dino_raw).cpu()
    # the final upsample (2304ch x native ~1080x810) doesn't fit on this pod's GPU
    # alongside the still-loaded SD model (7.5GB attempted alloc vs ~4.8GB free) --
    # same class of issue as AnyUp's own NHWC-bilinear/INT_MAX memory workarounds
    # (see PROJECT_NOTES.md); fused_raw itself is tiny ([1,2304,14,14]), so just
    # finish the interpolate chain on CPU instead of chunking on GPU.
    del sd_extractor
    torch.cuda.empty_cache()
    fused = postprocess_features(fused_raw, input_size=input_size, original_size=original_size)
    fused = fused[0].permute(1, 2, 0).contiguous()  # [H, W, C_fused]
    print(f'fused features: shape={tuple(fused.shape)}, dtype={fused.dtype}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(fused, args.output_path)
    print(f'saved to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dinov3_raw_path', type=str, required=True)
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    main(parser.parse_args())
