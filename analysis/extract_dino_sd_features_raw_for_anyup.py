"""
Stage A of the AnyUp integration (see PROJECT_NOTES.md, "AnyUp" entry) -- runs
in the bsb env (python 3.9). Thin variant of extract_dino_sd_features.py's
extract_fused_features(): identical up through fuse_dino_sd(sd_raw, dino_raw),
then STOPS -- does NOT call dino_model.postprocess_features (the bilinear
upsample being replaced). Saves the raw fused grid + native image instead, for
Stage B (a separate, isolated `anyup` env, python 3.12/torch 2.9) to pick up.
Cross-env handoff via a plain .pt file only -- same disposable-isolated-env
pattern as the (since-removed) geoaware env; the two envs never import each
other.

Run (bsb env):
cd /workspace/bsb && python analysis/extract_dino_sd_features_raw_for_anyup.py \\
  --img_path ./images/guitar_01.png \\
  --output_path ./demo/guitar/features_analyze/dino_sd_poc/guitar_01_dino_sd_raw_for_anyup.pth
"""
import argparse
import os
import sys

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder import preprocess_image
from modules.dino import DINOWrapper
from modules.dino_sd_fusion import SD_INPUT_RES, SDFeatureExtractor, fuse_dino_sd, sd_square
from modules.utils import device


def extract_raw_fused_features(image, dino_model, sd_extractor, dino_target_size):
    """image: HxWx3 uint8 RGB numpy array. Returns (fused_raw, input_image_square,
    input_size). fused_raw: [1, C_fused, h_p, w_p] on DINO's raw patch grid
    (16x16 for a 224 dino_target_size) -- same tensor
    extract_dino_sd_features.py's postprocess_features call would otherwise
    upsample; here we stop one step earlier and hand it to Stage B instead.
    input_image_square/input_size: the exact 224x224 padded-square image DINO
    itself consumed (raw [0,1], NOT ImageNet-normalized -- DINOWrapper.forward
    does that internally) and the true pre-padding size within it -- Stage B
    needs these to mirror the original algorithm's own two-step upsample
    (16->224 guided by this square, crop, then 224(cropped)->native guided by
    the true native photo) instead of jumping straight to native resolution.
    """
    input_image_square, input_size = preprocess_image(image, dino_target_size)
    with torch.no_grad():
        dino_raw = dino_model(input_image_square)  # [1, 1024, h_p, w_p], patch grid

    image_t = torch.as_tensor(image, device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    image_sd = sd_square(image_t)
    sd_raw = sd_extractor(image_sd)  # [1, 1280, h_sd, w_sd]

    fused_raw = fuse_dino_sd(sd_raw, dino_raw)  # [1, C_fused, h_p, w_p], on DINO's grid
    return fused_raw.cpu(), input_image_square.cpu(), input_size


def main(args):
    image = cv2.imread(args.img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]

    print('loading DINOWrapper...')
    dino_model = DINOWrapper(device=device, small=True, target_size=args.dino_target_size).to(device)
    print('loading SD components...')
    sd_extractor = SDFeatureExtractor(device=device)

    fused_raw, input_image_square, input_size = extract_raw_fused_features(
        image, dino_model, sd_extractor, args.dino_target_size)
    print(f'fused_raw: shape={tuple(fused_raw.shape)}, dtype={fused_raw.dtype}, '
          f'input_image_square: shape={tuple(input_image_square.shape)}, input_size={input_size}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save({
        'fused_raw': fused_raw,                     # [1, C_fused, h_p, w_p], CPU float32
        'image_native': image,                      # HxWx3 uint8 RGB, untouched
        'original_size': original_size,              # (H, W)
        'input_image_square': input_image_square,    # [1,3,224,224] float [0,1], CPU, un-normalized
        'input_size': tuple(int(x) for x in input_size),  # true (H,W) within the padded square
    }, args.output_path)
    print(f'saved to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--dino_target_size', type=int, default=224,
                        help='matches extract_dino_sd_features.py\'s own default')

    args = parser.parse_args()
    main(args)
