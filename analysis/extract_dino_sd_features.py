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
from modules.dino import DINOWrapper
from modules.dino_sd_fusion import SD_INPUT_RES, SDFeatureExtractor, fuse_dino_sd
from modules.utils import device


def extract_fused_features(image, dino_model, sd_extractor, dino_target_size):
    """image: HxWx3 uint8 RGB numpy array. Returns [H, W, C_fused] float32
    CPU tensor, same convention as best_segmentation_buddies.get_dino_features.
    """
    original_size = image.shape[:2]

    input_image_square, input_size = preprocess_image(image, dino_target_size)
    with torch.no_grad():
        dino_raw = dino_model(input_image_square)  # [1, 1024, h_p, w_p], patch grid

    image_t = torch.as_tensor(image, device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    image_sd = torch.nn.functional.interpolate(
        image_t, size=(SD_INPUT_RES, SD_INPUT_RES), mode='bilinear', align_corners=False)
    sd_raw = sd_extractor(image_sd)  # [1, 1280, h_sd, w_sd]

    fused_raw = fuse_dino_sd(sd_raw, dino_raw)  # [1, C_fused, h_p, w_p], on DINO's grid
    fused = dino_model.postprocess_features(fused_raw, input_size=input_size, original_size=original_size)
    return fused[0].permute(1, 2, 0).contiguous().cpu()  # [H, W, C_fused]


def main(args):
    image = cv2.imread(args.img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print('loading DINOWrapper...')
    dino_model = DINOWrapper(device=device, small=True, target_size=args.dino_target_size).to(device)
    print('loading SD components (first run downloads several GB, cached under $HF_HOME after)...')
    sd_extractor = SDFeatureExtractor(device=device)

    fused = extract_fused_features(image, dino_model, sd_extractor, args.dino_target_size)
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
    parser.add_argument('--dino_target_size', type=int, default=224,
                        help='matches best_segmentation_buddies.get_dino_features\'s own default')

    args = parser.parse_args()
    main(args)
