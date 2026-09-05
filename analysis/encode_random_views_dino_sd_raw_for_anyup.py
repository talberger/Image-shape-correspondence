"""
Render-side Stage A of the AnyUp integration (see PROJECT_NOTES.md, "AnyUp"
entry) -- extends the swap from the real-photo path (item #1) to the
phase-1 training-TARGET path (item #2/#6, Tal's request). Runs in the bsb
env. Mirrors train_encoder_dino_sd.py's encode_random_views_dino_sd() exactly
up through fuse_dino_sd(sd_raw, dino_raw), then STOPS -- does not call
postprocess_features. Unlike the real-photo case, renders are already square
(render_res == target_image_size, DINOWrapper.pad is a no-op) so there is NO
padding/crop complexity here -- Stage B can read each render PNG directly off
disk (already shared across envs, just a file path) and run ONE direct AnyUp
call per view straight to the 64x64 target, no capping/chunking needed either
(64x64x2304ch output is tiny, ~37.7MB).

Saves ONLY fused_raw + elev/azim per view (not the image -- Stage B re-reads
target_save_{i}.png directly, no bsb imports needed for that, avoids
duplicating ~1000 images across the handoff).

Run (bsb env, mirrors train_encoder_dino_sd.py's own --encoder_data_dir):
cd /workspace/bsb && python analysis/encode_random_views_dino_sd_raw_for_anyup.py \\
  --encoder_data_dir ./data/guitar/encoder_data_frontcentered_dinosd
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder import preprocess_image
from modules.dino import DINOWrapper
from modules.dino_sd_fusion import SD_INPUT_RES, SDFeatureExtractor, fuse_dino_sd, sd_square
from modules.utils import device


def main(args):
    save_dir = os.path.join(args.encoder_data_dir, 'dino_sd_fused_raw_for_anyup', str(args.render_res))
    os.makedirs(save_dir, exist_ok=True)

    data = torch.load(os.path.join(args.encoder_data_dir, str(args.render_res), 'random_viewing_angles.pt'))
    elev_list, azim_list = data['elev'], data['azim']

    print('loading DINOWrapper...')
    dino_model = DINOWrapper(device=device, small=True, target_size=args.target_image_size).to(device)
    print('loading SD components...')
    sd_extractor = SDFeatureExtractor(device=device)

    for i in tqdm(np.arange(len(elev_list))):
        elev, azim = elev_list[i], azim_list[i]
        save_path = os.path.join(save_dir, f'target_fused_raw_{i}.pt')
        if os.path.exists(save_path):
            continue

        image = cv2.imread(os.path.join(args.encoder_data_dir, str(args.render_res), f'target_save_{i}.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_image_square, input_size = preprocess_image(image, args.target_image_size)
        # renders are square (render_res == target_image_size) -> input_size == (target_image_size,)*2,
        # i.e. DINOWrapper.pad is a no-op here (confirmed, not assumed) -- no padding/crop needed downstream.
        assert tuple(input_size) == (args.target_image_size, args.target_image_size), (
            f'expected square render with no padding, got input_size={input_size}')
        with torch.no_grad():
            dino_raw = dino_model(input_image_square)

        image_t = torch.as_tensor(image, device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
        image_sd = sd_square(image_t)
        sd_raw = sd_extractor(image_sd, noise_index=int(i))

        fused_raw = fuse_dino_sd(sd_raw, dino_raw)  # [1, 2304, 16, 16]
        torch.save({'fused_raw': fused_raw.cpu(), 'elev': elev, 'azim': azim}, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_data_dir', type=str, required=True)
    parser.add_argument('--render_res', type=int, default=224)
    parser.add_argument('--target_image_size', type=int, default=224)
    args = parser.parse_args()
    main(args)
