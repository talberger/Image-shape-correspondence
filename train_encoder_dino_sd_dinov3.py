"""
Stage C (bsb env) of the DINOv2->DINOv3 swap, render/phase-1-target side --
straight fork of train_encoder_dino_sd.py with ONE change: dino_raw comes from
dinov3_repo/extract_dinov3_render_features.py's cached output (Stage B, run in
the isolated `dinov3` env -- bsb's own torch 1.12/transformers 4.34 can't load
DINOv3) instead of a live DINOWrapper(small=True) call. SD extraction, fusion,
postprocessing, and the {'model_f','elev','azim'} output schema are all
UNCHANGED, so this is a drop-in for train_encoder_dfd_crossview.py
--feature_model_type dino_sd_dinov3 (that flag is fully generic already --
see PROJECT_NOTES.md item 3 / train_encoder_dfd_crossview.py's own feat_dir
construction, no code changes needed there).

Prerequisite: extract_dinov3_render_features.py already run (dinov3 env) against
the SAME renders this script reads, producing
<encoder_data_dir>/dinov3_raw/<render_res>/target_dinov3_raw_{i}.pt.

Run:
cd $DISC_ROOT && python train_encoder_dino_sd_dinov3.py \
  --encoder_data_dir ./data/guitar/encoder_data_frontcentered_dinosd \
  --dinov3_raw_dir ./data/guitar/encoder_data_frontcentered_dinosd/dinov3_raw/224 \
  --render_res 224
"""
import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.dino_sd_fusion import SD_INPUT_RES, SDFeatureExtractor, fuse_dino_sd, sd_square
from modules.utils import device


def postprocess_features(features, input_size, original_size, target_size=224):
    """Same 3 ops as modules/dino.py's DINOWrapper.postprocess_features, pulled out
    standalone so this script doesn't need to load the (unrelated, DINOv2) model
    just for this method."""
    features = F.interpolate(features, (target_size, target_size), mode='bilinear', align_corners=False)
    features = features[..., :input_size[0], :input_size[1]]
    features = F.interpolate(features, original_size, mode='bilinear', align_corners=False)
    return features


def encode_random_views_dino_sd_dinov3(args, sd_extractor):
    print('Generate encoded image features for model dino_sd_dinov3')
    save_dir = os.path.join(args.encoder_data_dir, 'dino_sd_dinov3_f', str(args.render_res))
    os.makedirs(save_dir, exist_ok=True)

    render_dir = os.path.join(args.encoder_data_dir, str(args.render_res))
    data = torch.load(os.path.join(render_dir, 'random_viewing_angles.pt'))
    elev_list, azim_list = data['elev'], data['azim']

    for i in tqdm(range(len(elev_list))):
        elev, azim = elev_list[i], azim_list[i]
        save_path = os.path.join(save_dir, f'target_dino_sd_dinov3_f_{i}.pt')
        if os.path.exists(save_path):
            continue

        raw = torch.load(os.path.join(args.dinov3_raw_dir, f'target_dinov3_raw_{i}.pt'), map_location='cpu')
        assert float(raw['elev']) == float(elev) and float(raw['azim']) == float(azim), \
            f'view {i}: dinov3_raw cache elev/azim mismatch vs random_viewing_angles.pt'
        dino_raw = raw['dino_raw'].to(device)
        input_size = raw['input_size']

        image_path = os.path.join(render_dir, f'target_save_{i}.png')
        import cv2
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_t = torch.as_tensor(image, device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
        image_sd = sd_square(image_t)
        sd_raw = sd_extractor(image_sd, noise_index=int(i))

        fused_raw = fuse_dino_sd(sd_raw, dino_raw)
        model_f = postprocess_features(fused_raw, input_size=input_size, original_size=(64, 64))

        torch.save({'model_f': model_f.cpu(), 'elev': elev, 'azim': azim}, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_data_dir', type=str, required=True)
    parser.add_argument('--dinov3_raw_dir', type=str, required=True)
    parser.add_argument('--render_res', type=int, default=224)
    args = parser.parse_args()

    sd_extractor = SDFeatureExtractor(device=device)
    encode_random_views_dino_sd_dinov3(args, sd_extractor)
