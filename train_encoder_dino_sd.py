"""
Phase-1 training with fused DINO+SD features instead of DINO alone. New,
additive script -- does NOT modify encoder.py. Imports
generate_random_views/train_encoder/preprocess_image (all unchanged) plus
Encoder/EncoderDataset (via train_encoder, also unchanged) directly from the
baseline; the only new code is encode_random_views_dino_sd() below, which
mirrors encoder.py's own encode_random_views (model_type=='dino' branch)
but calls the fused extractor (modules/dino_sd_fusion.py) per rendered view
instead of dino_features_encoder.

See PROJECT_NOTES.md (roadmap item 1) / the approved plan for the full
design: why fused features (SHIC cites this fusion technique but its own
repo doesn't contain the code), why no PCA (sd-dino's own PCA is fit per
image PAIR, doesn't fit ~1000 renders + 1 photo needing one shared space;
Diff3F -- same task as this project -- sidesteps it with L2-normalize +
fixed-weight concat instead, no fitting step at all), why unet.up_blocks[1]
(matches Diff3F's own modified UNet exactly, confirmed via a Stage-0 smoke
test).

model_type is fixed to 'dino_sd' (not a CLI choice) -- EncoderDataset
(modules/dataset.py) is already fully generic over this string, no changes
needed there.

Run (mirrors encoder_dino_frontcentered's own config; --n_views/--width/
--n_classes are this run's own, since the fused feature dim (2304 = 1280 SD
+ 1024 DINO) differs from plain DINO's 1024):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python train_encoder_dino_sd.py \
  --encoder_data_dir ./data/guitar/encoder_data_frontcentered_dinosd \
  --encoder_model_dir ./experiments/guitar/encoder_dino_sd_frontcentered \
  --view_sampling random --frontview_center 0 1.5707963267948966 --frontview_std 4 \
  --n_views 1000 --width 2304 --n_classes 2304 \
  --generate_random_views 1 --encode_random_views 1 --start_training 1

Stage-2 sanity check (this project's own "measure before the full
commitment" convention): same command with --n_views 50.
"""
import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from encoder import generate_random_views, preprocess_image, train_encoder
from analysis.compare_dino_patch_stride import set_patch_stride
from modules.dino import DINOWrapper
from modules.dino_sd_fusion import (SD_INPUT_RES, SDFeatureExtractor, dino_only, fuse_dino_sd,
                                    fuse_dino_sd_common_grid, sd_frame_aspect_nopad, sd_square)
from modules.utils import device, loadmesh


def encode_random_views_dino_sd(args, dino_model, sd_extractor):
    """Mirrors encoder.py's encode_random_views (model_type=='dino' branch)
    exactly, except the per-view feature comes from the fused DINO+SD
    extractor instead of dino_features_encoder. Saves under
    <encoder_data_dir>/dino_sd_f/<render_res>/target_dino_sd_f_{i}.pt, same
    {'model_f', 'elev', 'azim'} dict EncoderDataset already reads generically.
    """
    print('Generate encoded image features for model dino_sd')
    name = args.feature_cache_name
    save_dir = os.path.join(args.encoder_data_dir, f'{name}_f', str(args.render_res))
    os.makedirs(save_dir, exist_ok=True)

    data = torch.load(os.path.join(args.encoder_data_dir, str(args.render_res), 'random_viewing_angles.pt'))
    elev_list, azim_list = data['elev'], data['azim']

    for i in tqdm(np.arange(len(elev_list))):
        elev, azim = elev_list[i], azim_list[i]
        save_path = os.path.join(save_dir, f'target_{name}_f_{i}.pt')
        if os.path.exists(save_path):
            continue

        image = cv2.imread(os.path.join(args.encoder_data_dir, str(args.render_res), f'target_save_{i}.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_image_square, input_size = preprocess_image(image, args.target_image_size)
        with torch.no_grad():
            dino_raw = dino_model(input_image_square)

        if args.dino_only:
            # The SD branch is not merely zeroed -- it is never computed. SD is ~80% of this
            # loop's cost (2026-08-28 timing: 672 ms of ~720 ms per view), so the DINO-only cache
            # is both the ablation arm and much cheaper to build.
            fused_raw = dino_only(dino_raw)
            model_f = dino_model.postprocess_features(
                fused_raw, input_size=input_size, original_size=(64, 64),
                fix_grid_alignment=args.fix_grid_alignment,
                patch_stride=args.dino_patch_stride)
            if args.cache_dtype == 'float16':
                model_f = model_f.half()
            torch.save({'model_f': model_f, 'elev': elev, 'azim': azim}, save_path)
            continue

        image_t = torch.as_tensor(image, device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
        # --- Tal edit, 2026-08-30 ---
        # Renders are SQUARE, so square_pad and aspect_nopad produce the identical 512x512 input
        # here; the framing flag only matters for the photo. --fuse_mode common_grid, however,
        # DOES change this cache: SD stops being downsampled onto DINO's 16x16 grid and is
        # upsampled from its own 32x32 straight to the 64x64 phase-1 target. A cache built with a
        # different --fuse_mode is NOT interchangeable with the production one.
        image_sd = (sd_frame_aspect_nopad(image_t) if args.sd_frame == 'aspect_nopad'
                    else sd_square(image_t))
        sd_raw = sd_extractor(image_sd, noise_index=int(i))

        if args.fuse_mode == 'common_grid':
            dino_at_grid = dino_model.postprocess_features(
                dino_raw, input_size=input_size, original_size=(64, 64),
                fix_grid_alignment=args.fix_grid_alignment,
                patch_stride=args.dino_patch_stride)
            model_f = fuse_dino_sd_common_grid(sd_raw, dino_at_grid, (64, 64))
            fused_raw = None
        else:
            fused_raw = fuse_dino_sd(sd_raw, dino_raw)
        # original_size=(64,64) hardcoded -- matches encoder.py's own
        # dino_features_encoder exactly: phase-1 always supervises at 64x64,
        # independent of the render/DINO-input resolution.
        if fused_raw is not None:
            model_f = dino_model.postprocess_features(
                fused_raw, input_size=input_size, original_size=(64, 64),
                fix_grid_alignment=args.fix_grid_alignment,
                patch_stride=args.dino_patch_stride)

        # --- cache dtype (added 2026-08-28) ---
        # The cached target is [1, 2304, 64, 64] = 38 MB in float32, x1000 views = 36 GB per
        # phase-1 cache, and four of those caches were 144 GB of a 225 GB volume. Stored as
        # float16 it is 19 MB / 18 GB, and the torch.save itself roughly halves. Every reader
        # casts back to float32 on load (train_encoder_dfd_crossview.py, train_encoder_dfd.py,
        # train_dino_sd_postproc.py), so training sees float32 either way -- the only difference
        # is the rounding of the stored values. Default stays float32 so every existing cache and
        # every existing script is untouched; pass --cache_dtype float16 explicitly.
        if args.cache_dtype == 'float16':
            model_f = model_f.half()
        torch.save({'model_f': model_f, 'elev': elev, 'azim': azim}, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')

    parser.add_argument('--encoder_data_dir', type=str, required=True)
    parser.add_argument('--encoder_model_dir', type=str, required=True)

    parser.add_argument('--target_image_size', type=int, default=224)
    parser.add_argument('--sd_frame', choices=['square_pad', 'aspect_nopad'], default='square_pad',
                        help='no-op for square renders; present so the photo and the cache can be '
                             'described by the same flags')
    parser.add_argument('--fuse_mode', choices=['dino_grid', 'common_grid'], default='dino_grid',
                        help="dino_grid = production. common_grid fuses at the 64x64 phase-1 "
                             "target instead of downsampling SD onto DINO's grid first.")
    parser.add_argument('--dino_only', action='store_true',
                        help='drop the SD branch entirely: cache the DINO half of the fusion '
                             '(1024 ch, not 2304). SD is never loaded or run. Use --width 1024 '
                             '--n_classes 1024 to match, and note --sd_frame/--fuse_mode become '
                             'no-ops. Added 2026-09-01 for the DINO vs DINO+SD ablation.')
    parser.add_argument('--dino_patch_stride', type=int, default=None,
                        help='overlapping-patch extraction: lower the patch-embedding stride below the '
                             'patch size (denser token grid, no retraining). None = unchanged.')
    parser.add_argument('--fix_grid_alignment', action='store_true',
                        help='place tokens at their true receptive-field centres when upsampling '
                             '(required for any stride != patch size; see modules/dino.py)')
    parser.add_argument('--feature_cache_name', type=str, default='dino_sd',
                        help="cache dir/file stem: <encoder_data_dir>/<name>_f/<render_res>/target_<name>_f_i.pt; "
                             'must match train_encoder_dfd_crossview.py --feature_model_type')
    parser.add_argument('--cache_dtype', type=str, default='float32',
                        choices=['float32', 'float16'],
                        help='dtype of the cached per-view target features. float16 halves the '
                             'cache on disk (36 GB -> 18 GB per 1000 views) and the torch.save '
                             'time; every reader casts back to float32 on load. Default float32 '
                             'keeps existing caches/scripts bit-identical.')

    parser.add_argument('--background', nargs=3, type=float, default=[1., 1., 1.])
    parser.add_argument('--frontview_std', type=float, default=4)
    parser.add_argument('--render_res', type=int, default=224)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.])
    parser.add_argument('--mesh_color', nargs=3, type=float, default=[2. / 3., 2. / 3., 2. / 3.])

    parser.add_argument('--n_views', type=int, default=1000)
    parser.add_argument('--view_sampling', type=str, default='random', choices=['random', 'fibonacci'])
    parser.add_argument('--data_percentage', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, required=True,
                        help='must match the fused feature dim (2304 = 1280 SD up_blocks[1] + 1024 DINO)')
    parser.add_argument('--n_classes', type=int, required=True, help='same as --width')
    parser.add_argument('--positional_encoding', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=5.0)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--network_verbose', type=int, default=1)

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=3)

    parser.add_argument('--generate_random_views', type=int, default=0)
    parser.add_argument('--encode_random_views', type=int, default=0)
    parser.add_argument('--start_training', type=int, default=0)

    args = parser.parse_args()
    args.model_type = 'dino_sd'

    args.mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)

    if args.generate_random_views == 1:
        generate_random_views(args)

    if args.encode_random_views == 1:
        dino_model = DINOWrapper(device=device, small=True, target_size=args.target_image_size).to(device)
        if args.dino_patch_stride is not None:
            set_patch_stride(dino_model, args.dino_patch_stride)
            print(f'  DINO patch stride overridden: {dino_model.patch_size()} -> {args.dino_patch_stride}')
        if args.dino_only:
            print('--dino_only: skipping Stable Diffusion entirely (1024-ch DINO cache)')
            sd_extractor = None
        else:
            print('loading SD components (first run downloads several GB, cached under $HF_HOME after)...')
            sd_extractor = SDFeatureExtractor(device=device)
        encode_random_views_dino_sd(args, dino_model, sd_extractor)
        del dino_model, sd_extractor
        torch.cuda.empty_cache()

    if args.start_training == 1:
        train_encoder(args, None)
