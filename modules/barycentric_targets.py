"""Barycentric xyz-target caching for DFD-style phase-1 distillation (threedle/dfd, CVPR 2026).

Unlike every existing target cache (dino_sd_f/, sam_f/, etc.), this one doesn't run any external
2D model -- the "encoder" for this target is kaolin's own rasterizer. Instead of painting the
network's *prediction* as the mesh's render color (what every existing train_encoder* script does,
every training step, since that prediction changes every step), this paints the mesh's *positions*
(mesh.vertices, constant every step) as the color -- once, here -- to get the exact continuous
barycentric-interpolated xyz point under every foreground pixel, for every cached view.

Saved in the exact {'model_f','elev','azim'} schema every existing target cache already uses
(encoder.py:296, train_encoder_dino_sd.py:89), under <encoder_data_dir>/xyz_bary_f/<render_res>/,
so EncoderDataset needs zero changes to read it if ever wanted generically -- train_encoder_dfd.py
instead reads it directly by file index (see that file's docstring for why).

Reuses the same random_viewing_angles.pt every other cache in this data_dir was built from, so this
cache aligns view-for-view (same elev/azim per index i) with the dino_sd_f cache already there --
critical since train_encoder_dfd.py pairs the two caches by explicit index, not by re-sampling.
"""
import os

import torch

from modules.utils import device, setcolor_mesh
from modules.render import Renderer

# Bounding-sphere-normalized mesh vertices always satisfy ||v|| <= 1, so every channel of a real
# surface point is in [-1, 1]. 2.0 is unreachable, unlike reusing 1.0 (which every existing
# feature-target cache uses, but real feature values can and do land near 1.0 in some channels).
BACKGROUND_SENTINEL = 2.0


def encode_random_views_xyz_bary(args):
    print('Generate barycentric xyz targets (positions painted as color, rendered once)')
    save_dir = os.path.join(args.encoder_data_dir, 'xyz_bary_f', str(args.render_res))
    os.makedirs(save_dir, exist_ok=True)

    data = torch.load(os.path.join(args.encoder_data_dir, str(args.render_res), 'random_viewing_angles.pt'))
    elev_list, azim_list = data['elev'], data['azim']

    render = Renderer(dim=(64, 64), radius=args.radius)
    mesh = args.mesh

    for i in range(len(elev_list)):
        save_path = os.path.join(save_dir, f'target_xyz_bary_f_{i}.pt')
        if os.path.exists(save_path):
            continue

        elev, azim = elev_list[i], azim_list[i]

        # Paint raw positions as color -- constant every call, unlike setcolor_mesh(mesh, pred_f)
        # in every existing train_encoder* script, which paints the changing prediction instead.
        setcolor_mesh(mesh, mesh.vertices)

        xyz_img, ret_elev, ret_azim, mask = render.render_views(
            mesh, num_views=1, random_views=False,
            center_elev=elev if torch.is_tensor(elev) else torch.tensor([elev], dtype=torch.float32),
            center_azim=azim if torch.is_tensor(azim) else torch.tensor([azim], dtype=torch.float32),
            return_views=True, return_features=True, lighting=False,
            background=torch.full((3,), BACKGROUND_SENTINEL).to(device), return_mask=True)

        torch.save({'model_f': xyz_img.squeeze(0).detach().cpu(), 'elev': ret_elev, 'azim': ret_azim},
                   save_path)

        if i % 100 == 0:
            print(f'xyz_bary target {i}/{len(elev_list)}')
