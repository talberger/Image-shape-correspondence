"""GUARD for the --dino_phases refactor (2026-09-07).

The phase-marginalized path fuses on the IMAGE frame instead of on DINO's 16x16 token grid, and
upsamples the SD and DINO halves SEPARATELY. That is only legitimate if it is bit-identical to
production for the SD half. Bilinear interpolation is linear and acts per channel, so

    crop(interp(cat[sd, dino]))  ==  cat[crop(interp(sd)), crop(interp(dino))]

This script asserts it on the real spatial dimensions of the guitar photo (channel count is
irrelevant to the identity, so it uses a small one to stay fast).

Run:  python scripts/check_phase_fuse_identity.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.extract_dino_sd_features import sd_half_image_frame
from modules.dino_sd_fusion import fuse_dino_sd

TARGET, INPUT_SIZE, ORIGINAL = 224, (224, 168), (1080, 810)
GRID = (16, 16)


def postprocess_like_production(fused):
    """modules/dino.py::postprocess_features, fix_grid_alignment=False branch, verbatim."""
    x = F.interpolate(fused, (TARGET, TARGET), mode='bilinear', align_corners=False)
    x = x[..., :INPUT_SIZE[0], :INPUT_SIZE[1]]
    return F.interpolate(x, ORIGINAL, mode='bilinear', align_corners=False)


def main():
    torch.manual_seed(0)
    sd_raw = torch.randn(1, 8, 32, 32)
    dino_raw = torch.randn(1, 6, *GRID)

    production = postprocess_like_production(fuse_dino_sd(sd_raw, dino_raw))

    sd_half = sd_half_image_frame(sd_raw, GRID, INPUT_SIZE, TARGET)
    dino_half = F.interpolate(F.normalize(dino_raw.float(), dim=1) * 0.5, (TARGET, TARGET),
                              mode='bilinear', align_corners=False)[..., :INPUT_SIZE[0], :INPUT_SIZE[1]]
    split = F.interpolate(torch.cat([sd_half, dino_half], dim=1), ORIGINAL, mode='bilinear',
                          align_corners=False)

    d = (production - split).abs().max().item()
    print(f'production vs split-halves : max abs diff = {d:.3e}   shapes {tuple(production.shape)}')
    assert d == 0.0, f'NOT bit-identical: {d:.3e}'
    print('OK -- the SD half is bit-identical to production; the only change is the DINO half.')


if __name__ == '__main__':
    main()
