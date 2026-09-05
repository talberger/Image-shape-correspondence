"""
Small learned post-processor for the fused SD+DINO per-pixel feature maps -- fallback
plan for left/right disambiguation (see PROJECT_NOTES.md), used only if the GeoAware-SC
pretrained-network experiment doesn't transfer to the guitar. Trained self-supervised via
modules/xyz_correspondence.py's free correspondence/mirror pairs (see
train_dino_sd_postproc.py, not yet written).

New, additive module -- doesn't touch modules/dino_sd_fusion.py or any training script.
Operates on the already-cached, already-upsampled [2304,64,64] dino_sd_f maps (matches
GeoAware-SC's own AggregationNetwork design choice of post-processing after all sources
share one spatial resolution, confirmed by reading their code this session).
"""
import torch
import torch.nn.functional as F
from torch import nn


class FeaturePostProcessor(nn.Module):
    """1x1 -> 3x3 -> 3x3 -> 1x1 bottleneck (GroupNorm+GELU between layers), residual with
    a zero-initialized final conv so the module is an exact identity function at init --
    training starts from today's production feature space and only moves where the
    correspondence loss actually pushes it, never worse than the untouched baseline by
    construction at step 0.
    """

    def __init__(self, in_dim=2304, hidden_dim=384, out_dim=None, num_norm_groups=32,
                 l2_normalize=True):
        super().__init__()
        out_dim = out_dim or in_dim
        assert out_dim == in_dim, 'residual connection requires out_dim == in_dim'
        self.l2_normalize = l2_normalize

        self.reduce = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.norm1 = nn.GroupNorm(num_norm_groups, hidden_dim)
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_norm_groups, hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm3 = nn.GroupNorm(num_norm_groups, hidden_dim)
        self.expand = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

        # zero-init the last conv -> the whole block outputs zeros at init -> residual
        # sum is an exact identity, verified in the standalone check at the bottom of
        # this file.
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x):
        """x: [B, in_dim, H, W], any H, W (all-conv). Returns [B, in_dim, H, W]."""
        h = F.gelu(self.norm1(self.reduce(x)))
        h = F.gelu(self.norm2(self.conv1(h)))
        h = F.gelu(self.norm3(self.conv2(h)))
        h = self.expand(h)
        out = x + h
        if self.l2_normalize:
            out = F.normalize(out, dim=1)
        return out


def save_checkpoint(model: FeaturePostProcessor, path, **ctor_kwargs):
    torch.save({'state_dict': model.state_dict(), 'ctor_kwargs': ctor_kwargs}, path)


def load_checkpoint(path, device='cpu') -> FeaturePostProcessor:
    ckpt = torch.load(path, map_location=device)
    model = FeaturePostProcessor(**ckpt['ctor_kwargs']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    return model


if __name__ == '__main__':
    # standalone identity-at-init check, per this module's build-order requirement
    torch.manual_seed(0)
    m = FeaturePostProcessor(in_dim=2304, hidden_dim=384, l2_normalize=False)
    x = torch.randn(1, 2304, 8, 8)
    out = m(x)
    assert torch.allclose(out, x, atol=1e-6), 'not identity at init!'
    n_params = sum(p.numel() for p in m.parameters())
    print(f'identity-at-init check passed. params={n_params:,} ({n_params/1e6:.2f}M)')
