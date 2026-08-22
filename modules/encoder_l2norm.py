"""
L2-normalized variant of modules/encoder.py's Encoder -- new, additive module, does not modify
that file. Identical architecture (same depth/width/positional-encoding/Fourier-feature setup),
except the final Tanh() is replaced with an explicit unit-L2-norm projection.

Why: Tanh bounds each output channel independently to [-1,1] but places no constraint on the
overall feature vector's magnitude -- a vertex's output norm can end up anywhere depending on how
many channels saturate. Both reference papers this project's DINO+SD fusion is modeled on
(Diff3F, arXiv 2311.17024; Surface-Aware Distilled 3D Semantic Features, arXiv 2503.18254) make
their final per-point 3D feature unit-norm by construction instead -- Diff3F as a near-byproduct
of averaging already-unit-norm per-view features, Surface-Aware via an explicit L2-normalize on
its embedding output. Every downstream similarity computation in this project (soft-snap, KNN,
PCA distance) is cosine-based, i.e. direction-only -- an unconstrained-magnitude Tanh output is
solving a harder problem (matching direction AND magnitude of a small, fixed-scale target) than
the one that's actually needed.

Experimental variant, isolated: paired with train_encoder_dino_sd_l2norm.py, writing to its own
experiments/guitar/encoder_dino_sd_frontcentered_l2norm_phase1/ output dir, so it can be dropped without
touching anything already relied on.
"""
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import FourierFeatureTransform


class L2Normalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, dim=-1)


class EncoderL2Norm(nn.Module):
    def __init__(self, depth, width, out_dim, input_dim=3, positional_encoding=False, sigma=5.0, network_verbose=1):
        super(EncoderL2Norm, self).__init__()
        layers = []
        if positional_encoding:
            layers.append(FourierFeatureTransform(input_dim, width, sigma))
            layers.append(nn.Linear(width * 2 + input_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width]))
        else:
            layers.append(nn.Linear(input_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width]))
        for i in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width]))
        layers.append(nn.Linear(width, out_dim))
        layers.append(L2Normalize())  # only line that differs from modules/encoder.py's Encoder

        self.mlp = nn.ModuleList(layers)
        if network_verbose == 1:
            print(self.mlp)

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x
