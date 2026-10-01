"""
LORA-W (2026-09-17, Tal): low-rank adaptation of the phase-1 encoder's WEIGHTS for phase-2.

Every wrapped nn.Linear keeps its phase-1 weight W and bias frozen and gains a trainable rank-r
update:  y = W x + b + alpha * B (A x).  B starts at zero, so at step 0 the encoder is bit-identical
to phase-1 (the resume gate still passes). `alpha` is a plain attribute, NOT a parameter: it is 1
during training and is swept at READ time (analysis/export_pred_f_lora_alpha.py) to dial how much of
phase-2's change is applied, without retraining.

Wrapping only the OUTPUT layer (the default, Tal 2026-09-17) makes the change in feature space
exactly B A h(x): rank <= r in the 2304-dim feature space, and smooth over the mesh because the
frozen hidden representation h(x) is.

Name map: LORA-W = this file (weights). LORA-F = a low-rank addition on the vertex FEATURE table,
not built. Both belong to the FROZEN-DELTA direction.
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.alpha = 1.0
        # Standard LoRA init: A random (kaiming-uniform, as nn.Linear), B zero -> the update starts at 0.
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def delta_weight(self):
        return self.lora_B @ self.lora_A

    def forward(self, x):
        return self.base(x) + self.alpha * ((x @ self.lora_A.t()) @ self.lora_B.t())


def linear_indices(encoder):
    return [i for i, m in enumerate(encoder.mlp) if isinstance(m, nn.Linear)]


def apply_lora_w(encoder, rank, layers='last'):
    """Freeze EVERY encoder parameter (weights, biases, LayerNorm), then wrap the chosen Linear layers.
    layers: 'last' (output layer only) or 'all'. Returns the wrapped mlp indices."""
    for p in encoder.parameters():
        p.requires_grad_(False)
    idx = linear_indices(encoder)
    chosen = idx[-1:] if layers == 'last' else idx
    for i in chosen:
        encoder.mlp[i] = LoRALinear(encoder.mlp[i], rank)
    return chosen


def lora_state(encoder):
    """Just the trainable update, keyed by mlp index -- enough to rebuild W + alpha*BA for any alpha."""
    return {i: {'A': m.lora_A.detach().cpu(), 'B': m.lora_B.detach().cpu(), 'rank': m.rank}
            for i, m in enumerate(encoder.mlp) if isinstance(m, LoRALinear)}


def merged_state_dict(encoder, alpha=1.0):
    """A plain EncoderL2Norm state_dict with W <- W + alpha*BA baked in, so every existing loader /
    export / analysis script reads a LORA-W checkpoint unchanged."""
    sd = {}
    for i, m in enumerate(encoder.mlp):
        if isinstance(m, LoRALinear):
            sd['mlp.%d.weight' % i] = (m.base.weight + alpha * m.delta_weight()).detach().clone()
            sd['mlp.%d.bias' % i] = m.base.bias.detach().clone()
        else:
            for k, v in m.state_dict().items():
                sd['mlp.%d.%s' % (i, k)] = v.detach().clone()
    return sd


def merge_lora_into_state_dict(base_sd, lstate, alpha):
    """Rebuild a merged state_dict from a phase-1 state_dict + a saved lora_state at a chosen alpha."""
    sd = {k: v.clone() for k, v in base_sd.items()}
    for i, s in lstate.items():
        key = 'mlp.%d.weight' % int(i)
        sd[key] = sd[key] + alpha * (s['B'].to(sd[key].device) @ s['A'].to(sd[key].device))
    return sd
