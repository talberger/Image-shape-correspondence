#!/bin/bash
# Isolated env for AnyUp ("AnyUp: Universal Feature Upsampling", ICLR 2026 Oral,
# github.com/wimmerth/anyup) -- python 3.12 + torch 2.9.0+cu128, fully separate
# from the bsb env (python 3.9 / torch 1.12). Same disposable-isolated-env pattern
# used for the (since-removed) geoaware env -- see PROJECT_NOTES.md. Never imports
# into bsb; only reads/writes plain .pt files that bsb's own scripts consume.
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate anyup
export TORCH_HOME=/workspace/.cache/torch_anyup
unset HF_HUB_ENABLE_HF_TRANSFER  # same fix as geoaware_sc -- not installed in this env
