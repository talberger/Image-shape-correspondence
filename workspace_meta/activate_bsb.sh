#!/usr/bin/env bash
# Source this file to restore the bsb dev environment on any pod
# (after migration, just: source /workspace/activate_bsb.sh)

export PIP_CACHE_DIR=/workspace/.cache/pip

# --- Tal edit start ---
# torch.hub's weight cache defaults to ~/.cache/torch, which lives on the pod's
# ephemeral overlay fs and is wiped on every pod switch. Point it at /workspace
# so downloaded checkpoints survive. Note this is NOT covered by dino.py's
# source='local' pin: that only replaces the *repo* fetch, while pretrained
# weights still go through torch.hub.load_state_dict_from_url (see
# dinov2/hub/backbones.py) and land in $TORCH_HOME/hub/checkpoints.
export TORCH_HOME=/workspace/.cache/torch

# Same reasoning, for Hugging Face downloads (diffusers/transformers/
# huggingface_hub all respect HF_HOME as the base cache dir): default is
# ~/.cache/huggingface, on the ephemeral pod fs. Added 2026-08-18 before the
# first Stable Diffusion download (SD v1-5 UNet+VAE+CLIP, several GB) so it
# persists across pod switches instead of re-downloading every time.
export HF_HOME=/workspace/.cache/huggingface
# --- Tal edit end ---

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate bsb

# /workspace is a network volume that doesn't support restrictive file
# permissions, so the GitHub deploy key (private key material lives there
# for persistence across pods) has to be copied to the pod's local/ephemeral
# filesystem before ssh will accept it (it refuses keys with open perms).
# This re-copies it fresh on every pod; /workspace/bsb's git config already
# points core.sshCommand at this local path.
if [ -f /workspace/.ssh/id_ed25519_github ]; then
    mkdir -p /root/.ssh
    cp /workspace/.ssh/id_ed25519_github /root/.ssh/id_ed25519_github
    chmod 600 /root/.ssh/id_ed25519_github
    ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null
fi

cd /workspace/bsb
