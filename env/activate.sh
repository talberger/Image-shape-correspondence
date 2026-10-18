# Source this on ANY machine (pod, Mac, Run:ai) before working:   source <repo>/env/activate.sh
# Replaces /workspace/activate_bsb.sh (2026-10-18; that file now just sources this one).
#   DISC_ROOT = this repo (code + data/ demo/ experiments/)
#   DISC_WS   = the folder around it: logs/, .cache/, sam3_repo/, pix2gestalt_repo/
#               (pod: /workspace; Mac: the repo's parent, e.g. ~/Projects; override by exporting DISC_WS first)

if [ -n "${BASH_VERSION:-}" ]; then _disc_src="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then eval '_disc_src="${(%):-%x}"'
else _disc_src="$0"; fi
export DISC_ROOT="$(cd "$(dirname "$_disc_src")/.." && pwd)"
export DISC_WS="${DISC_WS:-$(dirname "$DISC_ROOT")}"
unset _disc_src

# Model caches persist next to the repo (on the pod: the network volume, not the ephemeral /root).
export TORCH_HOME="$DISC_WS/.cache/torch"
export HF_HOME="$DISC_WS/.cache/huggingface"

if [ -d /workspace/miniconda3 ]; then
    # RunPod: conda lives on the volume; env 'disc' is a symlink to the old env dir 'bsb'.
    source /workspace/miniconda3/etc/profile.d/conda.sh
    conda activate disc
    # /workspace can't hold restrictive perms -> copy the GitHub deploy key to the pod's /root
    # (the repo's core.sshCommand points at /root/.ssh/id_ed25519_github).
    if [ -f /workspace/.ssh/id_ed25519_github ]; then
        mkdir -p /root/.ssh
        cp /workspace/.ssh/id_ed25519_github /root/.ssh/id_ed25519_github
        chmod 600 /root/.ssh/id_ed25519_github
        grep -q github.com /root/.ssh/known_hosts 2>/dev/null || \
            ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null
    fi
elif command -v conda >/dev/null 2>&1 && [ "${CONDA_DEFAULT_ENV:-}" != disc ]; then
    # Mac / any machine with its own conda: env 'disc' from env/requirements_disc_*.txt
    eval "$(conda shell."$(basename "${SHELL:-bash}")" hook 2>/dev/null)" && conda activate disc
fi
# Docker image (Run:ai): the image's python already is the disc env; nothing to activate.

export DISC_PY="$(command -v python)"
cd "$DISC_ROOT"
