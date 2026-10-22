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
    if [ -f /root/disc_env/READY ]; then
        # local copy from env/pod_local_env.sh (2026-10-22): same env, imports fast (the volume is slow for small files)
        case $- in *u*) set +u; source /root/disc_env/bin/activate; set -u ;;   # it reads unset vars (scripts run set -u)
                   *) source /root/disc_env/bin/activate ;; esac
    else
        conda activate disc
    fi
    # /workspace can't hold restrictive perms -> copy the GitHub deploy key to the pod's /root
    # (the repo's core.sshCommand points at /root/.ssh/id_ed25519_github).
    if [ -f /workspace/.ssh/id_ed25519_github ]; then
        mkdir -p /root/.ssh
        cp /workspace/.ssh/id_ed25519_github /root/.ssh/id_ed25519_github
        chmod 600 /root/.ssh/id_ed25519_github
        grep -q github.com /root/.ssh/known_hosts 2>/dev/null || \
            ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null
    fi
elif [ "${CONDA_DEFAULT_ENV:-}" != disc ]; then
    # Mac / any machine with its own conda: env 'disc' from env/requirements_disc_*.txt.
    # Activate WITHOUT conda's shell hook (it fails in Claude Code's shell on the Mac: "__conda_exe:
    # permission denied"): put the env first on PATH, which is all the scripts need.
    for _p in "${CONDA_PREFIX_DISC:-}" "$HOME/miniconda3/envs/disc" "$HOME/anaconda3/envs/disc" "/opt/miniconda3/envs/disc"; do
        if [ -n "$_p" ] && [ -x "$_p/bin/python" ]; then
            export CONDA_PREFIX="$_p" CONDA_DEFAULT_ENV=disc PATH="$_p/bin:$PATH"; break
        fi
    done
    unset _p
fi
if [ "$(uname)" = Darwin ] && [ -n "${CONDA_PREFIX:-}" ]; then
    # open3d 0.17 links Homebrew's libomp; on the Mac it comes from conda-forge llvm-openmp in the env.
    export DYLD_FALLBACK_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/lib:/usr/lib"
fi
# Docker image (Run:ai): the image's python already is the disc env; nothing to activate.

export DISC_PY="$(command -v python)"
cd "$DISC_ROOT"
