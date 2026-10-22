#!/bin/bash
# Pod only (Tal 2026-10-22): unpack the 'disc' env from ONE tar on the volume to the pod's local disk.
# Why: on a slow /workspace volume `import torch` from /workspace/miniconda3 took ~60 s (thousands of small files);
# one big tar reads fast, and the local copy imports in seconds. env/activate.sh uses the local copy once it is READY.
#   bash env/pod_local_env.sh          # /start-work runs it in the background; ~1-2 min
# The tar is built once with conda-pack (rebuild it after installing anything into the env -- the local copy is a snapshot):
#   /workspace/miniconda3/bin/conda-pack -p /workspace/miniconda3/envs/bsb -o /workspace/env_packs/disc.tar.part \
#     --format tar --ignore-editable-packages --ignore-missing-files -j 8 && mv /workspace/env_packs/disc.tar{.part,}
set -euo pipefail
TAR=/workspace/env_packs/disc.tar
DST=/root/disc_env
[ -f "$TAR" ] || { echo "no $TAR -- build it first (see header)"; exit 1; }
if [ -f $DST/READY ] && [ $DST/READY -nt "$TAR" ]; then echo "local env up to date: $DST"; exit 0; fi
t0=$(date +%s)
rm -rf $DST $DST.tmp; mkdir -p $DST.tmp
tar -xf "$TAR" -C $DST.tmp
mv $DST.tmp $DST
$DST/bin/conda-unpack                         # rewrites the hard-coded /workspace prefixes to $DST
touch $DST/READY
echo "local env ready: $DST ($(( $(date +%s) - t0 )) s)"
