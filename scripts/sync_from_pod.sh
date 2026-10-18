#!/bin/bash
# Copy data from a RunPod pod to this machine (the Mac). Code is NOT synced here -- that is git.
# Pulls only; never deletes anything on either side. Re-running copies only what changed.
# Usage (on the Mac):  bash scripts/sync_from_pod.sh <pod IP> <pod SSH port> [set ...]
#   IP / port: RunPod pod page -> Connect -> "SSH over exposed TCP" (they change with every pod).
#   Sets (default: base): see the case below. Any other argument = a path relative to the repo, e.g.
#   bash scripts/sync_from_pod.sh 213.173.111.73 57684 demo/shic/crowding/horse_lvis56069
set -euo pipefail
IP=${1:?pod IP}; PORT=${2:?pod SSH port}; shift 2
[ $# -gt 0 ] || set -- base
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POD_ROOT=/workspace/disc

paths=()
for s in "$@"; do
    case "$s" in
        base)       # meshes, SHIC templates, LVIS annotations + subsets (~240 MB)
            paths+=(meshes data/shic_horse data/shic_bear data/shic_elephant
                    data/densepose_lvis/densepose_lvis_v1_ds1_val_v1.json data/densepose_lvis/densepose_lvis_v1_ds2_val_v1.json
                    data/densepose_lvis/subset20_bear.json data/densepose_lvis/subset20_bear_evenspaced.json
                    data/densepose_lvis/subset20_bear_stratified.json) ;;
        horse56069) # last experiment (pods 44-48): crowding study + the current-recipe run + its figures (~3.7 GB)
            paths+=(demo/shic/crowding/horse_lvis56069
                    data/densepose_lvis/runs_debug_merge0.015_amodal_fillseg_sam3mask/lvis56069
                    demo/shic/debug_twostage/horse_lvis56069_merge0.015_amodal_fillseg_sam3mask) ;;
        *) paths+=("$s") ;;
    esac
done

for p in "${paths[@]}"; do
    echo "== $p"
    mkdir -p "$ROOT/$(dirname "$p")"
    rsync -a -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new" "root@$IP:$POD_ROOT/$p" "$ROOT/$(dirname "$p")/"
done
echo "SYNC DONE"
