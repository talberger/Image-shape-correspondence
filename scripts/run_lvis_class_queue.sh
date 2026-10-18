#!/bin/bash
# 2026-10-04: run_lvis_queue.sh + CLASS (bear/horse/elephant). A SEPARATE file because the bear queue was running when
# this was written, and bash reads a running script by byte offset -- never edit a queue script that is in flight.
# The full DensePose-LVIS run of one class: all instances, split across N GPUs, one worker per GPU.
#
# Each worker is scripts/run_lvis_worker.py -- ONE persistent process that does every stage of every
# instance it owns, so the ~52s/stage of python imports is paid once per worker rather than seven
# times per instance. Instances are dealt round-robin so a worker that draws several large bears does
# not end up carrying the whole tail.
#
# RESUMABLE BY DESIGN: a worker skips any instance that already has a results JSON, so if the pod
# dies, relaunching this script with the same arguments picks up where it stopped. Nothing needs to
# be cleaned up first. Use --redo (edit below) only to deliberately rescore.
#
# DISK: ~2.1 GB in flight per worker (features 1.2 GB + geo table 0.66 GB at long_side 400), deleted
# as soon as that instance is scored. With N workers the peak is ~N x 2.1 GB, NOT 132 x anything.
#
# Usage:
#   setsid nohup env NUM_GPUS=4 bash scripts/run_lvis_queue.sh > $DISC_WS/logs/lvis_queue.log 2>&1 &
#   NUM_GPUS=2 ANN_IDS="904 117683" bash scripts/run_lvis_queue.sh     # a subset, for a smoke test
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

NUM_GPUS=${NUM_GPUS:-1}
# 2026-09-22: NUM_WORKERS may exceed NUM_GPUS (several workers share a GPU round-robin; RAM is the limit,
# ~11 GB peak each). WORKER_ARGS is passed to every worker, e.g. "--mu_scope segment_aniso --mu_lr 5e-2".
NUM_WORKERS=${NUM_WORKERS:-$NUM_GPUS}; WORKER_ARGS=${WORKER_ARGS:-}
# CLASS (2026-10-04): bear (default, unchanged) | horse | elephant
CLASS=${CLASS:-bear}
RESULTS=${RESULTS:-data/densepose_lvis/results}
LOGDIR=${LOGDIR:-$DISC_WS/logs}
mkdir -p "$RESULTS" "$LOGDIR"

# The instance list: every ds2_val bear that carries GT points (132 of them), unless overridden.
if [ -z "${ANN_IDS:-}" ]; then
  ANN_IDS=$(CLASS=$CLASS python - <<'PY'
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
from analysis.eval_densepose_lvis import load_instances
anns, _ = load_instances(os.environ['CLASS'])
print(' '.join(str(a['id']) for a in sorted(anns, key=lambda a: a['id'])))
PY
)
fi
read -r -a IDS <<< "$ANN_IDS"
echo "queue: ${#IDS[@]} instances across $NUM_GPUS GPU(s), results -> $RESULTS"

for ((g = 0; g < NUM_WORKERS; g++)); do
  SHARD=()
  for ((i = g; i < ${#IDS[@]}; i += NUM_WORKERS)); do SHARD+=("${IDS[$i]}"); done
  [ ${#SHARD[@]} -eq 0 ] && continue
  LOG="$LOGDIR/lvis_queue_gpu$g.log"
  echo "  gpu $g: ${#SHARD[@]} instances -> $LOG"
  # Plain `&`, NOT `setsid nohup`: setsid forks and the parent exits immediately, so `wait` below
  # would return straight away and the aggregate would run against an empty results dir. Launch
  # THIS script detached instead (setsid nohup bash scripts/run_lvis_queue.sh ...) so the whole
  # queue survives the laptop closing -- the workers are then children of a detached parent.
  CUDA_VISIBLE_DEVICES=$((g % NUM_GPUS)) nohup python -u scripts/run_lvis_worker.py --class_name "$CLASS" $WORKER_ARGS \
    --ann_ids "${SHARD[@]}" --results_dir "$RESULTS" >> "$LOG" 2>&1 < /dev/null &
done

wait
echo "ALL WORKERS EXITED"
python analysis/aggregate_lvis_results.py --results_dir "$RESULTS" --class_name "$CLASS" --expected "${#IDS[@]}" \
  --out_json "$RESULTS/aggregate.json"
