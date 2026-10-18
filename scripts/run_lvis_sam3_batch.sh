#!/bin/bash
# Two-stage eval of a list of LVIS instances with the 2026-10-08 recipe (Tal): NO polygon -- object mask from SAM 3 (box = the crop),
# amodal mask (pix2gestalt, 1 of 3 samples) only where the mask is split, unlabelled pixels join the nearest segment, segment merge 1.5%.
# The models are loaded ONCE for the whole list: all crops -> all SAM 3 masks -> all amodal masks -> one worker for the training.
# The worker scores ZS / end of stage 1 (arm "stageA") / final per instance into $RESULTS/lvis<id>.json and slim-keeps the run dir
# (~95 MB: alpha pred_f of stage 1 + final, masks, segments; 2026-10-10. WORKER_ARGS="--keep_intermediates" = everything, "--delete_run_dir" = nothing).
# Usage: CLASS=bear ANN_IDS="76191 904 ..." RESULTS=data/densepose_lvis/results_X RUNS=data/densepose_lvis/runs_X \
#        setsid nohup bash scripts/run_lvis_sam3_batch.sh >> $DISC_WS/logs/<name>.log 2>&1 < /dev/null &
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
CLASS=${CLASS:-bear}; ANN_IDS=${ANN_IDS:?}; RESULTS=${RESULTS:?}; RUNS=${RUNS:?}; WORKER_ARGS=${WORKER_ARGS:-}
mkdir -p $RESULTS/masks $RUNS; RUNS=$(realpath $RUNS); T0=$(date +%s)
TODO=""; for i in $ANN_IDS; do [ -f $RESULTS/lvis$i.json ] || TODO="$TODO $i"; done   # resumable: scored instances are skipped everywhere
echo "##### $(echo $ANN_IDS | wc -w) instances, $(echo $TODO | wc -w) still to do #####"
if [ -n "$TODO" ]; then
echo "##### 1. crops #####"
CLASS=$CLASS RUNS=$RUNS TODO="$TODO" python - <<'PY'
import os, runpy, sys
for i in os.environ['TODO'].split():
    sys.argv = ['analysis/prep_lvis_instance.py', '--ann_id', i, '--class_name', os.environ['CLASS'], '--out_dir', f"{os.environ['RUNS']}/lvis{i}", '--long_side', '400']
    sys.path.insert(0, 'analysis'); runpy.run_path('analysis/prep_lvis_instance.py', run_name='__main__'); sys.path.pop(0)
PY
# MASK CACHE (Tal, 2026-10-11): the SAM 3 mask and the amodal mask depend only on the crop, so they are computed once per instance
# and reused by every later run. MASK_CACHE=none turns it off.
MC=${MASK_CACHE:-$DISC_ROOT/data/densepose_lvis/mask_cache_sam3/$CLASS}; [ "$MC" = none ] || mkdir -p $MC
NEED=""; for i in $TODO; do
  if [ "$MC" != none ] && [ -f $MC/lvis$i\_sam3_mask.pth ]; then cp $MC/lvis$i\_sam3_mask.p* $RUNS/lvis$i/; else NEED="$NEED $i"; fi
done
echo "##### 2. SAM 3 object masks (one process), $(echo $NEED | wc -w) to compute, the rest from $MC [$(( $(date +%s) - T0 )) s] #####"
TODO_ALL="$TODO"; TODO="$NEED"
[ -z "$NEED" ] || ( cd $DISC_WS/sam3_repo && env -u PYTHONPATH -u CONDA_DEFAULT_ENV -u CONDA_PREFIX PATH=$DISC_WS/sam3_repo/env/bin:/usr/local/bin:/usr/bin:/bin \
    HF_HOME=$DISC_WS/.cache/huggingface python run_box_mask.py --image $(for i in $TODO; do echo $RUNS/lvis$i/lvis$i.png; done) \
    --out $(for i in $TODO; do echo $RUNS/lvis$i/lvis$i\_sam3_mask.pth; done) 2>&1 | grep 'sam3 mask' )
TODO="$TODO_ALL"
echo "##### 3. which masks are split (largest part < 95%) -> amodal inputs [$(( $(date +%s) - T0 )) s] #####"
SPLIT=$(RUNS=$RUNS TODO="$TODO" python - <<'PY'
import os, sys, torch, numpy as np
from scipy import ndimage
out = []
for i in os.environ['TODO'].split():
    m = torch.load(f"{os.environ['RUNS']}/lvis{i}/lvis{i}_sam3_mask.pth").numpy().astype(bool)
    lab, _ = ndimage.label(m, structure=np.ones((3, 3))); share = np.bincount(lab[m]).max() / m.sum()
    print(f'lvis{i}: largest part {share:.3f} of the mask -> {"AMODAL" if share < 0.95 else "one piece"}', file=sys.stderr)
    if share < 0.95: out.append(i)
print(' '.join(out))
PY
)
echo "amodal needed for: ${SPLIT:-none}"
SPLIT_ALL="$SPLIT"; SPLIT=""; for i in $SPLIT_ALL; do
  if [ "$MC" != none ] && [ -f $MC/lvis$i\_amodal_mask.pth ]; then cp $MC/lvis$i\_amodal_mask.p* $RUNS/lvis$i/; else SPLIT="$SPLIT $i"; fi
done
echo "amodal to compute (not in the cache): ${SPLIT:-none}"
if [ -n "$SPLIT" ]; then
  for i in $SPLIT; do
    python analysis/prep_lvis_instance.py --ann_id $i --class_name $CLASS --out_dir $RUNS/lvis$i --long_side 400 --mask_source sam3 | tail -1
    python analysis/prep_amodal_input.py --ann_id $i --class_name $CLASS --out_dir $RUNS/lvis$i --mask_source sam3 | tail -1
  done
  echo "##### 4. pix2gestalt (one process) [$(( $(date +%s) - T0 )) s] #####"
  ( cd $DISC_WS/pix2gestalt_repo/pix2gestalt && env -u PYTHONPATH -u CONDA_DEFAULT_ENV PATH=$DISC_WS/pix2gestalt_repo/venv/bin:/usr/local/bin:/usr/bin:/bin \
      PYTHONPATH=../stubs:taming-transformers:CLIP python ../run_amodal.py --run_dirs $(for i in $SPLIT; do echo $RUNS/lvis$i; done) \
      --tags $(for i in $SPLIT; do echo lvis$i; done) --n_samples 3 --steps 50 2>&1 | grep 'model load\|samples x\|DONE' )
  for i in $SPLIT; do python analysis/merge_amodal_mask.py --run_dir $RUNS/lvis$i --tag lvis$i | cut -c1-200; done
fi
for i in $TODO; do   # the worker deletes the run dir after scoring: keep the masks; .done = "amodal step finished (mask or no mask)"
  cp $RUNS/lvis$i/lvis$i\_sam3_mask.p* $RESULTS/masks/; cp $RUNS/lvis$i/lvis$i\_amodal_mask.p* $RESULTS/masks/ 2>/dev/null || true; touch $RUNS/lvis$i/lvis$i\_amodal.done
  [ "$MC" = none ] || { cp -n $RUNS/lvis$i/lvis$i\_sam3_mask.p* $MC/; cp -n $RUNS/lvis$i/lvis$i\_amodal_mask.p* $MC/ 2>/dev/null || true; }
done
echo "##### 5. worker: training + eval (ZS / stage 1 / final) [$(( $(date +%s) - T0 )) s] #####"
PYTHONUNBUFFERED=1 python scripts/run_lvis_worker.py --ann_ids $TODO --class_name $CLASS --results_dir $RESULTS --runs_dir $RUNS \
  --mu_scope global_then_segment_aniso --mu_switch_step 150 --mu_lr 5e-2 --merge_below_frac 0.015 --fill_unlabelled_segments \
  --mask_source sam3 --amodal_mask --amodal_wait 5 $([ "$MC" = none ] || echo "--fine_seg_cache $MC/fine_segments") $WORKER_ARGS
fi
python analysis/aggregate_lvis_results.py --results_dir $RESULTS --class_name $CLASS --expected $(echo $ANN_IDS | wc -w) --out_json $RESULTS/aggregate.json
echo "SAM3 BATCH DONE $RESULTS [$(( $(date +%s) - T0 )) s]"
