#!/bin/bash
# Amodal ("as if the occluder is not there") mask of one LVIS instance, for --amodal_mask runs (Tal, 2026-10-07).
# prep (bsb env) -> pix2gestalt (its own venv, $DISC_WS/pix2gestalt_repo) -> merge (bsb env) => <D>/lvis<id>_amodal_mask.pth
# Usage: CLASS=bear ANN_ID=7107 D=<run dir of the instance> [N_SAMPLES=3] [STEPS=50] [MASK_SOURCE=sam3] [ONLY_IF_SPLIT=1] [SKIP_PREP=1] [PRELOADED=1] bash scripts/make_amodal_mask.sh
# 2026-10-08 (speed): ONLY_IF_SPLIT=1 = skip pix2gestalt when the object mask is already in one piece (largest connected part >= 95% of it;
# no amodal mask is written, the run then uses the plain mask). SKIP_PREP=1 = the crop / mask / gt are already in <D> (the caller ran prep).
# <D>/lvis<id>_amodal.done is written at the end in both cases: a worker started with --amodal_wait waits for it.
set -euo pipefail
CLASS=${CLASS:?}; ANN_ID=${ANN_ID:?}; D=${D:?}; TAG=lvis$ANN_ID; N=${N_SAMPLES:-3}; STEPS=${STEPS:-50}
[ -n "${DISC_ROOT:-}" ] || source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT; mkdir -p $D; D=$(realpath $D); rm -f $D/${TAG}_amodal.done; [ "${PRELOADED:-0}" = 1 ] || rm -f $D/${TAG}_amodal_input.ready $D/${TAG}_amodal_samples.done   # PRELOADED: the caller cleared them
MS=${MASK_SOURCE:-polygon}   # sam3 = the visible mask is SAM 3's, not the LVIS polygon (2026-10-08)
[ "${SKIP_PREP:-0}" = 1 ] || python analysis/prep_lvis_instance.py --ann_id $ANN_ID --class_name $CLASS --out_dir $D --long_side 400 --mask_source $MS
if [ "${ONLY_IF_SPLIT:-0}" = 1 ]; then
  SHARE=$(python -c "
import sys, torch, numpy as np
from scipy import ndimage
m = torch.load(sys.argv[1]).numpy().astype(bool); lab, n = ndimage.label(m, structure=np.ones((3, 3)))
print('%.4f' % (np.bincount(lab[m]).max() / m.sum()))" $D/${TAG}_mask.pth)
  if python -c "import sys; sys.exit(0 if float('$SHARE') >= 0.95 else 1)"; then
    rm -f $D/${TAG}_amodal_mask.pth
    echo "AMODAL SKIPPED: the largest part is $SHARE of the mask (>= 0.95), no occlusion to complete"; touch $D/${TAG}_amodal.done; exit 0
  fi
  echo "amodal needed: the largest part is only $SHARE of the mask"
fi
python analysis/prep_amodal_input.py --ann_id $ANN_ID --class_name $CLASS --out_dir $D --mask_source $MS
touch $D/${TAG}_amodal_input.ready
if [ "${PRELOADED:-0}" = 1 ]; then   # the caller already started run_amodal.py --wait_for_input (model loaded meanwhile): wait for its samples
  for _ in $(seq 900); do [ -f $D/${TAG}_amodal_samples.done ] && break; sleep 1; done
  [ -f $D/${TAG}_amodal_samples.done ] || { echo "no samples from the preloaded pix2gestalt process"; exit 1; }
else
( cd $DISC_WS/pix2gestalt_repo/pix2gestalt && env -u PYTHONPATH -u CONDA_DEFAULT_ENV PATH=$DISC_WS/pix2gestalt_repo/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONPATH=../stubs:taming-transformers:CLIP python ../run_amodal.py --run_dirs $D --tags $TAG --n_samples $N --steps $STEPS )
fi
python analysis/merge_amodal_mask.py --run_dir $D --tag $TAG ${MIN_VOTES:+--min_votes $MIN_VOTES}   # default 1 of the samples (2026-10-08)
touch $D/${TAG}_amodal.done
echo "AMODAL MASK DONE $D/${TAG}_amodal_mask.pth"
