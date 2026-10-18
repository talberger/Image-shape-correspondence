#!/bin/bash
# Object mask of one LVIS instance from SAM 3 instead of the LVIS polygon (Tal, 2026-10-08: only the image goes in).
# Box prompt = the whole bbox crop, no text. prep (bsb env, writes the crop) -> SAM 3 (its own env, $DISC_WS/sam3_repo/env)
# => <D>/lvis<id>_sam3_mask.pth (bool [H, W] on the run-crop grid). Used by --mask_source sam3.
# Usage: CLASS=bear ANN_ID=206034 D=<run dir of the instance> bash scripts/make_sam3_mask.sh
set -euo pipefail
CLASS=${CLASS:?}; ANN_ID=${ANN_ID:?}; D=${D:?}; TAG=lvis$ANN_ID
[ -n "${DISC_ROOT:-}" ] || source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT; mkdir -p $D; D=$(realpath $D)
[ "${SKIP_PREP:-0}" = 1 ] || python analysis/prep_lvis_instance.py --ann_id $ANN_ID --class_name $CLASS --out_dir $D --long_side 400   # SKIP_PREP=1: the crop is already in <D>
( cd $DISC_WS/sam3_repo && env -u PYTHONPATH -u CONDA_DEFAULT_ENV -u CONDA_PREFIX PATH=$DISC_WS/sam3_repo/env/bin:/usr/local/bin:/usr/bin:/bin \
    HF_HOME=$DISC_WS/.cache/huggingface python run_box_mask.py --image $D/$TAG.png --out $D/${TAG}_sam3_mask.pth )
echo "SAM3 MASK DONE $D/${TAG}_sam3_mask.pth"
