#!/usr/bin/env bash
# Small analyze + comparison-vs-spatial-only for a pairwise-angle arm (2026-09-04).
# Usage: ./run_isobbox_pairangle_small_analyze.sh <attached|detachfeat> [weight, default 0.01]
#
# NAMING, AND WHY THIS WRAPPER EXISTS. This project uses TWO OPPOSITE orderings for the same run:
#   experiments/guitar/encoder_isobbox_DINO_SD_stride14_...
#   demo/guitar/runs/.../bsb_isobbox_SD_DINO_stride14_...
# Getting that backwards does not fail -- it silently creates a SECOND demo folder and splits one
# run's outputs across two names (which is exactly what happened on 2026-09-04: the step
# progression landed in the sd_dino folder and the analyze/eval/comparisons in a dino_sd one).
# Both names are derived here, once, from a single arm argument so they cannot drift apart again.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /workspace/bsb

MODE=${1:-}
W=${2:-0.01}
NB=${3:-all}          # all | in_segment -- must match how the run was TRAINED
SPATIAL=${4:-on}      # on | off -- must match how the run was TRAINED
DW=${5:-0}            # RKD-D weight -- must match how the run was TRAINED
case "$MODE" in
  attached|gradFW)  MODE=gradFW ;;
  detachfeat|gradW) MODE=gradW ;;
  detachw|gradF)    MODE=gradF ;;
  *) echo "usage: $0 <gradFW|gradW|gradF> [weight] [all|in_segment]"; exit 1 ;;
esac
case "$NB" in all) NBTAG="" ;; in_segment) NBTAG="_nbrClose" ;; *) echo "3rd arg must be all|in_segment"; exit 1 ;; esac
case "$SPATIAL" in on) SPTAG="" ;; off) SPTAG="_noSpatial" ;; *) echo "4th arg must be on|off"; exit 1 ;; esac
DTAG=""; [ "$DW" != "0" ] && DTAG="_dist${DW}"

EXP_NAME=encoder_isobbox_pairangle${W}${NBTAG}${DTAG}_soft_${MODE}${SPTAG}_T1_seed0_phase2
DEMO_NAME=runs/rkd_feature/bsb_isobbox_pairangle${W}${NBTAG}${DTAG}_soft_${MODE}${SPTAG}_T1_seed0_phase2
EVAL_TAG=isobbox_pairangle${W}${NBTAG}${DTAG}_soft_${MODE}${SPTAG}_T1_seed0

[ -d "experiments/guitar/$EXP_NAME" ] || { echo "no such run: experiments/guitar/$EXP_NAME"; exit 1; }
echo "=== small analyze: $MODE (w=$W, neighbours=$NB) ==="
echo "    experiments: $EXP_NAME"
echo "    demo:        $DEMO_NAME"
./scripts/small_analyze_isobbox_pairangle.sh "$EXP_NAME" "$DEMO_NAME" "$EVAL_TAG"
