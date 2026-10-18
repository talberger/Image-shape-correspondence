#!/bin/bash
# Tal 2026-09-27: global->per-seg switch at step 150 (guitar + horse 164135), + guitar global_aniso in8out8 control.
cd ${DISC_ROOT:-/workspace/disc}
MU=globalthenaniso NIN=8 NOUT=8 bash scripts/run_isobbox_mu_loraW.sh > ${DISC_WS:-/workspace}/logs/guitar_globalthenaniso_sw150.log 2>&1
CLASS=horse IMG_TAG=lvis164135 MU=globalthenaniso MU_LR=5e-2 NIN=8 NOUT=8 SUFFIX=_mulr5e-2_in8out8_sw150 \
  bash scripts/run_shicbear_mu_lora.sh > ${DISC_WS:-/workspace}/logs/horse164135_globalthenaniso_sw150.log 2>&1
MU=globalaniso NIN=8 NOUT=8 bash scripts/run_isobbox_mu_loraW.sh > ${DISC_WS:-/workspace}/logs/guitar_globalaniso_in8out8.log 2>&1
echo CHAIN DONE
