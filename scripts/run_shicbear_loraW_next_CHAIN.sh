#!/bin/bash
# 2026-09-17 (Tal agreed): after the L_dist-1/10 sweep -- finer read-time alpha on LORA-W r16, then ranks 2 and 4.
set -uo pipefail
cd /workspace/bsb
while kill -0 107277 2>/dev/null; do sleep 30; done
echo "== r16 finer alpha =="; ALPHAS="0.6 0.65 0.7" bash scripts/run_shicbear_loraW.sh
for R in 2 4; do echo "== rank $R =="; RANK=$R ALPHAS="0.5 0.75" bash scripts/run_shicbear_loraW.sh; done
echo "CHAIN DONE"
