#!/bin/bash
# Tal, 2026-09-16: two separate bear arms on the MAIN recipe -- (1) learned mu, T=1; (2) closed-form per-segment mu, T=0.01.
cd ${DISC_ROOT:-/workspace/disc}
T=1 MU=learn ./scripts/run_shicbear_main_T.sh
T=0.01 MU=closed ./scripts/run_shicbear_main_T.sh
echo "===== SHIC-BEAR CHAIN DONE ====="
