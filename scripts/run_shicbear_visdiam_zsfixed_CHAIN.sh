#!/bin/bash
# Tal, 2026-09-16, bear proportions: two separate one-change arms on the MAIN recipe (T=0.01).
#  idea 1: d_3D normalized by the VISIBLE part's geodesic diameter 2.00053 (0.856 D; silhouette-matched views)
#  idea 2: per-segment mu FIXED from the ZS mapping (p99 extents)
cd ${DISC_ROOT:-/workspace/disc}
T=0.01 MU=learn VISDIAM3D=2.00053 ./scripts/run_shicbear_main_T.sh
T=0.01 MU=zsfixed ./scripts/run_shicbear_main_T.sh
echo "===== SHIC-BEAR VISDIAM+ZSFIXED CHAIN DONE ====="
