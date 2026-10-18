#!/bin/bash
# One-shot: stop the analysis driver as soon as it finishes T=0.01 and reaches T=0.001.
# Tal's call 2026-08-31: leave the T=0.001 arm for the next pod.
LOG=${DISC_WS:-/workspace}/logs/nosdsquare_ANALYSIS_DRIVER.log
until grep -q "step progression\] T=0.001" "$LOG"; do sleep 10; done
pkill -f "gridalign_nosdsquare_ANALYSIS_ALL"
sleep 3
# kill any T=0.001 analysis child that already started
pkill -f "nosdsquare_T0.001_seed0_analysis.sh"
pkill -f "export_pred_f_l2norm"
echo "STOPPED after T=0.01 at $(date +%H:%M:%S)"
