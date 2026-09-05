#!/bin/bash
# One-shot (2026-09-01): re-run the FOUR-arm comparison set once the T=0.001 arm is analysed,
# replacing the three-arm version that actually ran at the 2026-08-31 pod close. CPU/PIL only,
# so it is safe to run alongside the stride-14 GPU driver. Waits on a PID, not a pattern.
set -e
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi
cd /workspace/bsb/scripts
exec ./run_stride7_gridalign_nosdsquare_COMPARISONS.sh
