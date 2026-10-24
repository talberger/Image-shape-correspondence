#!/bin/bash
# XSEG on the 20-bear subset (Tal, 2026-10-24; first multi-instance test of XSEG, run on Run:ai). Four arms per bear, same inputs:
#   twostage = the current LVIS recipe (run_lvis_worker.py: global mu steps 1-149, then per-segment aniso) -- re-run here, not reused
#   perseg   = per-segment only, 16 in + 0 out, no stage 1
#   nofeat   = XSEG split, feature term OFF (8 in-seg spatial + cycle)
#   Fonly    = XSEG split, cross-segment feature RKD (Fonly) at the equal-grad-norm weight from a per-bear ARM=balance run
# Segment map = SAM 1 (mask cache: SAM 3 object mask, amodal where split, merge 1.5%, fillseg), as in the full-bear eval of 10-10.
# A bear with ONE fine segment has no cross pairs: only perseg runs, nofeat / Fonly are left out (n/a) and the summary says so.
#
#   STEP=prep     two-stage via scripts/run_lvis_sam3_batch.sh with --keep_intermediates (keeps the photo features + 2D geo table
#                 the XSEG arms need; ~1.5 GB per bear)
#   STEP=arms     perseg / nofeat / balance + Fonly per bear via scripts/run_xseg_horse.sh (CLASS=bear, NO_FIG=1); resumable
#   STEP=summary  per-bear table + means + paired differences vs twostage -> $LOGS/summary.txt (+ .json)
#   STEP=all      the three in order (default)
# Usage: setsid nohup bash scripts/run_xseg_bears.sh > $DISC_WS/logs/xseg_bears.log 2>&1 < /dev/null &
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
IDS=${IDS:-$(python -c "import json; print(' '.join(map(str, json.load(open('data/densepose_lvis/subset20_bear.json'))['ann_ids'])))")}
NAME=${NAME:-runai_sub20}; STEP=${STEP:-all}
RUNS_IN=data/densepose_lvis/runs_$NAME; RES_IN=data/densepose_lvis/results_$NAME
XRUNS=data/densepose_lvis/runs_xseg_bears_$NAME; XDEMO=demo/shic/xseg_bears_$NAME; LOGS=$DISC_WS/logs/xseg_bears_$NAME
mkdir -p $LOGS
ts() { date -u +%F_%T; }

if [ $STEP = prep ] || [ $STEP = all ]; then
  echo "##### prep: two-stage on $(echo $IDS | wc -w) bears [$(ts)] #####"
  # GEO_TABLE_SUBPROCESS=1: the 2D geo table in a fresh process (the in-process fork pool deadlocks on the Run:ai image, see run_lvis_worker.py)
  GEO_TABLE_SUBPROCESS=${GEO_TABLE_SUBPROCESS:-1} CLASS=bear ANN_IDS="$IDS" RESULTS=$RES_IN RUNS=$RUNS_IN WORKER_ARGS=--keep_intermediates bash scripts/run_lvis_sam3_batch.sh
fi

if [ $STEP = arms ] || [ $STEP = all ]; then
  for id in $IDS; do
    tag=lvis$id; src=$RUNS_IN/$tag; L=$LOGS/$tag; mkdir -p $L
    nseg=$(python -c "import torch; print(int((torch.unique(torch.as_tensor(torch.load('$src/${tag}_fine_segments.pth'))) > 0).sum()))")
    arms="perseg"; [ $nseg -gt 1 ] && arms="perseg nofeat Fonly"
    echo "##### $tag: $nseg segment(s), arms: $arms [$(ts)] #####"
    for arm in $arms; do
      [ -f $XDEMO/bear_$tag/$arm/eval_arms.json ] && { echo "  $arm done, skipped"; continue; }
      t0=$(date +%s)
      run() { CLASS=bear TAG=$tag SRC=$src RUNS=$XRUNS DEMO=$XDEMO LOGS=$L NO_FIG=1 "$@" bash scripts/run_xseg_horse.sh >> $L/driver_$arm.log 2>&1; }
      if [ $arm = Fonly ]; then
        run env ARM=balance || { echo "  balance FAILED, see $L/driver_Fonly.log"; continue; }
        w=$(python -c "
import re, sys
for l in open('$L/balance.log'):
    g = re.match(r'XSEG_WEIGHT mode=Fonly weight_equal=(\S+)', l)
    if g: print('%.4e' % float(g.group(1))); sys.exit()
sys.exit('no XSEG_WEIGHT line for Fonly in $L/balance.log')") || { echo "  no Fonly weight"; continue; }
        echo "  Fonly weight $w"
        run env ARM=Fonly WEIGHT=$w || { echo "  Fonly FAILED, see $L/driver_Fonly.log"; continue; }
      else
        run env ARM=$arm || { echo "  $arm FAILED, see $L/driver_$arm.log"; continue; }
      fi
      echo "  $arm $(python -c "import json; r=json.load(open('$XDEMO/bear_$tag/$arm/eval_arms.json'))['arms']; print(' | '.join('%s %.1f' % (a, v['soft_snap']['all']['mean']) for a, v in r.items()))")  [$(( $(date +%s) - t0 )) s]"
    done
  done
fi

if [ $STEP = summary ] || [ $STEP = all ]; then
  python - $XDEMO "$IDS" $LOGS/summary <<'EOF'
import json, os, sys, math
demo, ids, out = sys.argv[1], sys.argv[2].split(), sys.argv[3]
arms = ['zs', 'twostage', 'perseg', 'nofeat', 'Fonly']
rows = {}
for i in ids:
    r = {}
    for arm in ['perseg', 'nofeat', 'Fonly']:
        p = f'{demo}/bear_lvis{i}/{arm}/eval_arms.json'
        if os.path.exists(p):
            a = json.load(open(p))['arms']
            for k in a: r[k] = a[k]['soft_snap']['all']['mean']
    rows[i] = r
lines = ['soft-snap geodesic error (228-normalised), mean of the GT points per bear; lower = better', '',
         '%-8s ' % 'bear' + ' '.join('%9s' % a for a in arms)]
for i in ids:
    lines.append('%-8s ' % i + ' '.join('%9s' % ('%.1f' % rows[i][a] if a in rows[i] else 'n/a') for a in arms))
def stats(xs):
    n = len(xs); m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else float('nan')
    return n, m, se
lines.append('')
for subset, keep in [('all bears with the arm', lambda r: True), ('multi-segment bears only', lambda r: 'nofeat' in r)]:
    lines.append(f'MEANS ({subset}):')
    for a in arms:
        xs = [rows[i][a] for i in ids if a in rows[i] and keep(rows[i])]
        if xs: n, m, se = stats(xs); lines.append(f'  {a:9s} n={n:2d}  mean {m:5.1f}')
    lines.append(f'  paired difference arm - twostage (negative = arm better):')
    for a in ['zs', 'perseg', 'nofeat', 'Fonly']:
        d = [rows[i][a] - rows[i]['twostage'] for i in ids if a in rows[i] and 'twostage' in rows[i] and keep(rows[i])]
        if len(d) > 1:
            n, m, se = stats(d); better = sum(x < 0 for x in d)
            lines.append(f'  {a:9s} n={n:2d}  {m:+5.2f} (SE {se:.2f}), better on {better}/{n}')
    lines.append('')
d = [rows[i]['Fonly'] - rows[i]['nofeat'] for i in ids if 'Fonly' in rows[i] and 'nofeat' in rows[i]]
if len(d) > 1:
    n, m, se = stats(d); lines.append(f'FEATURE TERM: Fonly - nofeat  n={n}  {m:+.2f} (SE {se:.2f}), Fonly better on {sum(x < 0 for x in d)}/{n}')
open(out + '.txt', 'w').write('\n'.join(lines) + '\n'); json.dump(rows, open(out + '.json', 'w'), indent=1)
print('\n'.join(lines))
EOF
fi
echo "XSEG BEARS DONE [$(ts)]"
