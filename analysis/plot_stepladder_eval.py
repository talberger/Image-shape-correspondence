"""
Plot eval error against phase-2 training step -- the shape of what phase-2 does to a good input.

2026-09-08. Phase-2 takes the baseline from 2.7% mean error at ZS to 4.7% at step 300, and the open
question is whether that is the relational loss genuinely pulling there or a bug. The two look
different here: a gradual slide over 300 steps says the loss; a jump inside the first 25 steps
followed by a flat line says initialization or a bug. Step 0 is the ZS eval.

Reads the per-step eval .pth files that run_isobbox_baseline_stepladder_EVAL.sh writes, plus the
ZS eval as step 0, and draws mean error and AUC for all three metrics.

Run:
cd /workspace/bsb && python analysis/plot_stepladder_eval.py \
  --step_dir ./experiments/guitar/encoder_isobbox_sdsquare_fuse16_T1_seed0_phase2_stepladder/stepladder \
  --zs_eval demo/guitar/eval_labels/v2_edge_inset/isobbox/eval_isobbox_ZS_sdsquare_fuse16_v2.pth \
  --output_path <run>/stepladder_eval_curve.png
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METRICS = ['geometric snap', 'feature argmax', 'soft mean distance']
MEAN_KEY = {'geometric snap': 'mean_error_hard_snap',
            'feature argmax': 'mean_error_argmax',
            'soft mean distance': 'mean_error_expected'}
COLORS = {'geometric snap': '#1f77b4', 'feature argmax': '#d62728', 'soft mean distance': '#2ca02c'}


def load_point(path):
    d = torch.load(path, map_location='cpu')
    diameter = float(d['geo_diameter'])
    return ({m: 100 * float(d[MEAN_KEY[m]]) / diameter for m in METRICS},
            {m: 100 * float(d['auc'][m]) for m in METRICS})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--step_dir', required=True)
    p.add_argument('--zs_eval', required=True, help='the ZS eval .pth, plotted as step 0')
    p.add_argument('--output_path', required=True)
    p.add_argument('--title', default='phase-2 step ladder -- baseline sdsquare_fuse16, T=1 seed0')
    args = p.parse_args()

    points = []
    zs_mean, zs_auc = load_point(args.zs_eval)
    points.append((0, zs_mean, zs_auc))
    for path in glob.glob(os.path.join(args.step_dir, 'eval_step*_v2.pth')):
        step = int(re.search(r'eval_step(\d+)_v2\.pth$', path).group(1))
        mean, auc = load_point(path)
        points.append((step, mean, auc))
    points.sort(key=lambda t: t[0])
    if len(points) < 2:
        raise SystemExit(f'found only {len(points)} eval points in {args.step_dir}')
    steps = [t[0] for t in points]

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for m in METRICS:
        axs[0].plot(steps, [t[1][m] for t in points], 'o-', color=COLORS[m], label=m, markersize=4)
        axs[1].plot(steps, [t[2][m] for t in points], 'o-', color=COLORS[m], label=m, markersize=4)
    axs[0].set_ylabel('mean error, % of mesh geodesic diameter (lower is better)')
    axs[1].set_ylabel('AUC, % (higher is better)')
    for ax in axs:
        ax.set_xlabel('phase-2 step  (0 = ZS, phase-1 only)')
        ax.axvline(0, color='#888888', linewidth=0.8, linestyle=':')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(args.title, fontsize=13)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    fig.savefig(args.output_path, dpi=130)
    plt.close(fig)
    print(f'saved {args.output_path}')

    print(f'\n{"step":>6}' + ''.join(f'{m:>22}' for m in METRICS))
    print(f'{"":>6}' + ''.join(f'{"mean err / AUC":>22}' for _ in METRICS))
    for step, mean, auc in points:
        row = ''.join(f'{mean[m]:>13.2f}% /{auc[m]:>6.1f}' for m in METRICS)
        print(f'{step:>6}{row}')


if __name__ == '__main__':
    main()
