"""Aggregate the per-instance JSONs from the DensePose-LVIS run into the SHIC Table 1 numbers.

scripts/run_lvis_worker.py writes one small JSON per instance (the 7.4GB -> 2.1GB of intermediates
are deleted as soon as that instance is scored, so these JSONs are the entire surviving record of a
132-instance run). This collapses them into the reported figures.

HEADLINE = mean over POINTS, not over instances -- that is what SHIC Table 1 reports and what
analysis/eval_densepose_lvis.py reproduces as 33.09 for their released bear weights. Instances carry
2-3 points each, so the two differ; mean-of-instance-means is printed alongside for reference, never
as the headline.

Also prints the noise context PROJECT_NOTES records for this set, because it is easy to over-read a
difference here: instance-level SE on the full 132 is 2.44, the mean is dominated by large errors
(median 18.7 vs mean 33), and a random 1/3 subset has a 90% range of 27.3-38.9. A gap smaller than
about 2.5 is not a result.

Run:
cd /workspace/bsb && python analysis/aggregate_lvis_results.py --results_dir data/densepose_lvis/results
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='data/densepose_lvis/results')
    ap.add_argument('--expected', type=int, default=132,
                    help='how many instances the full set has, to report coverage')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(args.results_dir, 'lvis*.json'))
                   if not p.endswith('_timing.txt'))
    if not paths:
        raise SystemExit(f'no per-instance JSONs found in {args.results_dir}')

    per_arm = {}          # arm -> rule -> {'points': [...], 'instance_means': [...]}
    n_inst = 0
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        if 'arms' not in r:
            continue
        n_inst += 1
        for arm, body in r['arms'].items():
            for rule in ('soft_snap', 'argmax'):
                if rule not in body:
                    continue
                d = per_arm.setdefault(arm, {}).setdefault(rule, {'points': [], 'instance_means': []})
                pts = body[rule]['per_point']
                d['points'].extend(pts)
                d['instance_means'].append(float(np.mean(pts)))

    print(f'{n_inst}/{args.expected} instances scored  ({args.results_dir})')
    print(f'{"arm":>10} {"rule":>10} {"MEAN/pt":>9} {"median":>8} {"mean/inst":>10} {"SE/inst":>8} {"n_pts":>6}')
    out = {'n_instances': n_inst, 'expected': args.expected, 'arms': {}}
    for arm in sorted(per_arm):
        for rule in ('soft_snap', 'argmax'):
            if rule not in per_arm[arm]:
                continue
            pts = np.asarray(per_arm[arm][rule]['points'], dtype=np.float64)
            ims = np.asarray(per_arm[arm][rule]['instance_means'], dtype=np.float64)
            se = float(ims.std(ddof=1) / np.sqrt(len(ims))) if len(ims) > 1 else float('nan')
            print(f'{arm:>10} {rule:>10} {pts.mean():>9.2f} {np.median(pts):>8.2f} '
                  f'{ims.mean():>10.2f} {se:>8.2f} {len(pts):>6}')
            out['arms'].setdefault(arm, {})[rule] = {
                'mean_point': float(pts.mean()), 'median_point': float(np.median(pts)),
                'mean_instance': float(ims.mean()), 'se_instance': se, 'n_points': int(len(pts))}

    print('\nreference (SHIC arXiv 2407.18907 Table 1, bear): SHIC 33.0 | CSE supervised 35.7 | '
          'zero-shot SD-DINO 48.2')
    print('reproduced here with SHIC\'s released weights: 33.09')
    print('NOISE: instance-level SE on the full 132 is 2.44; median 18.7 vs mean 33 (the mean is '
          'driven by left/right and front/back flips). Treat gaps under ~2.5 as not a result.')
    if n_inst < args.expected:
        print(f'WARNING: only {n_inst} of {args.expected} instances present -- this is a PARTIAL '
              f'number and a random 1/3 subset has a 90% range of 27.3-38.9, so it is not '
              f'comparable to 33.0 yet.')

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(out, f, indent=1)
        print(f'wrote {args.out_json}')


if __name__ == '__main__':
    main()
