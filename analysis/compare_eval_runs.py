"""
Compare several runs on the ground-truth evaluation: one panel per METRIC, one curve per RUN,
plus a table of every scalar. The companion to analysis/eval_geodesic_error.py -- that script
evaluates one run and saves a result file, this one puts several of those side by side.

Layout follows Tal's call (2026-08-26): the metric is held fixed within a panel and the runs
are what vary, so the comparison being made is unambiguous.

Run:
python analysis/compare_eval_runs.py \
  --runs demo/guitar/eval_labels/geodesic_error_dinov2_currentbest_phase2_v2.pth \
         demo/guitar/eval_labels/eval_T0.001_phase2.pth \
         demo/guitar/eval_labels/eval_dinov3_phase2.pth \
  --labels "baseline T=1" "T=0.001" "DINOv3" \
  --save_path demo/guitar/eval_labels/eval_compare_baseline_T0.001_dinov3.png
"""
import argparse
import os
import sys
import textwrap

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.Normalization.Normalizer import Normalizer
from modules.utils import load_mesh_normalizer_spec


def mesh_frame_scale(obj_path):
    """The factor loadmesh divides this mesh's raw coordinates by.

    Needed to compare runs across MESHES (2026-09-02). Each eval result's errors are in its own
    mesh's normalized frame, and that frame is not shared: loadmesh scales by the max distance from
    the vertex centroid (or, with a sidecar, the bounding-box centre), so the same physical distance
    is a different number on each mesh. Multiplying by this scale puts every run back into the raw
    coordinates the .obj files share, which is the only space in which they ARE comparable.
    """
    verts = []
    with open(obj_path) as fh:
        for line in fh:
            if line.startswith('v '):
                verts.append([float(v) for v in line.split()[1:4]])
    v = torch.tensor(verts, dtype=torch.float32)
    if load_mesh_normalizer_spec(obj_path) == 'bbox':
        return float(Normalizer.get_bounding_sphere_normalizer_bbox(v)._scale)
    return float(Normalizer.get_bounding_sphere_normalizer(v)._scale)

METRICS = ['geometric snap', 'feature argmax', 'soft mean distance']
MEAN_KEY = {'geometric snap': 'mean_error_hard_snap',
            'feature argmax': 'mean_error_argmax',
            'soft mean distance': 'mean_error_expected'}
ERR_KEY = {'geometric snap': 'error_hard_snap',
           'feature argmax': 'error_argmax',
           'soft mean distance': 'error_expected'}
RUN_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:brown']
RUN_STYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
ROW_LABEL_WRAP = 24   # characters per line in a table row label, before wrapping


def auc(curve, eps_grid, eps_max):
    return float(torch.trapz(curve, eps_grid) / eps_max)


def main(args):
    runs = [torch.load(p) for p in args.runs]
    labels = args.labels if args.labels else [d.get('run_label', os.path.basename(p))
                                              for d, p in zip(runs, args.runs)]
    if len(labels) != len(runs):
        raise ValueError(f'{len(labels)} labels for {len(runs)} runs')
    # Row labels sit OUTSIDE the table, to its left, and the three per-metric tables are
    # side by side -- so a long run name used to run into its left-hand neighbour (Tal,
    # 2026-08-30). Wrap onto extra lines instead of letting the label column get wider.
    wrapped_labels = ['\n'.join(textwrap.wrap(lab, ROW_LABEL_WRAP)) or lab for lab in labels]
    label_lines = max(w.count('\n') + 1 for w in wrapped_labels)
    D = runs[0]['geo_diameter']
    mixed = any(abs(d['geo_diameter'] - D) > 1e-6 for d in runs[1:])
    if mixed and not args.rescale_across_meshes:
        raise ValueError(
            'runs disagree on the mesh geodesic diameter -- they were evaluated on different '
            'meshes (or different heat-method settings), so their error units are not the same. '
            'Pass --rescale_across_meshes to convert every run into the raw coordinates the .obj '
            'files share and score them all against one reference length.')
    eps_grid = runs[0]['eps_grid']
    if mixed:
        # Convert each run's errors into raw .obj coordinates, then re-derive its acc(eps) curve
        # against ONE reference length so every panel is on a single denominator.
        # Result files written before 2026-08 do not record obj_path; every one of them predates
        # the isotropic mesh, so the original mesh is the only possible source.
        LEGACY_OBJ = './meshes/guitar.obj'
        obj_paths = []
        for d, lab in zip(runs, labels):
            op = d.get('obj_path')
            if op is None:
                print('  %-34s has no recorded obj_path -- assuming %s (pre-dates the field)'
                      % (lab, LEGACY_OBJ))
                op = LEGACY_OBJ
            obj_paths.append(op)
        scales = [mesh_frame_scale(op) for op in obj_paths]
        ref = max(d['geo_diameter'] * sc for d, sc in zip(runs, scales))
        print('MIXED MESHES -- rescaling to raw .obj coordinates, reference length %.5f' % ref)
        for d, lab, sc, op in zip(runs, labels, scales, obj_paths):
            print('  %-34s %-38s frame scale %.6f, diameter %.5f -> %.5f raw'
                  % (lab, os.path.basename(op), sc, d['geo_diameter'], d['geo_diameter'] * sc))
            for metric in METRICS:
                rel = (d[ERR_KEY[metric]] * sc) / ref
                d['acc_curves'][metric] = torch.stack(
                    [(rel <= t).float().mean() for t in eps_grid])
                d[MEAN_KEY[metric]] = float(rel.mean() * ref)
            d['geo_diameter'] = ref
        D = ref
    n = len(runs[0][ERR_KEY['geometric snap']])

    fig = plt.figure(figsize=(14.0, 7.6), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.3, 0.95 + 0.30 * (label_lines - 1)],
                          hspace=0.52, wspace=0.42)

    for ci, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, ci])
        for ri, (d, lab) in enumerate(zip(runs, labels)):
            curve = d['acc_curves'][metric]
            ax.plot(100 * eps_grid.numpy(), 100 * curve.numpy(),
                    color=RUN_COLORS[ri % len(RUN_COLORS)],
                    linestyle=RUN_STYLES[ri % len(RUN_STYLES)], linewidth=2.0,
                    label=f'{lab}   AUC {100*auc(curve, eps_grid, args.eps_max):.1f}%')
        ax.set_title(metric, fontsize=11, weight='bold')
        ax.set_xlabel('error tolerance (% of D)', fontsize=9)
        if ci == 0:
            ax.set_ylabel('correspondence accuracy (%)', fontsize=9.5)
        ax.set_xlim(0, 100 * args.eps_max); ax.set_ylim(0, 100)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
        ax.legend(loc='lower right', fontsize=7.5)

    # One table PER METRIC, sitting under its own panel (Tal, 2026-08-26): a single
    # combined table mixes metric and run on the same axis and is hard to read.
    show_t = [0.01, 0.03, 0.05, 0.10]
    col_labels = ['mean'] + [f'@{100*e:g}%' for e in show_t] + ['AUC']
    for ci, metric in enumerate(METRICS):
        ax_t = fig.add_subplot(gs[1, ci]); ax_t.axis('off')
        cells, rows, tcolors = [], [], []
        for ri, (d, lab) in enumerate(zip(runs, labels)):
            m = d[MEAN_KEY[metric]]
            curve = d['acc_curves'][metric]
            r = [f'{m:.4f}']   # raw geodesic distance, not % of D (Tal, 2026-08-26)
            for e in show_t:
                j = int(torch.argmin((eps_grid - e).abs()))
                r.append(f'{100*float(curve[j]):.0f}%')
            r.append(f'{100*auc(curve, eps_grid, args.eps_max):.1f}%')
            cells.append(r); rows.append(wrapped_labels[ri])
            tcolors.append(RUN_COLORS[ri % len(RUN_COLORS)])
        tbl = ax_t.table(cellText=cells, rowLabels=rows, colLabels=col_labels,
                         cellLoc='center', rowLoc='right', loc='upper center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(6.6)
        tbl.scale(0.80, 1.30 * label_lines)
        for (r_i, c_i), cell in tbl.get_celld().items():
            cell.set_linewidth(0.6)
            if r_i == 0:
                cell.set_text_props(weight='bold')
            elif c_i == -1:
                cell.set_text_props(color=tcolors[r_i - 1], weight='bold')
        ax_t.set_title(metric, fontsize=9, color='0.3', pad=2)

    fig.suptitle(f'ground-truth evaluation, n={n} labeled points   '
                 f'(errors as % of mesh geodesic diameter D={D:.4f})', fontsize=11, y=0.985)

    # ONE LINE PER RUN: the result .pth path (Tal, 2026-09-05 -- the previous four-line-per-run
    # block was too much figure real estate). Nothing is lost: each result file carries every
    # source path that produced its numbers, and `eval_geodesic_error.py --replot_from <file>`
    # prints them back.
    #
    # THE ONE THING THAT IS NOT DROPPED is the label-set check. Comparing runs evaluated against
    # DIFFERENT label sets is invalid, and since 2026-09-05 two sets exist (v1, and the
    # edge-inset v2 that scores ~2 AUC higher on the same encoder), so silence here would be an
    # actively misleading figure. A mismatch is called out on its own line, in red.
    mismatch = [lab for lab, key in [('eval pairs', 'eval_pairs_path'),
                                     ('geodesic table', 'geo_table_path'), ('mesh', 'obj_path')]
                if len({d.get(key) for d in runs}) > 1]

    y = -0.015
    fig.text(0.005, y, 'sources', ha='left', va='top', family='monospace',
             fontsize=7.0, color='0.25', weight='bold')
    y -= 0.026
    for ri, (lab, path) in enumerate(zip(labels, args.runs)):
        fig.text(0.005, y, f'[{ri + 1}] {lab}  :  {path}', ha='left', va='top',
                 family='monospace', fontsize=6.5,
                 color=RUN_COLORS[ri % len(RUN_COLORS)])
        y -= 0.024
    if mismatch:
        # A label-set difference means two OPPOSITE things depending on the encoder. If every run
        # shares one pred_f, the figure is deliberately measuring the label sets against each
        # other on a fixed encoder (v1 vs v2) -- valid, and worth naming. If the encoders ALSO
        # differ, the run difference and the label-set difference are confounded and the figure
        # cannot be read at all.
        same_encoder = len({d.get('vertex_features_path') for d in runs}) == 1
        msg, col = ((f'note: runs differ in {", ".join(mismatch)} but share one pred_f '
                     f'-- this compares the LABEL SETS, not the runs', 'tab:blue')
                    if same_encoder else
                    (f'WARNING: runs differ in BOTH the encoder and the '
                     f'{", ".join(mismatch)} -- confounded, not a valid comparison', 'tab:red'))
        fig.text(0.005, y - 0.006, msg, ha='left', va='top', family='monospace',
                 fontsize=7.0, color=col, weight='bold')

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    fig.savefig(args.save_path, bbox_inches='tight', facecolor='white')
    print(f'saved comparison to {args.save_path}')

    # printed version, so the numbers land in the log too
    for metric in METRICS:
        print(f'\n{metric}')
        print(f'  {"run":<20}{"mean":>10}' + ''.join(f'{100*e:>8.0f}%' for e in show_t)
              + f'{"AUC":>9}')
        for d, lab in zip(runs, labels):
            m = d[MEAN_KEY[metric]]; curve = d['acc_curves'][metric]
            cells_ = ''.join(
                f'{100*float(curve[int(torch.argmin((eps_grid - e).abs()))]):>8.0f}%'
                for e in show_t)
            print(f'  {lab:<20}{100*m/D:>9.1f}%{cells_}{100*auc(curve, eps_grid, args.eps_max):>8.1f}%')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=str, nargs='+', required=True,
                    help='eval_geodesic_error.py --save_path result files, in plot order')
    ap.add_argument('--labels', type=str, nargs='*', default=None)
    ap.add_argument('--rescale_across_meshes', action='store_true',
                    help='allow runs evaluated on DIFFERENT meshes: convert every run into the raw '
                         '.obj coordinate scale and re-score all of them against one reference '
                         'length. Off by default -- silently mixing frames is exactly the failure '
                         'this guard exists to prevent (2026-09-02).')
    ap.add_argument('--eps_max', type=float, default=0.20)
    ap.add_argument('--save_path', type=str, required=True)
    main(ap.parse_args())
