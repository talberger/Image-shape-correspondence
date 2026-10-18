"""
Ground-truth geodesic-error metric against the hand-labeled pixel<->vertex evaluation set
(see PROJECT_NOTES.md's "hand-labeled pixel<->vertex evaluation set" thread). For each labeled
pixel: run the SAME soft mapping the project's own analysis/training uses, then measure how far
the predicted location is from Tal's hand-picked ground-truth vertex, in GEODESIC distance on
the mesh surface (looked up in the precomputed analysis/build_geodesic_tables.py table).

Reports TWO variants of "how far", because a softmax mapping does not produce a single vertex
and the choice genuinely matters (both are computed -- they are each cheap, and comparing them
is itself informative):

  (A) hard-snap      -- collapse the softmax to y_hat = sum_j w_j * pos(v_j), snap that to the
                        geometrically nearest actual vertex, look up geo(snap, ground_truth).
                        This is exactly what color_mesh_from_soft_mapping.py already persists as
                        `top1_vertex_idx`, i.e. the same decision every coverage number in this
                        project is built on. Simple, but throws away the distribution: a confident
                        and a hopelessly spread-out prediction can snap to the same vertex.

  (B) expected       -- never collapse: E[geo] = sum_j w_j * geo(v_j, ground_truth), a weighted
                        expectation over the pixel's own top-k candidate vertices. This is the
                        SAME mechanism modules/relational_loss.py's expected_geodesic_distance_3d
                        uses during phase-2 training, and the same reason it gives (a softmax-
                        averaged POSITION is generally off-surface, where geodesic distance is
                        undefined -- see that module's geodesic-variant docstring). Rewards a
                        confident correct prediction over a diffuse one that happens to snap right.

Both are reported raw and as a percentage of the mesh's geodesic diameter (the `diagonal` field
stored in the geodesic table itself, 1.9026 for guitar.obj) -- the raw numbers are in normalized
bounding-sphere units and are hard to read on their own.

NOTE on cost: the softmax weights are NOT recoverable from the saved `_soft_snap_correspondence.pth`
files -- those persist only the already-snapped `top1_vertex_idx` (variant A), the top-k indices and
weights are discarded inside compute_soft_snap. So this script recomputes the mapping from features.
That is cheap here because it only ever touches the handful of LABELED pixels, never the full ~212k
masked set.

Run (evaluate the current-best recipe's phase-2 checkpoint):
cd $DISC_ROOT && python analysis/eval_geodesic_error.py \
  --eval_pairs_path demo/guitar/eval_labels/pixel_vertex_eval_pairs.pth \
  --features_path demo/guitar/features_analyze/dino_sd_poc/guitar_01_pix_dino_sd_features.pth \
  --vertex_features_path experiments/guitar/encoder_dino_sd_dfd_crossview_relational_geodesic_mu3dscale1_phase2/pred_f.pth \
  --run_label "DINOv2 current-best phase-2"
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pixel_to_vertex_nn import load_features, load_vertex_features
from modules.mapping_loss import soft_map_to_vertices
from modules.utils import loadmesh



def _auc(curve, eps_grid, eps_max):
    """Normalized area under acc(eps) over [0, eps_max] -> mean accuracy, in [0, 1]."""
    return float(torch.trapz(curve, eps_grid) / eps_max)


def format_sources(sources, indent=''):
    """Aligned `label : full path` lines for a figure footer.

    Tal, 2026-09-01: an eval figure must be self-describing -- a PNG found on disk weeks later
    has to say what produced it. **Narrowed 2026-09-05, also Tal**: the original six-line block
    (photo features, vertex features, eval pairs, geodesic table, mesh, read setting) was too
    much of the figure, so only the RESULT FILE is printed. That loses nothing -- the .pth holds
    every one of those paths, and `--replot_from <that file>` prints them back -- and the label
    set is still named whenever it is not v1 (see build_sources). Paths are printed whole and
    never elided: a truncated path cannot be pasted back into a command, which is the entire
    point of putting it on the figure.

    `sources` is an ordered dict of label -> path (or None, printed as "(not recorded)" -- result
    files written before 2026-09-01 do not carry every key).
    """
    width = max(len(k) for k in sources)
    return '\n'.join(f'{indent}{k:<{width}} : {v if v else "(not recorded)"}'
                      for k, v in sources.items())


def draw_summary(plot_path, run_label, n, geo_diameter, eps_grid, curves, means,
                 eps_max, variant, sources=None):
    """Render the one-image evaluation summary: per-metric acc(eps) panels + metric table.

    Split out of main() so --replot_from can rebuild the figure from a saved result file
    without re-loading the (large, slow) pixel-feature tensor -- figure tweaks then cost
    seconds instead of minutes.
    """
    # One figure carrying the WHOLE evaluation: acc(eps) curves on top, every scalar
    # metric in a table below. Layout rule (Tal, 2026-08-26): acc(eps) is the ONLY
    # tolerance-dependent metric, so it gets curves + acc@/AUC columns; the mean-error
    # variants are tolerance-independent and contribute one number each.
    # Curves are drawn for (A) -- the canonical variant -- and (C), the DPC-faithful
    # one. (B) is table-only: its value moves with the softmax sharpness, so a (B)
    # curve invites exactly the cross-run comparison that is invalid for it.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    canonical = {'A': 'geometric snap', 'B': 'soft mean distance',
                 'C': 'feature argmax'}[variant]
    curve_variants = ['geometric snap', 'feature argmax', 'soft mean distance']
    styles = {'geometric snap': ('tab:blue', 2.2, '-'),
              'feature argmax': ('tab:orange', 1.6, '--'),
              'soft mean distance': ('tab:green', 1.6, ':')}

    fig = plt.figure(figsize=(12.6, 6.6), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.42, wspace=0.24)

    # One panel PER METRIC (Tal, 2026-08-26): mixing the three in one axes is confusing,
    # and the axes that matters later is method-vs-method -- each panel then holds one
    # curve per run being compared, with the metric held fixed.
    for ci, name in enumerate(curve_variants):
        col, lw, ls = styles[name]
        axc = fig.add_subplot(gs[0, ci])
        axc.plot(100 * eps_grid.numpy(), 100 * curves[name].numpy(),
                 color=col, linewidth=2.1)
        axc.set_title(f'{name}\nAUC {100*_auc(curves[name], eps_grid, eps_max):.1f}%   '
                      f'mean {100*means[name]/geo_diameter:.1f}%',
                      fontsize=10, color=col, weight='bold')
        axc.set_xlabel('error tolerance (% of D)', fontsize=8.5)
        if ci == 0:
            axc.set_ylabel('correspondence accuracy (%)', fontsize=9)
        axc.set_xlim(0, 100 * eps_max); axc.set_ylim(0, 100)
        axc.grid(alpha=0.3); axc.tick_params(labelsize=8)

    fig.suptitle(f'{run_label}          acc(eps),  n={n}', fontsize=11, y=0.995)

    ax_t = fig.add_subplot(gs[1, :]); ax_t.axis('off')
    show_t = [0.01, 0.03, 0.05, 0.10, 0.20]
    col_labels = ['mean err'] + [f'acc@{100*e:g}%' for e in show_t] + ['AUC']
    note = {'geometric snap': 'mesh space',
            'feature argmax': 'feature space',
            'soft mean distance': 'soft, sharpness-dependent'}
    rows, cells = [], []
    for name in ('geometric snap', 'feature argmax', 'soft mean distance'):
        m = means[name]
        r = [f'{m:.4f} ({100*m/geo_diameter:.1f}%)']
        if name in curve_variants:
            for e in show_t:
                idx = int(torch.argmin((eps_grid - e).abs()))
                r.append(f'{100*float(curves[name][idx]):.0f}%')
            r.append(f'{100*_auc(curves[name], eps_grid, eps_max):.1f}%')
        else:
            r += ['--'] * (len(show_t) + 1)  # unreachable now; kept for --variant safety
        rows.append(f'{name}  [{note[name]}]')
        cells.append(r)
    tbl = ax_t.table(cellText=cells, rowLabels=rows, colLabels=col_labels,
                     cellLoc='center', rowLoc='left', loc='upper center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.5)
    for (r_i, _), cell in tbl.get_celld().items():
        if r_i == 0:
            cell.set_text_props(weight='bold')
    ax_t.text(0.5, 0.02,
              f'errors as % of mesh geodesic diameter {geo_diameter:.4f}   |   '
              f'AUC = area under acc(eps) over eps in [0, {100*eps_max:g}%], '
              f'normalized by the range   |   acc(eps) is the only tolerance-dependent '
              f'metric\nsoft mean distance moves with softmax sharpness -- valid at a '
              f'fixed temperature, not across runs that differ in it',
              ha='center', va='center', transform=ax_t.transAxes, fontsize=7.5,
              color='0.35')

    # Sources go BELOW the axes in figure coords (negative y) and are pulled back in by
    # bbox_inches='tight' -- that way the block grows downward with the number of lines instead
    # of stealing height from the panels.
    if sources:
        fig.text(0.005, -0.015, 'sources\n' + format_sources(sources, indent='  '),
                 ha='left', va='top', family='monospace', fontsize=6.5, color='0.25')

    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
    fig.savefig(plot_path, bbox_inches='tight')
    print(f'\nsaved acc(eps) curves + metric table to {plot_path}')



def build_sources(result_path, eval_pairs_path=None):
    """The source block for one evaluated run -- deliberately ONE line (Tal, 2026-09-05: the
    six-line block was too much on every figure). The result .pth carries every path that
    produced the numbers, so pointing at it loses nothing; `--replot_from <that file>` prints
    them all back.

    The label set is the ONE exception and is named only when it is not the v1 default: v1 and
    v2 coexist since 2026-09-05, v2 scores ~2 AUC higher on the same encoder, and a figure that
    does not say which set it used invites exactly that comparison."""
    src = {'result file': result_path}
    if eval_pairs_path and 'v2_edge_inset' in str(eval_pairs_path):
        src['label set  '] = 'EVAL v2 (edge-inset) -- NOT comparable to v1 numbers'
    return src


def main(args):
    if args.replot_from:
        # Rebuild the figure from a saved result file -- no feature loading, no recompute.
        d = torch.load(args.replot_from)
        # Result files written before 2026-09-01 carry only the three path keys; the rest come
        # back as None and print as "(not recorded)" rather than as this run's argparse defaults,
        # which would be a silent lie about what the saved numbers were computed with.
        draw_summary(args.plot_path, args.run_label if args.run_label != '(unlabeled run)'
                     else d.get('run_label', '(unlabeled run)'),
                     len(d['error_hard_snap']), d['geo_diameter'], d['eps_grid'],
                     d['acc_curves'],
                     {'geometric snap': d['mean_error_hard_snap'],
                      'feature argmax': d['mean_error_argmax'],
                      'soft mean distance': d['mean_error_expected']},
                     args.eps_max, args.variant,
                     sources=build_sources(args.replot_from,
                                           d.get('eval_pairs_path')))
        return

    pairs = torch.load(args.eval_pairs_path)
    label_id = pairs['label_id']
    rows, cols = pairs['row'], pairs['col']
    gt_vertex = pairs['vertex_idx']
    n = len(label_id)
    print(f'{n} labeled pixel<->vertex pairs from {args.eval_pairs_path}')

    geo = torch.load(args.geo_table_path, map_location='cpu')
    geo_table, geo_diameter = geo['table'], geo['diagonal']
    print(f'geodesic table {tuple(geo_table.shape)}, mesh geodesic diameter {geo_diameter:.4f}')

    print(f'loading pixel features from {args.features_path} (large file, one-time)...')
    features = load_features(args.features_path)
    vertex_features = load_vertex_features(args.vertex_features_path, features.shape[-1])
    num_vertices = vertex_features.shape[0]
    if geo_table.shape[0] != num_vertices:
        raise ValueError(f'geodesic table is {geo_table.shape[0]}x{geo_table.shape[1]} but '
                          f'{args.vertex_features_path} has {num_vertices} vertices')

    mesh = loadmesh(dir=args.obj_path, name=args.name)
    if mesh.vertices.shape[0] != num_vertices:
        raise ValueError(f'mesh has {mesh.vertices.shape[0]} vertices, expected {num_vertices}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Same normalization compute_soft_snap applies before soft_map_to_vertices -- pixel features
    # must be L2-normalized for the cosine similarity inside it to mean what it says.
    pixel_feats = features[rows, cols].to(device).float()
    pixel_feats = F.normalize(pixel_feats.detach(), dim=-1)
    vert_feats = vertex_features.to(device)
    vert_pos = mesh.vertices.to(device).float()

    y_hat, topk_idx, weights = soft_map_to_vertices(
        pixel_feats, vert_feats, vert_pos, args.k_vertices_per_pixel, return_topk=True,
        temperature=args.temperature)

    # (A) hard-snap: geometric nearest actual vertex to the collapsed point.
    dist2 = (y_hat.unsqueeze(1) - vert_pos.unsqueeze(0)).pow(2).sum(-1)  # [n, V]
    snap_vertex = dist2.argmin(dim=1).cpu()

    gt = gt_vertex.long()
    err_snap = geo_table[snap_vertex, gt]                                # [n]

    # (C) DPC-faithful: argmax over the cosine similarity itself -- no averaging, no
    # geometric snap. This is what DPC's own evaluation code does
    # (models/shape_corr_trainer.py: _prob_to_corr_test takes torch.argmax over the
    # similarity matrix, and acc_mean_dist indexes the target shape with that argmax),
    # i.e. the faithful mesh-form analogue of the paper's Eq. 13. Variant (A) below
    # differs in that it averages the top-k POSITIONS first and then snaps geometrically,
    # so it can be fooled by a symmetric spread of candidates; (C) cannot.
    sim_all = pixel_feats @ F.normalize(vert_feats, dim=-1).T            # [n, V]
    argmax_vertex = sim_all.argmax(dim=1).cpu()
    err_argmax = geo_table[argmax_vertex, gt]                            # [n]

    # (B) expected geodesic over the pixel's own top-k candidates, no collapse.
    topk_cpu, w_cpu = topk_idx.cpu(), weights.cpu().double()
    geo_to_gt = geo_table[topk_cpu, gt.unsqueeze(1)]                     # [n, k]
    err_expected = (w_cpu * geo_to_gt.double()).sum(dim=1).float()       # [n]

    # DPC Eq. 14, mesh form: acc(eps) = fraction of points whose error <= eps * d, with
    # d the mesh's maximal geodesic distance (verified equal to the table's own `diagonal`).
    # Swept 0..eps_max in eps_step increments to match DPC's Figure 6 (0-20%).
    eps_grid = torch.arange(0.0, args.eps_max + 1e-9, args.eps_step, dtype=torch.float64)

    def acc_curve(err):
        rel = (err.double() / geo_diameter).unsqueeze(0)                 # [1, n]
        return (rel <= eps_grid.unsqueeze(1)).double().mean(dim=1)       # [len(eps_grid)]

    def auc(curve):
        # Normalized area under acc(eps) over [0, eps_max] -> a single scalar in [0, 1].
        # Trapezoid over the same grid; dividing by eps_max makes it the MEAN accuracy
        # across the tolerance range, so it is comparable regardless of eps_max/eps_step.
        return float(torch.trapz(curve, eps_grid) / args.eps_max)

    curves = {'feature argmax': acc_curve(err_argmax),
              'geometric snap': acc_curve(err_snap),
              'soft mean distance': acc_curve(err_expected)}

    print(f'\nrun: {args.run_label}')
    print(f'k_vertices_per_pixel = {args.k_vertices_per_pixel}\n')
    print(f'{"label":>5}  {"pixel(row,col)":>16}  {"gt_vert":>7}  '
          f'{"feature argmax":>19}  {"geometric snap":>19}  {"soft mean distance":>19}')
    for i in range(n):
        c, a, b = float(err_argmax[i]), float(err_snap[i]), float(err_expected[i])
        print(f'{int(label_id[i]):>5}  {f"({int(rows[i])},{int(cols[i])})":>16}  '
              f'{int(gt[i]):>7}  '
              f'{c:>9.4f} ({100*c/geo_diameter:>4.1f}%)  '
              f'{a:>9.4f} ({100*a/geo_diameter:>4.1f}%)  '
              f'{b:>9.4f} ({100*b/geo_diameter:>4.1f}%)')

    mean_c = float(err_argmax.mean())
    mean_a, mean_b = float(err_snap.mean()), float(err_expected.mean())
    print(f'\n{"MEAN":>5}  {"":>16}  {"":>7}  '
          f'{mean_c:>9.4f} ({100*mean_c/geo_diameter:>4.1f}%)  '
          f'{mean_a:>9.4f} ({100*mean_a/geo_diameter:>4.1f}%)  '
          f'{mean_b:>9.4f} ({100*mean_b/geo_diameter:>4.1f}%)')
    print(f'\n(percentages are of the mesh geodesic diameter, {geo_diameter:.4f})')

    print(f'\naccuracy at tolerance, acc(eps) = % of the {n} points within eps * '
          f'{geo_diameter:.4f}')
    show = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
    print(f'  {"variant":<15}' + ''.join(f'{100*e:>7.1f}%' for e in show) + f'{"AUC":>10}')
    for name, curve in curves.items():
        cells = ''
        for e in show:
            j = int(torch.argmin((eps_grid - e).abs()))
            cells += f'{100*float(curve[j]):>7.0f}%'
        print(f'  {name:<15}{cells}{100*auc(curve):>9.1f}%')
    print(f'  (AUC = normalized area under acc(eps) over eps in [0, {args.eps_max}], '
          f'i.e. mean accuracy across that range; higher is better)')

    if args.plot_path:
        draw_summary(args.plot_path, args.run_label, n, geo_diameter, eps_grid,
                     curves, {'geometric snap': mean_a, 'feature argmax': mean_c,
                              'soft mean distance': mean_b}, args.eps_max, args.variant,
                     sources=build_sources(args.save_path, args.eval_pairs_path))

    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        torch.save({
            'label_id': label_id, 'row': rows, 'col': cols,
            'gt_vertex_idx': gt, 'snap_vertex_idx': snap_vertex,
            'topk_vertex_idx': topk_cpu, 'topk_weights': weights.cpu(),
            'argmax_vertex_idx': argmax_vertex,
            'error_hard_snap': err_snap, 'error_expected': err_expected,
            'error_argmax': err_argmax,
            'mean_error_hard_snap': mean_a, 'mean_error_expected': mean_b,
            'mean_error_argmax': mean_c,
            'eps_grid': eps_grid,
            'acc_curves': {k: v for k, v in curves.items()},
            'auc': {k: auc(v) for k, v in curves.items()},
            'geo_diameter': geo_diameter,
            'k_vertices_per_pixel': args.k_vertices_per_pixel,
            'run_label': args.run_label,
            'eval_pairs_path': args.eval_pairs_path,
            'features_path': args.features_path,
            'vertex_features_path': args.vertex_features_path,
            # Added 2026-09-01 so --replot_from and compare_eval_runs.py can print the full
            # provenance. `temperature` in particular was never persisted, which is exactly the
            # read-temperature confound the 2026-08-31 notes flagged: a saved .pth gave no way
            # to tell which read T produced it.
            'geo_table_path': args.geo_table_path,
            'obj_path': args.obj_path,
            'temperature': args.temperature,
        }, args.save_path)
        print(f'saved per-point results to {args.save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replot_from', type=str, default=None,
                        help='rebuild the summary figure from a previously saved --save_path '
                             'result file, skipping all computation. Needs --plot_path.')
    parser.add_argument('--eval_pairs_path', type=str, required=False, default=None,
                         help='output of analysis/join_eval_pixel_labels.py')
    parser.add_argument('--features_path', type=str, required=False, default=None,
                         help='real-photo pix features for the run being evaluated')
    parser.add_argument('--vertex_features_path', type=str, required=False, default=None,
                         help="the run's pred_f.pth")
    parser.add_argument('--run_label', type=str, default='(unlabeled run)')
    parser.add_argument('--geo_table_path', type=str,
                         default='experiments/guitar/geodesic_tables/guitar_geo_table_3d.pth')
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')
    parser.add_argument('--k_vertices_per_pixel', type=int, default=10,
                         help='must match the k used elsewhere for comparability (project default 10)')
    parser.add_argument('--temperature', type=float, default=1.0,
                         help='softmax temperature for the soft mapping. 1.0 = the plain softmax '
                               'every previous number in this project was computed with. Lower '
                               'concentrates the weights on the top candidates; ~0.01-0.02 is where '
                               'that actually starts to happen on this project scale (the within-top-k '
                               'similarity gap is ~0.03 median).')
    parser.add_argument('--eps_max', type=float, default=0.20,
                        help='largest error tolerance in the acc(eps) sweep, as a fraction of '
                             'the mesh geodesic diameter. 0.20 matches DPC Figure 6.')
    parser.add_argument('--eps_step', type=float, default=0.01,
                        help='tolerance increment in the acc(eps) sweep (0.01 = 1%% steps).')
    parser.add_argument('--variant', type=str, default='A', choices=['A', 'B', 'C'],
                        help="which error definition the plotted acc(eps) curve uses. 'A' "
                             "(hard-snap) is this project's canonical choice, 2026-08-26: it is "
                             "what color_mesh_from_soft_mapping does everywhere else, and it "
                             "scores better than the DPC-faithful argmax here because the "
                             "top1-vs-top2 similarity margin is only ~0.003, making a single "
                             "argmax noisy. 'C' is the DPC-faithful one, kept in the table for "
                             "comparability with the paper. 'B' should not be used for "
                             "cross-run comparison (sharpness confound).")
    parser.add_argument('--plot_path', type=str, default=None,
                        help='if set, save a DPC-Figure-6-style acc(eps) plot here.')
    parser.add_argument('--save_path', type=str, default=None)
    _a = parser.parse_args()
    if not _a.replot_from:
        missing = [f'--{k}' for k in ('eval_pairs_path', 'features_path', 'vertex_features_path')
                   if getattr(_a, k) is None]
        if missing:
            parser.error('required unless --replot_from is used: ' + ', '.join(missing))
    elif not _a.plot_path:
        parser.error('--replot_from needs --plot_path')
    main(_a)
