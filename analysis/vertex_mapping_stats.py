"""
Vertex-mapping distribution: given a *_soft_snap_correspondence.pth (the per-pixel hard-snap
vertex assignment from color_mesh_from_soft_mapping.py -- soft_map_to_vertices's softmax-weighted
3D point, snapped to the nearest actual mesh vertex), show how many pixels landed on each vertex
across the WHOLE object (coarse mask, no per-segment breakdown -- Tal's explicit call, since a
segment breakdown needs a notion of "part" and only fine_segments has one, which he doesn't want
here).

Two views of the same per-vertex pixel counts (torch.bincount over top1_vertex_idx):
  1. histogram: X = pixels mapped to a vertex, Y = number of vertices with that count (log Y --
     most vertices sit at low counts, a few at very high counts).
  2. concentration curve: covered vertices sorted by descending pixel count, X = rank, Y = pixel
     count (log Y) -- shows how fast the counts drop off, i.e. how few vertices absorb how much of
     the mask.

Run:
cd $DISC_ROOT && python analysis/vertex_mapping_stats.py \
  --correspondence_path ./demo/guitar/runs/legacy/bsb_relational_geodesic/analyze/color_mesh_from_soft_mapping/guitar_01_click_130_140_sam_mask_2_soft_snap_correspondence.pth
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ACCENT = '#2a78d6'


def load_correspondence(path):
    d = torch.load(path, map_location='cpu')
    top1 = d['top1_vertex_idx']
    if top1.dim() != 1:
        raise ValueError(f'expected top1_vertex_idx of shape [n_pixels], got {tuple(top1.shape)}')
    return top1, int(d['num_vertices'])


def compute_counts(top1_vertex_idx, num_vertices):
    return torch.bincount(top1_vertex_idx, minlength=num_vertices)


def print_summary(counts, n_pixels, num_vertices):
    covered = counts > 0
    num_covered = int(covered.sum())
    covered_counts = counts[covered]

    print(f'n_pixels={n_pixels}, num_vertices={num_vertices}')
    print(f'covered vertices: {num_covered}/{num_vertices} ({100.0 * num_covered / num_vertices:.1f}%)')
    print(f'mean pixels/covered-vertex: {n_pixels / num_covered:.1f}')
    print(f'count stats over covered vertices: min={covered_counts.min().item()}, '
          f'median={covered_counts.median().item():.0f}, max={covered_counts.max().item()}')

    sorted_counts, _ = torch.sort(covered_counts, descending=True)
    cum = torch.cumsum(sorted_counts, dim=0)
    print('concentration:')
    for pct in [0.01, 0.05, 0.10, 0.25]:
        k = max(1, int(pct * num_covered))
        share = cum[k - 1].item() / n_pixels * 100.0
        print(f'  top {pct * 100:.0f}% of covered vertices ({k}) absorb {share:.1f}% of all pixels')


def plot_histogram(ax, covered_counts, cutoff, bin_width=50):
    """Linear-scale histogram, clipped at `cutoff` so the long tail of a handful of
    heavily-overloaded vertices doesn't compress the rest of the chart into one bar (the
    trade-off Tal chose over a log axis). Everything above `cutoff` is folded into one
    separate, visually distinct overflow bar with its own count/range label."""
    counts_np = covered_counts.numpy()
    in_range = counts_np[counts_np <= cutoff]
    overflow = counts_np[counts_np > cutoff]

    bin_edges = np.arange(0, cutoff + bin_width, bin_width)
    ax.hist(in_range, bins=bin_edges, color=ACCENT, edgecolor='white', linewidth=0.5, zorder=3)

    xticks = list(np.arange(0, cutoff + 1, 300))
    xticklabels = [str(int(t)) for t in xticks]
    if len(overflow) > 0:
        gap_x = cutoff + bin_width * 1.5
        ax.bar(gap_x, len(overflow), width=bin_width, color='#9a9a9a', edgecolor='white',
               linewidth=0.5, hatch='///', zorder=3)
        ax.annotate(f'{len(overflow)} vertices\nup to {int(counts_np.max())}px',
                    xy=(gap_x, len(overflow)), xytext=(0, 6), textcoords='offset points',
                    ha='center', fontsize=8, color='#555555')
        xticks.append(gap_x)
        xticklabels.append(f'≥{cutoff}')
        ax.set_xlim(-bin_width, gap_x + bin_width * 2.5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('pixels mapped to a vertex')
    ax.set_ylabel('number of vertices')
    ax.set_title('distribution of pixels-per-vertex')
    ax.grid(True, axis='y', color='#e0e0e0', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def plot_concentration(ax, covered_counts):
    sorted_counts, _ = torch.sort(covered_counts, descending=True)
    rank = np.arange(1, sorted_counts.numel() + 1)
    ax.plot(rank, sorted_counts.numpy(), color=ACCENT, linewidth=2)
    ax.set_xlabel('vertex rank (sorted by pixel count, descending)')
    ax.set_ylabel('pixels mapped to that vertex')
    ax.set_title('concentration across covered vertices')
    ax.grid(True, axis='both', color='#e0e0e0', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--correspondence_path', type=str, required=True,
                         help='a *_soft_snap_correspondence.pth from color_mesh_from_soft_mapping.py')
    parser.add_argument('--output_path', type=str, default=None,
                         help='defaults to <correspondence_path without extension>_vertex_mapping_stats.png '
                              '-- same directory as the correspondence file, per project convention')
    parser.add_argument('--histogram_max_pixels', type=int, default=None,
                         help='linear-scale histogram X-axis is clipped here, with a separate overflow '
                              'bar for everything above (Tal\'s call, over a log axis). Defaults to the '
                              '95th percentile of covered-vertex pixel counts, rounded up to the nearest 100.')
    args = parser.parse_args()

    output_path = args.output_path or (
        os.path.splitext(args.correspondence_path)[0] + '_vertex_mapping_stats.png')

    top1, num_vertices = load_correspondence(args.correspondence_path)
    n_pixels = top1.shape[0]
    counts = compute_counts(top1, num_vertices)

    print_summary(counts, n_pixels, num_vertices)

    covered_counts = counts[counts > 0]
    cutoff = args.histogram_max_pixels
    if cutoff is None:
        p95 = np.percentile(covered_counts.numpy(), 95)
        cutoff = max(100, int(np.ceil(p95 / 100.0) * 100))
    print(f'histogram clipped at {cutoff}px (95th percentile rounded up), overflow bar covers the rest')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_histogram(ax1, covered_counts, cutoff)
    plot_concentration(ax2, covered_counts)
    fig.suptitle(f'vertex-mapping distribution ({os.path.basename(args.correspondence_path)})', fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=110)
    plt.close(fig)
    print(f'saved to {output_path}')


if __name__ == '__main__':
    main()
