"""View-angle distribution scatter: original random sampling vs. --view_sampling fibonacci,
colored by front/back/left/right quadrant (not raw xyz), with an arrow marking the empirically
verified front direction (elev=0, azim=90 deg).

Replaces an earlier ad-hoc (unsaved, 2026-08-15) matplotlib comparison that colored points by
raw (x,y,z) position and marked front with a star -- Tal found the star hard to spot among 1000
scatter points, and asked for quadrant coloring + an arrow instead, plus wanted the underlying
script to actually exist this time.

Coordinate convention: unit_xyz() below is copied verbatim from modules/utils.py's
get_camera_from_view2 -- the ONE function actually used to place cameras for every real render
in this project (training-view generation, front verification, ZR comparison). This is
deliberate: a *different* xyz formula was found in analysis/view_angle_spheres.py (swaps Y/Z
vs. get_camera_from_view2 despite its docstring claiming otherwise) -- confirmed to place the
elev=0/azim=90 front marker at the top pole instead of the horizon. Not repeating that bug here.

Quadrant definition matches the one already used in PROJECT_NOTES.md's corrected-front quadrant
counts: front = azim in [45,135), right = [-45,45), back = [135,225), left = [225,315)
(degrees, azim=0 wrapped to [-180,180) first).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def unit_xyz(elev, azim):
    """elev/azim in radians -> unit-sphere xyz, exactly modules/utils.py:get_camera_from_view2."""
    x = np.cos(elev) * np.cos(azim)
    y = np.sin(elev)
    z = np.cos(elev) * np.sin(azim)
    return x, y, z


def load_angles(path):
    d = torch.load(path, map_location='cpu')
    elev = torch.cat(d['elev']).numpy() if isinstance(d['elev'], list) else d['elev'].numpy()
    azim = torch.cat(d['azim']).numpy() if isinstance(d['azim'], list) else d['azim'].numpy()
    return elev, azim  # radians


def quadrant_labels(azim_rad):
    """front=[45,135), left=[-45,45), back=[-135,-45), right=[135,225) -- matches the
    convention already established in PROJECT_NOTES.md's corrected quadrant counts (verified
    by reproducing its exact 233/125/228/414 random-scheme numbers with this mapping)."""
    azim_deg = (np.degrees(azim_rad) + 180) % 360 - 180  # wrap to [-180, 180)
    labels = np.full(azim_deg.shape, 'right', dtype=object)
    labels[(azim_deg >= 45) & (azim_deg < 135)] = 'front'
    labels[(azim_deg >= -45) & (azim_deg < 45)] = 'left'
    labels[(azim_deg >= -135) & (azim_deg < -45)] = 'back'
    return labels


QUADRANT_COLORS = {'front': '#e15759', 'right': '#4e79a7', 'back': '#59a14f', 'left': '#f1ce63'}
QUADRANT_ORDER = ['front', 'right', 'back', 'left']


def plot_panel(ax, elev, azim, title):
    x, y, z = unit_xyz(elev, azim)
    labels = quadrant_labels(azim)
    for q in QUADRANT_ORDER:
        m = labels == q
        ax.scatter(x[m], y[m], z[m], s=14, color=QUADRANT_COLORS[q], label=q, alpha=0.85,
                   depthshade=True)

    # arrow from center to the verified front direction (elev=0, azim=90deg), pushed a bit
    # past the unit sphere so it's visible over the point cloud
    fx, fy, fz = unit_xyz(0.0, np.pi / 2)
    ax.quiver(0, 0, 0, fx, fy, fz, length=1.35, color='black', linewidth=2.5,
              arrow_length_ratio=0.18)
    ax.text(fx * 1.45, fy * 1.45, fz * 1.45, 'front\n(elev=0, azim=90)', color='black',
            fontsize=9, ha='center', weight='bold')

    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_zlim(-1.4, 1.4)

    counts = {q: int((labels == q).sum()) for q in QUADRANT_ORDER}
    return counts


def add_count_table(fig, ax_pos, counts, n_total):
    ax = fig.add_axes(ax_pos)
    ax.axis('off')
    cell_text = [[f"{counts[q]} ({counts[q] / n_total * 100:.1f}%)" for q in QUADRANT_ORDER]]
    table = ax.table(cellText=cell_text, colLabels=QUADRANT_ORDER, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    for q, cell in zip(QUADRANT_ORDER, [table[0, i] for i in range(4)]):
        cell.set_facecolor(QUADRANT_COLORS[q])
        cell.set_alpha(0.3)


def main(args):
    rand_elev, rand_azim = load_angles(args.random_angles)
    fib_elev, fib_azim = load_angles(args.fib_angles)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('View-angle distribution: original random sampling vs. Fibonacci lattice\n'
                 '(colored by quadrant; front verified empirically at azim=90, elev=0)',
                 fontsize=13)

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    c1 = plot_panel(ax1, rand_elev, rand_azim,
                    f'Original (random, gaussian-around-azim=0)\n(n={len(rand_elev)})')
    ax1.legend(loc='upper left', fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    c2 = plot_panel(ax2, fib_elev, fib_azim, f'Fibonacci lattice (new)\n(n={len(fib_elev)})')
    ax2.legend(loc='upper left', fontsize=8)

    add_count_table(fig, [0.08, 0.02, 0.38, 0.12], c1, len(rand_elev))
    add_count_table(fig, [0.55, 0.02, 0.38, 0.12], c2, len(fib_elev))

    plt.subplots_adjust(top=0.85, bottom=0.18)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=130, bbox_inches='tight')
    print(f'saved to {args.output}')
    print('random:', c1)
    print('fib:   ', c2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_angles', type=str,
                        default='data/guitar/encoder_data/224/random_viewing_angles.pt')
    parser.add_argument('--fib_angles', type=str,
                        default='data/guitar/encoder_data_fib1000/224/random_viewing_angles.pt')
    parser.add_argument('--output', type=str,
                        default='demo/guitar/runs/legacy/bsb_fib1000_ZR/view_angle_comparison_random_vs_fibonacci.png')
    args = parser.parse_args()
    main(args)
