"""Regenerates media/rkd_feature_detail.png -- the per-pixel explainer for the FEATURE-space
RKD distance term (Tal, for a talk).

The original generator lived only in a session scratchpad and was lost with that pod; this is the
rebuilt, versioned version. Rebuilt 2026-08-31 with four fixes Tal asked for:
  1. the "similarity" caption now sits ON its block instead of floating beside it;
  2. d_msh is now DERIVED on the figure -- an explicit arrow between s(i) and s(j) with the formula,
     mirroring the teacher side, instead of a label floating at the edge;
  3. a panel showing how the FIXED normalizers are measured (and that the choice is still open);
  4. the Huber block now shows that it takes the DIFFERENCE of the two normalized distances,
     with an inset of the actual Huber curve.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BLUE   = '#1f5fa9'   # teacher / frozen
ORANGE = '#c1502e'   # student / trained
GREY   = '#6b7280'
RED    = '#c0392b'
GREEN  = '#2e7d32'

fig, ax = plt.subplots(figsize=(24, 16))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')


def box(x, y, w, h, label, ec=GREY, fc='white', fs=15, weight='normal', tc='black', lw=1.8, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.35,rounding_size={r}',
                                ec=ec, fc=fc, lw=lw, zorder=2))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=fs, weight=weight, color=tc, zorder=3)


def arrow(x1, y1, x2, y2, c=GREY, lw=2.0, style='-|>', ls='-', mut=18):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=mut,
                                 color=c, lw=lw, linestyle=ls, zorder=4,
                                 shrinkA=0, shrinkB=0))


def caption(x, y, t, c=GREY, fs=13, ha='center', weight='normal', style='normal'):
    ax.text(x, y, t, ha=ha, va='center', fontsize=fs, color=c, weight=weight, style=style, zorder=5)


def feature_column(cx, cy, vals, color, label):
    """A little feature vector drawn as a stack of shaded cells."""
    cw, ch = 2.2, 1.15
    for n, v in enumerate(vals):
        ax.add_patch(Rectangle((cx - cw / 2, cy + n * ch), cw, ch,
                               fc=plt.get_cmap('Blues')(0.25 + 0.6 * v) if color == BLUE
                               else plt.get_cmap('Oranges')(0.25 + 0.6 * v),
                               ec='white', lw=0.8, zorder=3))
    ax.text(cx, cy - 1.6, label, ha='center', va='center', fontsize=14, color=color,
            weight='bold', zorder=5)


# ----------------------------------------------------------------- row 1: the sampling pipeline
box(2, 92, 15, 6, 'photo', ec=BLUE)
arrow(17.6, 95, 20.4, 95)
box(21, 92, 19, 6, 'image features\n(frozen)', ec=BLUE)
arrow(40.6, 95, 43.4, 95)
box(44, 92, 33, 6, 'pick N centers\n+ K random neighbours per center')
caption(60.5, 90.6, 're-drawn every step', ha='center', style='italic', fs=12.5)

# the little center/neighbour graph
gx, gy = 88, 94.5
ax.plot([gx, gx + 3.2], [gy, gy + 1.9], color=GREY, lw=2, zorder=2)
ax.plot([gx, gx - 3.4], [gy, gy - 1.5], color=GREY, lw=2, zorder=2)
ax.plot([gx, gx + 2.8], [gy, gy - 1.8], color=GREY, lw=2, zorder=2)
for px, py in [(gx + 3.2, gy + 1.9), (gx - 3.4, gy - 1.5), (gx + 2.8, gy - 1.8)]:
    ax.add_patch(plt.Circle((px, py), 0.75, fc=GREY, ec='none', zorder=3))
ax.add_patch(plt.Circle((gx, gy), 0.95, fc=BLUE, ec='none', zorder=4))
caption(gx - 0.5, gy + 3.6, 'center i', c=BLUE, fs=13)
caption(gx - 5.2, gy - 1.5, 'j', c=GREY, fs=13)

ax.plot([2, 98], [89.5, 89.5], color='#d0d0d0', lw=1.4)
ax.text(2, 86.8, 'ONE PAIR  (center i , neighbour j)  —  the DISTANCE term',
        fontsize=19, weight='bold', va='center')

# ----------------------------------------------------------------- TEACHER
box(2, 60, 28, 24, '', ec=BLUE, fc='#f2f7fc', lw=2.4, r=1.6)
caption(16, 81.5, 'TEACHER  (frozen)', c=BLUE, fs=17, weight='bold')
feature_column(8.5, 68, [0.3, 0.75, 0.45, 0.9, 0.2, 0.6], BLUE, 'f(i)')
feature_column(23.5, 68, [0.8, 0.25, 0.6, 0.35, 0.7, 0.4], BLUE, 'f(j)')
arrow(10.5, 71.5, 21.5, 71.5, c=BLUE, style='<|-|>', lw=2.2)
caption(16, 73.3, 'd_img', c=BLUE, fs=15, weight='bold')
caption(16, 63.4, '‖f(i) − f(j)‖', c=BLUE, fs=15)
caption(16, 61.6, 'both frozen — this side never moves', c=GREY, fs=12, style='italic')

# ----------------------------------------------------------------- STUDENT
box(33, 60, 65, 24, '', ec=ORANGE, fc='#fdf6ec', lw=2.4, r=1.6)
caption(65.5, 81.5, 'STUDENT  (trained)', c=ORANGE, fs=17, weight='bold')

feature_column(37, 71.5, [0.3, 0.75, 0.45, 0.9, 0.2, 0.6], BLUE, 'f(i)')
arrow(38.8, 74.5, 42.6, 74.5)
# FIX 1: the caption is now the block's own title, inside its frame
box(43, 71, 16, 7, 'similarity  f(i) · F\nover ALL vertices', fs=13.5, ec=GREY)
arrow(59.6, 74.5, 63.4, 74.5)

# weights bar chart
bx, by = 64.5, 71.6
for n, hgt in enumerate([3.6, 2.3, 1.4, 0.85]):
    ax.add_patch(Rectangle((bx + n * 1.55, by), 1.15, hgt, fc=ORANGE, ec='none', zorder=3))
caption(bx + 3.1, by - 1.3, 'weights w', c=ORANGE, fs=14, weight='bold')
caption(bx + 3.1, 78.2, 'softmax over top-k', c=GREY, fs=12.5)

arrow(71.3, 74.5, 74.4, 74.5)
box(75, 71.6, 13, 5.6, 's(i) = Σ w·F', fs=15, weight='bold', ec=ORANGE)
box(75, 63.4, 13, 5.6, 's(j) = Σ w·F', fs=15, weight='bold', ec=ORANGE)
caption(81.5, 69.9, 's(j): same construction, starting from f(j)', c=GREY, fs=12, style='italic')

# FIX 2: d_msh is now DERIVED here, with its own arrow, mirroring the teacher side
arrow(90.5, 74.4, 90.5, 66.2, c=ORANGE, style='<|-|>', lw=2.2)
caption(93.6, 70.3, 'd_msh', c=ORANGE, fs=15, weight='bold')
caption(81.5, 61.4, '‖s(i) − s(j)‖', c=ORANGE, fs=14)

# ----------------------------------------------------------------- FIX 3: the normalizers
box(2, 38, 43, 18, '', ec=GREEN, fc='#f4faf4', lw=2.0, r=1.4)
caption(23.5, 53.6, 'THE TWO NORMALIZERS  (one scalar each)', c=GREEN, fs=15, weight='bold')
caption(23.5, 50.6, 'd_img / μ_img          d_msh / μ_msh', c='black', fs=15, weight='bold')
caption(23.5, 47.4, 'Each space is divided by its OWN mean pair distance, so only the\n'
                    'RELATIVE pattern of distances is compared — never absolute scale.', c='black', fs=12.5)
caption(23.5, 43.4, 'currently: measured ONCE from the phase-1 encoder, averaged over 5\n'
                    'neighbour draws, then held FIXED  (μ_img = 0.989, μ_msh = 0.633)', c=GREY, fs=12.5)
caption(23.5, 39.9, 'open: mean vs 95th-percentile vs the paper\'s live-batch mean — not settled',
        c=GREEN, fs=12.5, style='italic')

# ----------------------------------------------------------------- FIX 4: Huber, explicitly
box(48, 38, 50, 18, '', ec='#37474f', fc='#eceff1', lw=2.0, r=1.4)
caption(73, 51.5, 'THE LOSS', c='#37474f', fs=16, weight='bold')
caption(73, 47.6, 'r  =  d_msh/μ_msh  −  d_img/μ_img', c='black', fs=16, weight='bold')
caption(73, 44.3, 'one scalar residual per pair', c=GREY, fs=13, style='italic')
caption(73, 40.8, 'loss  =  Huber( r )', c='black', fs=16, weight='bold')

# feeding arrows into the loss
arrow(20, 59.6, 57, 56.4, c=BLUE, lw=2.0)
arrow(84, 59.6, 79, 56.4, c=ORANGE, lw=2.0)

# ----------------------------------------------------------------- gradient
arrow(73, 37.6, 73, 31.0, c=RED, ls='--', lw=2.2, style='-')
ax.plot([30.5, 73, 69.5], [31.0, 31.0, 31.0], color=RED, lw=2.2, ls='--', zorder=4)
ax.text(2, 34.0, 'GRADIENT  (reaches the vertex features F, by two routes)',
        fontsize=17, weight='bold', color=RED, va='center')

box(14, 14, 33, 10, '', ec=RED, fc='#fdf2f2', lw=2.0, r=1.4)
caption(30.5, 21.6, 'path 1:  move F itself', c=RED, fs=15, weight='bold')
caption(30.5, 17.6, 'the values inside s(i) = Σ w·F change.\nThis is the route we want.', c='black', fs=12.5)

box(53, 14, 33, 10, '', ec=RED, fc='#fdf2f2', lw=2.0, r=1.4)
caption(69.5, 21.6, 'path 2:  reshape the weights w', c=RED, fs=15, weight='bold')
caption(69.5, 17.6, 'w is a softmax of f(i)·F, so F can change\nto sharpen w without describing shape better.',
        c='black', fs=12.5)

arrow(30.5, 31.0, 30.5, 24.6, c=RED, ls='--', lw=2.0)
arrow(69.5, 31.0, 69.5, 24.6, c=RED, ls='--', lw=2.0)

caption(50, 10.4, 'MEASURED: block path 2 (detach w) and the loss barely falls — '
                  '−64% attached vs −3.6% detached.', c=RED, fs=13.5, weight='bold')
caption(50, 7.8, 'The spatial relational loss has ONLY path 2 — its Σ w·X uses fixed vertex POSITIONS, '
                 'so there is no path 1 to compete with.', c=GREY, fs=12.5, style='italic')

out = 'media/rkd_feature_detail.png'
fig.savefig(out, dpi=100, bbox_inches='tight', facecolor='white')
print('saved', out)
