"""
Formula sheet for the evaluation metrics. Diagram only -- reads nothing, computes nothing.

Run: python analysis/make_eval_metrics_explainer.py
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, GREEN, GREY, INK = 'tab:blue', 'tab:orange', 'tab:green', '0.45', '0.12'


def panel(ax, x, y, w, h, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.012',
                                facecolor='white', edgecolor=color, linewidth=1.4, zorder=1))


def main(args):
    fig, ax = plt.subplots(figsize=(12.4, 7.2), dpi=150)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # ---- notation ---------------------------------------------------------------------
    ax.text(0.5, 0.965,
            r'$s_v=\cos(f_p,\,g_v)$      $\mathcal{N}=\mathrm{top}\!-\!k$ by $s$, $k=10$      '
            r'$w_j=\mathrm{softmax}(s_j/T)$      $x_v$ = vertex position      '
            r'$v^*$ = labeled vertex      $D=1.9026$',
            ha='center', fontsize=10.5, color=INK)
    ax.plot([0.03, 0.97], [0.935, 0.935], color='0.85', lw=1)

    # ---- three metrics ----------------------------------------------------------------
    cols = [
        (0.030, BLUE, 'geometric snap',
         [r'$\hat{y}=\sum_{j\in\mathcal{N}} w_j\,x_j$',
          r'$\hat{v}=\arg\min_v \|x_v-\hat{y}\|_2$',
          r'$e=\mathrm{geo}(\hat{v},\,v^*)$'],
         'mesh space'),
        (0.355, ORANGE, 'feature argmax',
         [r'$\hat{v}=\arg\max_v \; s_v$',
          r'$e=\mathrm{geo}(\hat{v},\,v^*)$',
          ''],
         'feature space'),
        (0.680, GREEN, 'soft mean distance',
         [r'$e=\sum_{j\in\mathcal{N}} w_j\,\mathrm{geo}(v_j,\,v^*)$',
          '',
          ''],
         'soft, depends on $T$'),
    ]
    for x, color, name, eqs, tag in cols:
        w = 0.29
        panel(ax, x, 0.500, w, 0.395, color)
        ax.text(x + w / 2, 0.855, name, ha='center', fontsize=13, weight='bold', color=color)
        ax.text(x + w / 2, 0.818, tag, ha='center', fontsize=9, color=GREY, style='italic')
        yy = 0.755
        for eq in eqs:
            if eq:
                ax.text(x + w / 2, yy, eq, ha='center', va='center', fontsize=12.5, color=INK)
            yy -= 0.078
        ax.text(x + w / 2, 0.528, r'$\rightarrow$  one error $e_i$ per labeled point',
                ha='center', fontsize=9, color=GREY)

    # ---- summaries --------------------------------------------------------------------
    ax.text(0.5, 0.462, r'each metric gives $e_1..e_n$,  $n=100$  —  summarized two ways:',
            ha='center', fontsize=10.5, color=INK)

    panel(ax, 0.085, 0.150, 0.35, 0.275, GREY)
    ax.text(0.26, 0.385, 'mean error', ha='center', fontsize=12.5, weight='bold', color=INK)
    ax.text(0.26, 0.288, r'$\bar{e}=\frac{1}{n}\sum_{i=1}^{n} e_i$',
            ha='center', va='center', fontsize=15, color=INK)
    ax.text(0.26, 0.190, 'one number', ha='center', fontsize=9, color=GREY)

    panel(ax, 0.565, 0.150, 0.35, 0.275, GREY)
    ax.text(0.74, 0.385, r'acc($\epsilon$)   and   AUC', ha='center', fontsize=12.5,
            weight='bold', color=INK)
    ax.text(0.74, 0.320, r'$\mathrm{acc}(\epsilon)=\frac{1}{n}\sum_{i=1}^{n}'
                         r'\mathbb{1}\!\left[e_i \leq \epsilon D\right]$',
            ha='center', va='center', fontsize=14, color=INK)
    ax.text(0.74, 0.229, r'$\mathrm{AUC}=\frac{1}{\epsilon_{max}}'
                         r'\int_{0}^{\epsilon_{max}}\mathrm{acc}(\epsilon)\,d\epsilon$',
            ha='center', va='center', fontsize=14, color=INK)
    ax.text(0.74, 0.176, r'$\epsilon = 0..20\%$ in $1\%$ steps', ha='center', fontsize=9,
            color=GREY)

    ax.text(0.5, 0.058,
            r'$\mathrm{geo}$ = geodesic distance on the mesh surface, reported as % of $D$'
            '\n' r'acc($\epsilon$) applies to any of the three — it thresholds $e_i$, '
            r'so every metric has both a mean and a curve',
            ha='center', fontsize=9, color=GREY, linespacing=1.7)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    fig.savefig(args.save_path, bbox_inches='tight', facecolor='white')
    print(f'saved metric formula sheet to {args.save_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_path', type=str,
                    default='./demo/guitar/eval_labels/eval_metrics_explained.png')
    main(ap.parse_args())
