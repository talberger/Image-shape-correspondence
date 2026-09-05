"""
Stack two arbitrary images with the same title/label bands make_run_vs_baseline_comparison.py uses.

That tool compares two runs on a FIXED set of six outputs (PCA, similarity heatmap, analysis
summary; plus the three ZS ones) and deliberately does not touch the back-projection on the
phase-2 side. The back-projection is the image Tal actually judges runs by, and since the
2026-08-31 read-temperature finding it also exists in two versions per run (analyze/ at read T=1
and analyze_readT/ at read T = the training T), so pairing two specific back-projections needs an
explicit choice of file -- hence this thin wrapper rather than another hard-coded job list.

Run:
python analysis/stack_images_labeled.py \
  --top_path  <run>/analyze_readT/back_projection/..._soft_snap_k1_side_by_side.png \
  --top_label "stride 14 + squash, read T=0.01" \
  --bottom_path <liked>/analyze/back_projection/..._soft_snap_k1_side_by_side.png \
  --bottom_label "LIKED: dfd_crossview stride 14 T=0.01" \
  --title "back-projection (checker16, soft-snap k=1), matched read temperature" \
  --output_path <out>/backproj_compare.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.make_run_vs_baseline_comparison import stack


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--top_path', required=True)
    p.add_argument('--top_label', required=True)
    p.add_argument('--bottom_path', required=True)
    p.add_argument('--bottom_label', required=True)
    p.add_argument('--title', required=True)
    p.add_argument('--output_path', required=True)
    args = p.parse_args()

    for path in (args.top_path, args.bottom_path):
        if not os.path.exists(path):
            raise SystemExit(f'missing input image: {path}')

    stack(args.top_path, args.top_label, args.bottom_path, args.bottom_label,
          args.output_path, args.title)


if __name__ == '__main__':
    main()
