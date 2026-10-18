"""
Generalized from the 2026-08-25 DINOv3-vs-baseline one-off (Tal's request: "we'll want this for
every new run going forward, worth keeping"). Stacks a new run's ZS/phase-2 outputs directly above
the current-best-recipe baseline's own equivalents, with a title bar + a labeled header on each
half naming its source path and coverage number, so both can be visually compared in one image.
6 outputs total (3 ZS + 3 phase-2), matching this project's own standard ZS/full_analyze output
structure -- any run missing a ZS or phase-2 dir just skips that half's 3 images, doesn't error.

Assumes both the new run and the baseline used this project's standard pipeline scripts, i.e. each
--*_zs_dir has pixel_to_vertex_nn/, back_projection/, features_analyze/{pca,sim_heatmap}/ (produced
by a run_*_ZS.sh script) and each --*_phase2_dir has analyze/features_analyze/{pca,sim_heatmap}/ +
analysis_summary.png (produced by a run_*_full_analyze.sh script) -- same filenames throughout this
project's history, no per-run naming variation.

Coverage numbers are NOT recomputed here (this is a visualization tool, not an analysis one) -- pass
them in explicitly (from color_mesh_from_soft_mapping.py's own printed "vertex coverage: X/N" line)
so the image labels are never silently stale or wrong.

Baseline defaults point at whichever recipe PROJECT_NOTES.md's top-of-file pointer currently calls
"current best" -- as of 2026-08-25 that's encoder_dino_sd_dfd_crossview_{phase1,relational_geodesic_
mu3dscale1_phase2}, coverage 603/4036 (14.94%) ZS / 1091/4036 (27.03%) phase-2. Update these defaults
(and the docstring note above) whenever PROJECT_NOTES.md promotes a new recipe -- the baseline label/
coverage the script actually used is always printed at run time, so a stale default is loud, not
silent.

Run (DINOv3 example, reproduces the 2026-08-25 one-off exactly):
cd $DISC_ROOT && python analysis/make_run_vs_baseline_comparison.py \
  --run_label "DINOv3 swap" \
  --run_zs_dir demo/guitar/runs/legacy/bsb_sd_dino_dinov3_dfd_crossview_frontcentered_ZS \
  --run_zs_coverage "465/4036 = 11.52%" \
  --run_phase2_dir demo/guitar/runs/legacy/bsb_sd_dino_dinov3_dfd_crossview_relational_geodesic_mu3dscale1_phase2 \
  --run_phase2_coverage "903/4036 = 22.37%"
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
HEADER_H = 40
TITLE_H = 36

BASELINE_LABEL_DEFAULT = 'stride14 + squash (no sd_square), T=0.01, train_seed 0 -- CURRENT BEST RECIPE'
BASELINE_ZS_DIR_DEFAULT = 'demo/guitar/runs/legacy/bsb_sd_dino_dfd_crossview_frontcentered_ZS'
BASELINE_ZS_COVERAGE_DEFAULT = '603/4036 = 14.94%'
# Promoted 2026-09-01 (Tal), on the back-projection rather than on eval -- see PROJECT_NOTES.md
# top of file. NOTE the approved renders are the run's analyze_readT/ pass (read T = 0.01);
# analyze/ (read T=1) is what this tool's PCA/simheatmap/summary jobs read, which is fine because
# those three are read-temperature independent. Coverage below is the read-T=0.01 number.
BASELINE_PHASE2_DIR_DEFAULT = 'demo/guitar/runs/oldmesh_sweeps/stride14/bsb_sd_dino_stride14_nosdsquare_T0.01_seed0_phase2'
BASELINE_PHASE2_COVERAGE_DEFAULT = '1230/4036 = 30.48%'


# 2026-09-14 (Tal): the labels carry every flag now, so two rows can differ in one token inside a very
# long line and the eye cannot find it. `shared_tokens` = the words a label has in common with the OTHER
# rows of the same figure: those are drawn DIMMED, everything else in DIFF_FG (yellow). Pass shared=None
# (the default) to draw the whole line in fg, exactly as before.
DIFF_FG = (255, 214, 10)
DIM_FG = (188, 188, 188)


def _tok_key(tok):
    """Trailing/leading punctuation is layout, not content: `stride 5,` and `stride 5` are the same
    word, and without this the comma alone would light `5` up as a difference."""
    return tok.strip(',.;:()[]').lower()


def is_reference_label(label):
    """A row that is a fixed REFERENCE, not one of the arms being compared: ZS or the baseline.

    2026-09-14 (Tal): the highlighting is meant to show what differs BETWEEN THE ARMS, so these rows
    are left out of the comparison and drawn plain -- their wording shares nothing with the arms and
    would otherwise both empty the shared set and light their own line up.
    """
    l = (label or '').strip().lower()
    return l.startswith('zs') or l.startswith('baseline') or 'phase-1 only' in l


def shared_tokens(labels):
    """Tokens common to MOST labels of a figure -- i.e. the part that is NOT the difference.

    Not the intersection of all of them: rows still differ in how many words they spend on the same
    thing, so a token counts as shared when at least half the rows carry it; with two rows that is
    still both of them. Reference rows (ZS / baseline) are excluded entirely.
    """
    labels = [l for l in labels if l and not is_reference_label(l)]
    sets = [{_tok_key(t) for t in l.split()} for l in labels if l]
    if not sets:
        return set()
    need = max(2, (len(sets) + 1) // 2)
    counts = {}
    for s in sets:
        for t in s:
            counts[t] = counts.get(t, 0) + 1
    return {t for t, c in counts.items() if c >= need}


def label_band(width, text, font_size=22, bg=(30, 30, 30), fg=(255, 255, 255), shared=None):
    band = Image.new('RGB', (width, HEADER_H), bg)
    draw = ImageDraw.Draw(band)
    max_w = width - 16
    size = font_size
    while size > 10:
        font = ImageFont.truetype(FONT_PATH, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_w:
            break
        size -= 1
    else:
        font = ImageFont.truetype(FONT_PATH, 10)
        bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    y = (HEADER_H - th) / 2 - bbox[1]
    if not shared:
        draw.text(((width - tw) / 2, y), text, font=font, fill=fg)
        return band
    # token-by-token, so the differing words can carry their own colour. Widths are measured on the
    # same font, and the space is drawn as advance width only -- the line lands where the one-shot
    # draw.text above would have put it.
    x = (width - tw) / 2
    space = draw.textlength(' ', font=font)
    for i, tok in enumerate(text.split(' ')):
        if tok:
            draw.text((x, y), tok, font=font, fill=(DIM_FG if _tok_key(tok) in shared else DIFF_FG))
            x += draw.textlength(tok, font=font)
        if i != len(text.split(' ')) - 1:
            x += space
    return band


SOURCE_H = 30


def source_band(width, path):
    """A second, thinner band naming WHERE a panel came from.

    Added 2026-09-03 (Tal): the coloured header says what a panel IS, but nothing on the figure said
    which run directory produced it, so a comparison could not be traced back without re-reading the
    command that built it. This prints the run folder as an ABSOLUTE path (Tal works in VS Code,
    where relative paths do not open), falling back to the image's own directory when the path does
    not sit under a bsb_<run> folder.
    """
    ap = os.path.abspath(path)
    parts = ap.split(os.sep)
    run_dir = os.path.dirname(ap)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith('bsb_') or parts[i].startswith('encoder_'):
            run_dir = os.sep.join(parts[:i + 1])
            break
    band = Image.new('RGB', (width, SOURCE_H), (245, 245, 245))
    draw = ImageDraw.Draw(band)
    text = f'source: {run_dir}'
    max_w = width - 16
    size = 17
    while size > 8:
        font = ImageFont.truetype(FONT_PATH, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_w:
            break
        size -= 1
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) / 2, (SOURCE_H - th) / 2 - bbox[1]), text, font=font, fill=(60, 60, 60))
    return band


def stack(top_path, top_label, bottom_path, bottom_label, out_path, title):
    top = Image.open(top_path).convert('RGB')
    bot = Image.open(bottom_path).convert('RGB')
    width = max(top.width, bot.width)

    def pad(im):
        if im.width == width:
            return im
        canvas = Image.new('RGB', (width, im.height), (255, 255, 255))
        canvas.paste(im, ((width - im.width) // 2, 0))
        return canvas

    top, bot = pad(top), pad(bot)

    title_band = label_band(width, title, font_size=26, bg=(10, 10, 60)).resize((width, TITLE_H))
    # 2026-09-14 (Tal): dim the tokens both labels share, highlight the ones that differ
    common = shared_tokens([top_label, bottom_label])
    top_header = label_band(width, top_label, bg=(120, 30, 30),
                            shared=None if is_reference_label(top_label) else common)
    bot_header = label_band(width, bottom_label, bg=(20, 90, 20),
                            shared=None if is_reference_label(bottom_label) else common)

    top_source = source_band(width, top_path)
    bot_source = source_band(width, bottom_path)

    total_h = (TITLE_H + HEADER_H + top.height + SOURCE_H
               + HEADER_H + bot.height + SOURCE_H)
    canvas = Image.new('RGB', (width, total_h), (255, 255, 255))
    y = 0
    canvas.paste(title_band, (0, y)); y += TITLE_H
    canvas.paste(top_header, (0, y)); y += HEADER_H
    canvas.paste(top, (0, y)); y += top.height
    canvas.paste(top_source, (0, y)); y += SOURCE_H
    canvas.paste(bot_header, (0, y)); y += HEADER_H
    canvas.paste(bot, (0, y)); y += bot.height
    canvas.paste(bot_source, (0, y)); y += SOURCE_H

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    print(f'saved {out_path}  ({width}x{total_h})')


def run_jobs(jobs, run_label, base_label, skip_missing):
    """Stack each job. With skip_missing, a job whose input does not exist is reported and skipped
    instead of raising -- small_analyze.sh deliberately does not build every panel full_analyze does."""
    for top_path, bottom_path, out_path, title in jobs:
        if skip_missing:
            absent = [p for p in (top_path, bottom_path) if not os.path.exists(p)]
            if absent:
                print(f'SKIP {os.path.basename(out_path)} -- missing: ' + ', '.join(absent))
                continue
        stack(top_path, run_label, bottom_path, base_label, out_path, title)


def main(args):
    if args.run_zs_dir:
        run_label = f'{args.run_label} -- ZS (phase-1 only), {args.run_zs_coverage} coverage  |  {args.run_zs_dir}'
        base_label = f'{args.baseline_label} -- ZS (phase-1 only), {args.baseline_zs_coverage} coverage  |  {args.baseline_zs_dir}'
        print(f'ZS baseline: {args.baseline_zs_dir} ({args.baseline_zs_coverage})')
        out_dir = args.zs_out_dir or os.path.join(args.run_zs_dir, 'comparison_vs_baseline')
        jobs = [
            (
                f'{args.run_zs_dir}/back_projection/guitar_01_click_270_370_sam_mask_2_backproj_guitar_mesh_checker16_color_k1_side_by_side.png',
                f'{args.baseline_zs_dir}/back_projection/guitar_01_click_270_370_sam_mask_2_backproj_guitar_mesh_checker16_color_k1_side_by_side.png',
                f'{out_dir}/1_back_projection_k1_side_by_side_compare.png',
                f'ZS back-projection (checkerboard, k=1) -- {args.run_label} (top) vs {args.baseline_label} (bottom)',
            ),
            (
                f'{args.run_zs_dir}/features_analyze/pca/guitar_mesh_pca_shared_color_side_by_side.png',
                f'{args.baseline_zs_dir}/features_analyze/pca/guitar_mesh_pca_shared_color_side_by_side.png',
                f'{out_dir}/2_pca_shared_color_side_by_side_compare.png',
                f'ZS PCA (shared basis) -- {args.run_label} (top) vs {args.baseline_label} (bottom)',
            ),
            (
                f'{args.run_zs_dir}/features_analyze/sim_heatmap/simheatmap_grid.png',
                f'{args.baseline_zs_dir}/features_analyze/sim_heatmap/simheatmap_grid.png',
                f'{out_dir}/3_simheatmap_grid_compare.png',
                f'ZS similarity heatmap (4+4 query set) -- {args.run_label} (top) vs {args.baseline_label} (bottom)',
            ),
        ]
        run_jobs(jobs, run_label, base_label, args.skip_missing)
    else:
        print('--run_zs_dir not given, skipping the 3 ZS comparisons')

    if args.run_phase2_dir:
        run_label = f'{args.run_label} -- phase-2, {args.run_phase2_coverage} coverage  |  {args.run_phase2_dir}'
        base_label = f'{args.baseline_label} -- phase-2, {args.baseline_phase2_coverage} coverage  |  {args.baseline_phase2_dir}'
        print(f'phase-2 baseline: {args.baseline_phase2_dir} ({args.baseline_phase2_coverage})')
        out_dir = args.phase2_out_dir or os.path.join(args.run_phase2_dir, 'comparison_vs_baseline')
        jobs = [
            (
                f'{args.run_phase2_dir}/analyze/features_analyze/pca/guitar_mesh_pca_shared_color_side_by_side.png',
                f'{args.baseline_phase2_dir}/analyze/features_analyze/pca/guitar_mesh_pca_shared_color_side_by_side.png',
                f'{out_dir}/4_pca_shared_color_side_by_side_compare.png',
                f'Phase-2 PCA (shared basis) -- {args.run_label} (top) vs {args.baseline_label} (bottom)',
            ),
            (
                f'{args.run_phase2_dir}/analyze/features_analyze/sim_heatmap/simheatmap_grid.png',
                f'{args.baseline_phase2_dir}/analyze/features_analyze/sim_heatmap/simheatmap_grid.png',
                f'{out_dir}/5_simheatmap_grid_compare.png',
                f'Phase-2 similarity heatmap (4+4 query set) -- {args.run_label} (top) vs {args.baseline_label} (bottom)',
            ),
            (
                f'{args.run_phase2_dir}/analysis_summary.png',
                f'{args.baseline_phase2_dir}/analysis_summary.png',
                f'{out_dir}/6_analysis_summary_compare.png',
                f'Phase-2 full analysis summary -- {args.run_label} (top) vs {args.baseline_label} (bottom)',
            ),
        ]
        run_jobs(jobs, run_label, base_label, args.skip_missing)
    else:
        print('--run_phase2_dir not given, skipping the 3 phase-2 comparisons')

    print('ALL DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_label', type=str, required=True,
                         help='short name for the new run, e.g. "DINOv3 swap"')
    parser.add_argument('--run_zs_dir', type=str, default=None,
                         help='new run\'s ZS output dir (from a run_*_ZS.sh script); omit to skip ZS comparisons')
    parser.add_argument('--run_zs_coverage', type=str, default=None,
                         help='e.g. "465/4036 = 11.52%%"; required if --run_zs_dir given')
    parser.add_argument('--run_phase2_dir', type=str, default=None,
                         help='new run\'s phase-2 output dir (from a run_*_full_analyze.sh script); omit to skip phase-2 comparisons')
    parser.add_argument('--run_phase2_coverage', type=str, default=None,
                         help='e.g. "903/4036 = 22.37%%"; required if --run_phase2_dir given')
    parser.add_argument('--baseline_label', type=str, default=BASELINE_LABEL_DEFAULT)
    parser.add_argument('--baseline_zs_dir', type=str, default=BASELINE_ZS_DIR_DEFAULT)
    parser.add_argument('--baseline_zs_coverage', type=str, default=BASELINE_ZS_COVERAGE_DEFAULT)
    parser.add_argument('--baseline_phase2_dir', type=str, default=BASELINE_PHASE2_DIR_DEFAULT)
    parser.add_argument('--baseline_phase2_coverage', type=str, default=BASELINE_PHASE2_COVERAGE_DEFAULT)
    parser.add_argument('--zs_out_dir', type=str, default=None,
                         help='default: <run_zs_dir>/comparison_vs_baseline')
    parser.add_argument('--skip_missing', action='store_true',
                        help='skip a comparison whose input image does not exist, instead of '
                             'failing (small_analyze.sh does not build every full_analyze panel)')
    parser.add_argument('--phase2_out_dir', type=str, default=None,
                         help='default: <run_phase2_dir>/comparison_vs_baseline')
    args = parser.parse_args()

    if args.run_zs_dir and not args.run_zs_coverage:
        parser.error('--run_zs_coverage is required when --run_zs_dir is given')
    if args.run_phase2_dir and not args.run_phase2_coverage:
        parser.error('--run_phase2_coverage is required when --run_phase2_dir is given')
    if not args.run_zs_dir and not args.run_phase2_dir:
        parser.error('give at least one of --run_zs_dir / --run_phase2_dir')

    main(args)
