"""Builds a minimal, non-colorful 2-slide deck explaining the cycle-consistency
loss (modules/cycle_consistency_loss.py hard-snap + modules/cycle_consistency_loss_soft.py
--cyc_soft), same plain style as make_phase2_relational_deck.py / make_phase2_deck.py /
make_phase2_repel_deck.py (white background, thin gray borders, one muted accent color
used only to mark the thing that trains -- the Encoder).

This loss is PARKED (--cyc_weight defaults to 0.0, Tal's call, 2026-08-15) -- both slides
say so explicitly rather than presenting it as a live/adopted part of the recipe.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

DARK = RGBColor(0x33, 0x33, 0x33)
BORDER = RGBColor(0x8A, 0x8A, 0x8A)
FILL = RGBColor(0xFF, 0xFF, 0xFF)
FILL_FROZEN = RGBColor(0xF6, 0xF6, 0xF6)
ACCENT = RGBColor(0x3B, 0x5B, 0x92)      # the only color used, and only for "trains" / backward
MUTED = RGBColor(0x77, 0x77, 0x77)
PARKED = RGBColor(0xA3, 0x3A, 0x3A)      # muted red, only for "PARKED" markers -- second and
                                          # last color used in the deck, deliberately restrained

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_title(slide, text, subtitle=None):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(12.3), Inches(0.55))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    run.font.size = Pt(23)
    run.font.bold = True
    run.font.color.rgb = DARK
    run.font.name = 'Calibri'
    if subtitle:
        tb2 = slide.shapes.add_textbox(Inches(0.5), Inches(0.66), Inches(12.3), Inches(0.35))
        tf2 = tb2.text_frame
        p2 = tf2.paragraphs[0]
        run2 = p2.add_run()
        run2.text = subtitle
        run2.font.size = Pt(12.5)
        run2.font.color.rgb = MUTED
        run2.font.name = 'Calibri'
        run2.font.italic = True


def add_box(slide, x, y, w, h, lines, fill=FILL, border=BORDER, border_pt=1.0,
            accent=False, size=11, title_size=12.5, radius=True, align=PP_ALIGN.CENTER,
            title_color=None):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = ACCENT if accent else border
    shape.line.width = Pt(2.0 if accent else border_pt)
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.1)
    tf.margin_right = Inches(0.1)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(title_size if i == 0 else size)
        run.font.bold = (i == 0)
        run.font.color.rgb = (title_color if (i == 0 and title_color) else
                              (ACCENT if (accent and i == 0) else DARK))
        run.font.name = 'Calibri'
    return shape


def _set_arrowhead(line_elm, tag, kind='triangle'):
    el = line_elm.makeelement(qn(tag), {'type': kind, 'w': 'med', 'len': 'med'})
    line_elm.append(el)


def add_arrow(slide, x1, y1, x2, y2, color=BORDER, width=1.25, dashed=False, elbow=False):
    ctype = MSO_CONNECTOR.ELBOW if elbow else MSO_CONNECTOR.STRAIGHT
    conn = slide.shapes.add_connector(ctype, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    conn.line.color.rgb = color
    conn.line.width = Pt(width)
    ln = conn.line._get_or_add_ln()
    _set_arrowhead(ln, 'a:tailEnd')
    if dashed:
        ln.append(ln.makeelement(qn('a:prstDash'), {'val': 'dash'}))
    return conn


def add_line(slide, x1, y1, x2, y2, color=BORDER, width=1.25, dashed=False):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    conn.line.color.rgb = color
    conn.line.width = Pt(width)
    if dashed:
        ln = conn.line._get_or_add_ln()
        ln.append(ln.makeelement(qn('a:prstDash'), {'val': 'dash'}))
    return conn


def add_label(slide, x, y, w, h, text, size=10.5, italic=True, color=MUTED, align=PP_ALIGN.CENTER, bold=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.italic = italic
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = 'Calibri'
    return tb


# ============================= SLIDE 1 — block diagram =============================
s1 = prs.slides.add_slide(BLANK)
add_title(s1, 'Cycle-Consistency Loss — Block Diagram',
          'Round-trip pixel → vertex → pixel, penalized by geodesic distance from the start  —  PARKED (cyc_weight=0)')

# --- trunk: top lane photo->DINOv2 (frozen), bottom lane mesh->Encoder (trains) ---
add_box(s1, 0.5, 1.1, 1.6, 0.7, ['Photo', 'n=2048 centers'], fill=FILL_FROZEN, size=9.5)
add_arrow(s1, 2.1, 1.45, 2.65, 1.45)
add_box(s1, 2.65, 1.1, 1.9, 0.7, ['DINOv2', 'frozen'], fill=FILL_FROZEN, size=10)

add_box(s1, 0.5, 2.0, 1.6, 0.7, ['Mesh', '4036 verts'], fill=FILL_FROZEN, size=9.5)
add_arrow(s1, 2.1, 2.35, 2.65, 2.35)
add_box(s1, 2.65, 2.0, 1.9, 0.7, ['Encoder', 'TRAINS'], accent=True, size=10)

add_arrow(s1, 4.55, 1.45, 5.05, 1.55)
add_arrow(s1, 4.55, 2.35, 5.05, 2.2)

# --- forward leg: soft_map_to_vertices, shared by both variants ---
add_box(s1, 5.05, 1.05, 3.6, 1.1,
        ['Forward: soft_map_to_vertices(return_topk=True)',
         'cosine sim. vs all 4036 vertex feats → top-10 → softmax',
         '= p(xk | u)   (u = a center pixel)'],
        size=9.5, title_size=10.5)

add_arrow(s1, 6.2, 2.15, 4.35, 2.85, elbow=True)
add_arrow(s1, 7.3, 2.15, 9.15, 2.85, elbow=True)

# --- two variants of the forward leg's HARD/SOFT choice ---
add_box(s1, 2.5, 2.85, 3.7, 0.85,
        ['HARD-SNAP (2026-08-14 sweep)',
         'xk = argmin ‖soft_point − vertex‖  (no_grad)',
         'p(xk|u) collapses to one-hot'],
        size=8.7, title_size=10)
add_box(s1, 7.15, 2.85, 3.7, 0.85,
        ['SOFT (--cyc_soft, 2026-08-15, SHIC-faithful)',
         'keep all K=10 candidates + weights_fwd',
         'p(xk|u) never collapsed'],
        size=8.7, title_size=10)

add_arrow(s1, 4.35, 3.7, 6.65, 4.15, elbow=True)
add_arrow(s1, 9.0, 3.7, 7.35, 4.15, elbow=True)

# --- backward leg: shared mechanism, over the FULL center pool (not the 16 neighbours) ---
add_box(s1, 2.7, 4.15, 8.0, 1.0,
        ['Backward: vertex feature(s) at forward candidate(s), normalized',
         '→ cosine sim. vs ALL n=2048 center-pixel features (not just 16 neighbours)',
         '→ softmax  =  p(v | xk)     [hard: 1 row of n  |  soft: K rows, marginalized over a]'],
        size=9.5, title_size=10.5, fill=FILL_FROZEN)

add_arrow(s1, 6.7, 5.15, 6.7, 5.5)

add_box(s1, 3.5, 5.5, 6.4, 0.75,
        ['p(v|u) = Σ_a p(xk=a|u) · p(v|xk=a)   (hard: single term, a = xk only)'],
        size=10, title_size=10)

add_box(s1, 10.0, 4.15, 2.85, 1.0,
        ['center_geo_dist [n,n]',
         'precomputed, FIXED, no grad',
         'geodesic dist between every',
         'pair of centers, normalized'],
        size=8.5, title_size=9.5, fill=FILL_FROZEN)

add_arrow(s1, 6.7, 6.25, 6.7, 6.5)
add_arrow(s1, 10.5, 5.15, 7.4, 6.35, elbow=True)

add_box(s1, 4.85, 6.5, 3.6, 0.65,
        ['L = mean_u [ Σ_v p(v|u) · center_geo_dist(u,v) ]'],
        size=11, title_size=11, border_pt=1.4)

# feedback: loss -> backward -> Encoder only, same routing as the relational deck
MARGIN_X = 0.3
BELOW_ENCODER_Y = 2.85
add_line(s1, 4.85, 6.82, MARGIN_X, 6.82, color=ACCENT, width=1.5, dashed=True)
add_line(s1, MARGIN_X, 6.82, MARGIN_X, BELOW_ENCODER_Y, color=ACCENT, width=1.5, dashed=True)
add_line(s1, MARGIN_X, BELOW_ENCODER_Y, 3.6, BELOW_ENCODER_Y, color=ACCENT, width=1.5, dashed=True)
add_arrow(s1, 3.6, BELOW_ENCODER_Y, 3.6, 2.7, color=ACCENT, width=1.5, dashed=True)
add_label(s1, 0.4, 6.95, 4.3, 0.3, 'loss.backward() — updates the Encoder only',
          size=9, italic=True, color=ACCENT, align=PP_ALIGN.LEFT, bold=True)

add_label(s1, 8.6, 6.5, 4.3, 0.65,
          'Total = relational_weight·relational_loss + cyc_weight·this loss.\ncyc_weight = 0 (default) — PARKED, see slide 2.',
          size=9.5, italic=False, color=PARKED, align=PP_ALIGN.LEFT, bold=True)

add_label(s1, 4.85, 7.15, 8.3, 0.3,
          'Frozen: DINOv2, pixel coords. center_geo_dist fixed. Penalizes reciprocity only, not geometric correctness.',
          size=8.5, italic=False, color=MUTED, align=PP_ALIGN.LEFT)

# ============================= SLIDE 2 — formulas & hard vs soft comparison =============================
s2 = prs.slides.add_slide(BLANK)
add_title(s2, 'Formulas & Hard-Snap vs. Soft Comparison',
          'Both tested, both PARKED (2026-08-15) — neither beat the no-cyc baseline')

# --- left: formulas ---
add_label(s2, 0.5, 1.25, 6.1, 0.35, 'Forward leg — p(xk | u)',
          size=12.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 0.5, 1.65, 6.1, 0.95,
        ['s_j = cos(center_pixel_u feature, vertex_j feature)',
         'w_j = softmax over j ∈ top-10 nearest by s_j   (= p(xk=j|u))',
         'HARD variant: xk = argmin_v ‖ Σ_j w_j·pos_j − pos_v ‖  (no_grad, one-hot)'],
        size=10, title_size=10, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 2.85, 6.1, 0.35, 'Backward leg — p(v | xk), over ALL n=2048 centers',
          size=12.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 0.5, 3.25, 6.1, 0.8,
        ['vf_xk = normalize(vertex_features)[xk]',
         'p(v|xk) = softmax_v( vf_xk · center_pixel_features_v )'],
        size=10, title_size=10, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 4.2, 6.1, 0.35, 'Round-trip loss (SHIC L_cyc, Sec. 3.6)',
          size=12.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 0.5, 4.6, 6.1, 1.1,
        ['p(v|u) = Σ_a p(xk=a|u) · p(v|xk=a)   (hard: 1 term; soft: K=10 terms)',
         'L = mean_u [ Σ_v p(v|u) · d_geo(u,v) ]',
         'd_geo = fixed 2D geodesic table, ÷ geodesic diameter (matches',
         'relational_loss\'s own norm_d_2D scale, so cyc_weight is comparable)'],
        size=9.5, title_size=9.5, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 5.85, 6.1, 1.15,
          'Why hard-snap struggles: two centers whose soft points snap to the SAME '
          'vertex get an identical vf_xk row → identical p(v|xk) → the loss cannot '
          'tell them apart — exactly the many-to-one collapse it was meant to fix. '
          'The soft fix removes that degeneracy but collapsed harder anyway (right).',
          size=9.5, italic=True, color=MUTED, align=PP_ALIGN.LEFT)

# --- right: hard vs soft comparison table ---
add_label(s2, 6.9, 1.25, 5.9, 0.35, 'Hard-snap  vs.  Soft (--cyc_soft)', size=13, italic=False, bold=True,
          color=DARK, align=PP_ALIGN.LEFT)
add_label(s2, 6.9, 1.58, 5.9, 0.3, 'refs = unique vertices touched by 2048 centers (of 4036)',
          size=9, italic=True, bold=False, color=MUTED, align=PP_ALIGN.LEFT)

NAME_X, NAME_W = 6.9, 2.5
H_X, H_W = 9.4, 1.72
S_X, S_W = 11.12, 1.71
row_h = 0.46
top = 2.0

header = ('', 'hard-snap', 'soft')
rows = [
    ('forward p(xk|u)', 'one-hot (argmin)', 'full top-10, never snapped'),
    ('refs @ cyc_weight=0.01', '334/4036', '51/4036'),
    ('refs @ cyc_weight=0.3', '19/4036', '19/4036, but by step 100'),
    ('drift @ 0.3, step 100', 'gradual', '~0.11–0.16 (crashed early)'),
    ('step-1 grad norm', '≈ soft (matched)', '≈ hard (matched)'),
    ('result', 'never beat baseline', 'WORSE, not better'),
]

add_box(s2, NAME_X, top, NAME_W, row_h, [header[0]], fill=FILL_FROZEN, size=9.5, title_size=9.5,
        radius=False, border_pt=0.75, align=PP_ALIGN.LEFT)
add_box(s2, H_X, top, H_W, row_h, [header[1]], fill=FILL_FROZEN, size=9.5, title_size=9.5,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)
add_box(s2, S_X, top, S_W, row_h, [header[2]], fill=FILL_FROZEN, size=9.5, title_size=9.5,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)

for i, (name, h, s) in enumerate(rows):
    y = top + (i + 1) * row_h
    is_result = (name == 'result')
    add_box(s2, NAME_X, y, NAME_W, row_h, [name], fill=FILL_FROZEN, size=8.3, title_size=8.3,
            radius=False, border_pt=0.75, align=PP_ALIGN.LEFT)
    add_box(s2, H_X, y, H_W, row_h, [h], fill=FILL, size=8.0, title_size=8.0,
            radius=False, border_pt=(1.4 if is_result else 0.75), align=PP_ALIGN.CENTER,
            title_color=(PARKED if is_result else None))
    add_box(s2, S_X, y, S_W, row_h, [s], fill=FILL, size=8.0, title_size=8.0,
            radius=False, border_pt=(1.4 if is_result else 0.75), align=PP_ALIGN.CENTER,
            title_color=(PARKED if is_result else None))

result_y = top + (len(rows) + 1) * row_h + 0.25
add_label(s2, 6.9, result_y, 5.9, 0.35, 'Verdict — PARKED (2026-08-15, Tal\'s call)',
          size=12.5, italic=False, bold=True, color=PARKED, align=PP_ALIGN.LEFT)
add_box(s2, 6.9, result_y + 0.4, 5.9, 1.85,
        ['Fixing the hard-snap gradient degeneracy was the mathematically',
         'correct move relative to SHIC\'s actual L_cyc — but it wasn\'t what was',
         'standing between this loss and being useful. Two structural gaps:',
         '(1) SHIC\'s L_cyc only ever rides on top of an ABSOLUTE anchor',
         '(L_pseudo/L_dist, from a separate frozen render-and-match pipeline)',
         '— this project has no such anchor anywhere, every loss here is',
         'purely relative. (2) SHIC trains on hundreds of real images per',
         'category; this trains on a SINGLE image. Neither gap is fixable by',
         'tuning cyc_weight or the snap mechanics. cyc_weight stays 0.'],
        size=9, title_size=9, align=PP_ALIGN.LEFT)

prs.save('/workspace/decks/phase2_cycle_consistency_diagram.pptx')
print('saved /workspace/decks/phase2_cycle_consistency_diagram.pptx')
