"""Builds a minimal, non-colorful 2-slide deck explaining the cycle-consistency loss
(modules/cycle_consistency_loss.py hard-snap + modules/cycle_consistency_loss_soft.py
--cyc_soft), same plain style as the other phase-2 explainer decks.

v2 (2026-08-16): unlike the first cycle-consistency deck, this one deliberately avoids
SHIC's own p(a|b) probability notation (Tal found it confusing) and instead uses this
project's OWN best_segmentation_buddies.py vocabulary -- forward match / backward match /
buddy score -- since this loss is already explicitly styled after that module's mutual
pixel<->vertex reciprocity check (see modules/cycle_consistency_loss.py's own docstring,
point 1), and best_segmentation_buddies.py's own variable names
(similarity_p_to_v / similarity_v_to_p / is_bsb) are the natural, already-in-repo
vocabulary for exactly these two directions. The math is identical to the first deck --
only the labeling changes.

This loss is PARKED (--cyc_weight defaults to 0.0, Tal's call, 2026-08-15).
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
PARKED = RGBColor(0xA3, 0x3A, 0x3A)      # muted red, only for "PARKED" / verdict markers

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


# ============================= SLIDE 1 — block diagram (SOFT variant only) =============================
s1 = prs.slides.add_slide(BLANK)
add_title(s1, 'Cycle-Consistency Loss — Block Diagram',
          'A fully differentiable round trip: pixel → vertex → pixel, penalized by how far it lands from the start')

# --- trunk: top lane photo->DINOv2 (frozen), bottom lane mesh->Encoder (trains) ---
add_box(s1, 0.5, 1.3, 1.6, 0.7, ['Photo', 'n=2048 centers'], fill=FILL_FROZEN, size=9.5)
add_arrow(s1, 2.1, 1.65, 2.65, 1.65)
add_box(s1, 2.65, 1.3, 1.9, 0.7, ['DINOv2', 'frozen'], fill=FILL_FROZEN, size=10)

add_box(s1, 0.5, 2.2, 1.6, 0.7, ['Mesh', 'V=4036 verts'], fill=FILL_FROZEN, size=9.5)
add_arrow(s1, 2.1, 2.55, 2.65, 2.55)
add_box(s1, 2.65, 2.2, 1.9, 0.7, ['Encoder', 'TRAINS'], accent=True, size=10)

add_arrow(s1, 4.55, 1.65, 5.1, 1.75)
add_arrow(s1, 4.55, 2.55, 5.1, 2.4)

# --- forward match: pixel -> vertex candidates, all K kept, never snapped ---
add_box(s1, 5.1, 1.3, 3.7, 1.1,
        ['FORWARD MATCH',
         's(u,j) = cos(u, vertex_j),  j = 1..V',
         'fwd(u,j) = softmax_j∈top10( s(u,j) )',
         '→ K=10 candidates j, each weighted by fwd(u,j)'],
        size=9, title_size=10.5)

add_arrow(s1, 6.95, 2.4, 6.95, 3.0)
add_label(s1, 5.1, 2.42, 3.7, 0.25, 'the K=10 candidates j found above, one at a time',
          size=8.3, italic=True, color=MUTED, align=PP_ALIGN.CENTER)

# --- backward match: takes each candidate j from the forward leg, over the full center pool ---
add_box(s1, 2.9, 3.0, 7.5, 1.15,
        ['BACKWARD MATCH  —  for each candidate vertex j found above:',
         'bwd(j,v) = softmax_v( vertex_j · pixel_v ),  v = 1..n=2048',
         '(v ranges over ALL centers, not just the 16 neighbours)'],
        size=9.5, title_size=10, fill=FILL_FROZEN)

add_arrow(s1, 6.95, 4.15, 6.95, 4.5)

add_box(s1, 3.3, 4.5, 7.3, 0.75,
        ['weight(u,v) = Σ_j  fwd(u,j) · bwd(j,v)',
         '(combines both legs — marginalized over all K=10 candidates)'],
        size=9.5, title_size=9.5)

add_arrow(s1, 6.95, 5.25, 6.95, 5.6)

add_box(s1, 4.55, 5.6, 4.2, 0.65,
        ['L = mean_u [ Σ_v  weight(u,v) · geo(u,v) ]'],
        size=10, title_size=10, border_pt=1.4)
add_label(s1, 8.9, 5.62, 3.9, 0.6,
          'geo(u,v): fixed geodesic pixel\ndistance, precomputed (no grad)',
          size=8.3, italic=True, color=MUTED, align=PP_ALIGN.LEFT)

# feedback: loss -> backward -> Encoder only, same routing as the other decks
MARGIN_X = 0.3
BELOW_ENCODER_Y = 3.05
add_line(s1, 4.55, 6.4, MARGIN_X, 6.4, color=ACCENT, width=1.5, dashed=True)
add_line(s1, MARGIN_X, 6.4, MARGIN_X, BELOW_ENCODER_Y, color=ACCENT, width=1.5, dashed=True)
add_line(s1, MARGIN_X, BELOW_ENCODER_Y, 3.6, BELOW_ENCODER_Y, color=ACCENT, width=1.5, dashed=True)
add_arrow(s1, 3.6, BELOW_ENCODER_Y, 3.6, 2.9, color=ACCENT, width=1.5, dashed=True)
add_label(s1, 0.4, 6.53, 4.3, 0.3, 'loss.backward() — updates the Encoder only',
          size=9, italic=True, color=ACCENT, align=PP_ALIGN.LEFT, bold=True)

add_label(s1, 0.5, 7.0, 12.3, 0.4,
          'No hard argmax or snapping anywhere in this pipeline — both the forward and backward '
          'matches stay full soft distributions all the way to the loss.',
          size=9.5, italic=True, color=MUTED, align=PP_ALIGN.LEFT)

# ============================= SLIDE 2 — formulas & results =============================
s2 = prs.slides.add_slide(BLANK)
add_title(s2, 'Formulas & Results',
          'cyc_weight swept at 0.01 and 0.3, on top of --geodesic --mu_3d_scale 1.1')

# --- left: formulas ---
add_label(s2, 0.5, 1.2, 6.1, 0.3, 'Forward match — pixel u → candidate vertices',
          size=11.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 0.5, 1.55, 6.1, 0.75,
        ['s_j = cos(center_pixel_u feature, vertex_j feature)',
         'forward_match[u,j] = softmax over j ∈ top-10 nearest by s_j'],
        size=9.7, title_size=9.7, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 2.5, 6.1, 0.3, 'Backward match — candidate xk → pixel v, over ALL n=2048',
          size=11.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 0.5, 2.85, 6.1, 0.8,
        ['vf_xk = normalize(vertex_features)[xk]',
         'backward_match[xk,v] = softmax_v( vf_xk · center_pixel_features_v )'],
        size=9.7, title_size=9.7, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 3.85, 6.1, 0.3, 'Round-trip weight → round-trip loss',
          size=11.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 0.5, 4.2, 6.1, 1.1,
        ['weight[u,v] = Σ_cand  forward_match[u,cand] · backward_match[cand,v]',
         '   (marginalized over all K=10 candidates)',
         'L = mean_u [ Σ_v  weight[u,v] · d_geo(u,v) ]',
         'd_geo = fixed 2D geodesic table, ÷ geodesic diameter (matches',
         'relational_loss\'s own norm_d_2D scale, so cyc_weight is comparable)'],
        size=9.2, title_size=9.2, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 5.45, 6.1, 1.3,
          'Both tested weights collapsed vertex coverage instead of improving it, and the '
          'collapse got worse the more weight this loss got — the opposite of the intended '
          'effect. Step-1 gradient magnitude was normal in both cases, so whatever drives the '
          'collapse builds up gradually over training, not from the first step.',
          size=9.5, italic=True, color=MUTED, align=PP_ALIGN.LEFT)

# --- right: results table ---
add_label(s2, 6.9, 1.25, 5.9, 0.35, 'cyc_weight sweep', size=13, italic=False, bold=True,
          color=DARK, align=PP_ALIGN.LEFT)
add_label(s2, 6.9, 1.58, 5.9, 0.3, 'refs = unique vertices touched by 2048 centers (of 4036)',
          size=9, italic=True, bold=False, color=MUTED, align=PP_ALIGN.LEFT)

NAME_X, NAME_W = 6.9, 2.9
C0_X, C0_W = 9.8, 1.5
C1_X, C1_W = 11.3, 1.5
row_h = 0.5
top = 2.05

header = ('', 'cyc_weight=0.01', 'cyc_weight=0.3')
rows = [
    ('refs (coverage)', '51/4036 (1.3%)', '19/4036 (0.5%) by step 100'),
    ('drift, step 100', 'no early crash', '~0.11–0.16 (crashed)'),
]

add_box(s2, NAME_X, top, NAME_W, row_h, [header[0]], fill=FILL_FROZEN, size=9.5, title_size=9.5,
        radius=False, border_pt=0.75, align=PP_ALIGN.LEFT)
add_box(s2, C0_X, top, C0_W, row_h, [header[1]], fill=FILL_FROZEN, size=9, title_size=9,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)
add_box(s2, C1_X, top, C1_W, row_h, [header[2]], fill=FILL_FROZEN, size=9, title_size=9,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)

for i, (name, c0, c1) in enumerate(rows):
    y = top + (i + 1) * row_h
    add_box(s2, NAME_X, y, NAME_W, row_h, [name], fill=FILL_FROZEN, size=8.7, title_size=8.7,
            radius=False, border_pt=0.75, align=PP_ALIGN.LEFT)
    add_box(s2, C0_X, y, C0_W, row_h, [c0], fill=FILL, size=8.3, title_size=8.3,
            radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)
    add_box(s2, C1_X, y, C1_W, row_h, [c1], fill=FILL, size=8.3, title_size=8.3,
            radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)

baseline_y = top + (len(rows) + 1) * row_h + 0.3
add_box(s2, 6.9, baseline_y, 5.9, 0.55,
        ['For reference, the no-cyc baseline (cyc_weight=0): 603/4036 (14.9%)'],
        size=9.5, title_size=9.5, fill=FILL_FROZEN, align=PP_ALIGN.LEFT)

prs.save('/workspace/decks/phase2_cycle_consistency_diagram_bsb_terms.pptx')
print('saved /workspace/decks/phase2_cycle_consistency_diagram_bsb_terms.pptx')
