"""Builds a minimal, non-colorful 2-slide deck explaining the relational
(distance-proportional) phase-2 training process, same plain style as
make_phase2_deck.py / make_phase2_repel_deck.py (white background, thin gray
borders, one muted accent color used only to mark the thing that trains --
the Encoder): deliberately not colorful.

Slide 2's right-hand table is a 3-way Version 1/2/3 comparison, not a single
run's config -- this deck documents the 2026-08-11 test of which 2026-08-10
collapse fix actually matters (a FIXED mu_3D_reference, FIXED neighbours, or
both together) rather than one settled run.
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
            accent=False, size=11, title_size=12.5, radius=True, align=PP_ALIGN.CENTER):
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
        run.font.color.rgb = ACCENT if (accent and i == 0) else DARK
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
    """A connector with no arrowhead -- for pass-through waypoints on a manually
    routed multi-segment path, where only the final segment should arrow into
    a box."""
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
add_title(s1, 'Relational (Distance-Proportional) Loss — Block Diagram',
          'RKD-style continuous loss: one proportionality term replaces map+repel\'s split attraction/repulsion')

# --- trunk: top lane photo->DINOv2 (frozen), bottom lane mesh->Encoder (trains) ---
add_box(s1, 0.5, 1.15, 1.6, 0.75, ['Photo'], fill=FILL_FROZEN, size=10.5)
add_arrow(s1, 2.1, 1.525, 2.65, 1.525)
add_box(s1, 2.65, 1.15, 2.0, 0.75, ['DINOv2', 'frozen'], fill=FILL_FROZEN, size=10)

add_box(s1, 0.5, 2.15, 1.6, 0.75, ['Mesh', '4036 verts'], fill=FILL_FROZEN, size=9.5)
add_arrow(s1, 2.1, 2.525, 2.65, 2.525)
add_box(s1, 2.65, 2.15, 2.0, 0.75, ['Encoder', 'TRAINS'], accent=True, size=10)

add_arrow(s1, 4.65, 1.525, 5.2, 1.7)
add_arrow(s1, 4.65, 2.525, 5.2, 2.35)

add_box(s1, 5.2, 1.15, 3.35, 1.95,
        ['soft_map_to_vertices()',
         'cosine sim.: pixel feature vs. all 4036 vertex feats.',
         'top-10 nearest (feature space) → softmax →',
         'weighted avg. position'],
        size=9.5, title_size=11.5)

add_arrow(s1, 8.55, 2.125, 9.05, 2.125)
add_box(s1, 9.05, 1.75, 1.55, 0.75, ['ŷ', 'per pixel'], size=10, title_size=15)

# --- centers x neighbours (ONE pair set feeds BOTH distance spaces below --
# unlike map+repel's two separate rings, there's only one neighbourhood here) ---
add_arrow(s1, 9.825, 2.5, 6.65, 3.65, elbow=True)

add_box(s1, 3.9, 3.65, 5.5, 0.85,
        ['Centers × Neighbours',
         '2048 fixed centers, 16 neighbours each (8 in-seg + 8 out-seg)',
         'resampled every step  —or—  fixed once  ⟵ 2026-08-11 A/B'],
        size=9.5, title_size=11.5)

# --- split into two branches: same pairs, measured in two different spaces ---
add_arrow(s1, 6.2, 4.5, 4.6, 5.15)
add_arrow(s1, 7.1, 4.5, 8.7, 5.15)

add_box(s1, 2.0, 5.15, 4.3, 0.85,
        ['d_2D = ‖coord_i − coord_l‖  (pixel space)',
         'norm_d_2D = d_2D / mu_2D — mu_2D always live'],
        size=9.5, title_size=11.5, fill=FILL_FROZEN)
add_box(s1, 7.0, 5.15, 4.3, 0.85,
        ['d_3D = ‖ŷ_i − ŷ_l‖  (mesh space)',
         'norm_d_3D = d_3D / mu_3D — FIXED or LIVE  ⟵ A/B'],
        size=9.5, title_size=11.5, fill=FILL_FROZEN)

add_arrow(s1, 4.15, 6.0, 5.9, 6.45, elbow=True)
add_arrow(s1, 9.1, 6.0, 7.4, 6.45, elbow=True)

add_box(s1, 4.85, 6.45, 3.6, 0.6,
        ['L = Huber_δ(norm_d_3D − norm_d_2D)'],
        size=11.5, title_size=11.5, accent=False, border_pt=1.4)

# feedback: loss -> backward -> Encoder only. Same routing as make_phase2_repel_deck.py.
MARGIN_X = 0.3
BELOW_ENCODER_Y = 3.05
add_line(s1, 4.85, 6.75, MARGIN_X, 6.75, color=ACCENT, width=1.5, dashed=True)
add_line(s1, MARGIN_X, 6.75, MARGIN_X, BELOW_ENCODER_Y, color=ACCENT, width=1.5, dashed=True)
add_line(s1, MARGIN_X, BELOW_ENCODER_Y, 3.65, BELOW_ENCODER_Y, color=ACCENT, width=1.5, dashed=True)
add_arrow(s1, 3.65, BELOW_ENCODER_Y, 3.65, 2.9, color=ACCENT, width=1.5, dashed=True)
add_label(s1, 0.4, 6.85, 3.6, 0.3, 'loss.backward() — updates the Encoder only',
          size=9.5, italic=True, color=ACCENT, align=PP_ALIGN.LEFT, bold=True)

add_label(s1, 4.85, 7.2, 8.0, 0.3,
          'Frozen throughout: DINOv2, pixel coordinates, mesh vertex positions. mu_2D always detached; mu_3D detached too when live.',
          size=9.5, italic=False, color=MUTED, align=PP_ALIGN.LEFT)

# ============================= SLIDE 2 — formulas & A/B comparison =============================
s2 = prs.slides.add_slide(BLANK)
add_title(s2, 'Formulas & the 2026-08-11 Fix Comparison',
          'Two independent fixes for the 2026-08-10 collapse, isolated and combined, run head to head')

# --- left: formulas ---
add_label(s2, 0.5, 1.25, 6.1, 0.35, 'Per-pixel soft point (DPC eq. 1–3, unchanged)',
          size=12.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 0.5, 1.65, 6.1, 1.15,
        ['s_ij = cos(pixel_i feature, vertex_j feature)',
         'w_ij = softmax over j ∈ top-10 nearest by s_ij',
         'ŷ_i = Σ_j w_ij · position_j'],
        size=11, title_size=11, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 3.0, 6.1, 0.35, 'Distance-wise relational loss (RKD, Park et al. CVPR 2019)',
          size=12.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 0.5, 3.4, 6.1, 1.75,
        ['d_2D = ‖coord_i − coord_l‖     (pixel space, ~310px scale)',
         'd_3D = ‖ŷ_i − ŷ_l‖     (mesh space, bbox diag ≈ 1.89)',
         'norm_d_2D = d_2D / mu_2D     mu_2D = live per-step mean',
         'norm_d_3D = d_3D / mu_3D     mu_3D = FIXED or LIVE (right)',
         'L = Huber_δ(norm_d_3D − norm_d_2D),  δ = 1.0'],
        size=10.5, title_size=10.5, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 5.35, 6.1, 0.85,
          'Why a LIVE mu_3D can collapse: normalizing by a mean is invariant to '
          'every pairwise 3D distance shrinking together in proportion — a loss '
          'that only compares ratios to its own batch\'s average cannot see or '
          'resist that move.',
          size=10, italic=True, color=MUTED, align=PP_ALIGN.LEFT)

# --- right: Version 1 vs 2 vs 3 comparison table ---
add_label(s2, 6.9, 1.25, 5.9, 0.35, 'Version 1  vs.  2  vs.  3', size=13, italic=False, bold=True,
          color=DARK, align=PP_ALIGN.LEFT)
add_label(s2, 6.9, 1.58, 5.9, 0.3, 'fixed_norm  /  fixed_neighbors  /  fixed_both',
          size=9, italic=True, bold=False, color=MUTED, align=PP_ALIGN.LEFT)

NAME_X, NAME_W = 6.9, 1.75
V1_X, V1_W = 8.65, 1.4
V2_X, V2_W = 10.05, 1.4
V3_X, V3_W = 11.45, 1.38
row_h = 0.46
top = 2.0

header = ('', 'V1', 'V2', 'V3')
rows = [
    ('mu_3D', 'FIXED', 'LIVE', 'FIXED'),
    ('neighbours', 'resampled', 'fixed', 'fixed'),
    ('mu_3D_live: 25→300', '0.425→0.420', '0.421→0.019', '0.424→0.418'),
    ('loss shape', 'smooth', 'spiked', 'smooth'),
    ('drift @ step 300', '0.843', '0.686', '0.833'),
    ('result', 'stable', 'COLLAPSED', 'stable'),
]

add_box(s2, NAME_X, top, NAME_W, row_h, [header[0]], fill=FILL_FROZEN, size=9.5, title_size=9.5,
        radius=False, border_pt=0.75, align=PP_ALIGN.LEFT)
add_box(s2, V1_X, top, V1_W, row_h, [header[1]], fill=FILL_FROZEN, size=9.5, title_size=9.5,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)
add_box(s2, V2_X, top, V2_W, row_h, [header[2]], fill=FILL_FROZEN, size=9.5, title_size=9.5,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)
add_box(s2, V3_X, top, V3_W, row_h, [header[3]], fill=FILL_FROZEN, size=9.5, title_size=9.5,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)

for i, (name, v1, v2, v3) in enumerate(rows):
    y = top + (i + 1) * row_h
    is_result = (name == 'result')
    add_box(s2, NAME_X, y, NAME_W, row_h, [name], fill=FILL_FROZEN, size=8.5, title_size=8.5,
            radius=False, border_pt=0.75, align=PP_ALIGN.LEFT)
    add_box(s2, V1_X, y, V1_W, row_h, [v1], fill=FILL, size=8.5, title_size=8.5,
            radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)
    add_box(s2, V2_X, y, V2_W, row_h, [v2], fill=FILL, size=8.5, title_size=8.5,
            radius=False, border_pt=(1.4 if is_result else 0.75), align=PP_ALIGN.CENTER)
    add_box(s2, V3_X, y, V3_W, row_h, [v3], fill=FILL, size=8.5, title_size=8.5,
            radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)

result_y = top + (len(rows) + 1) * row_h + 0.25
add_label(s2, 6.9, result_y, 5.9, 0.35, 'Reading (renders confirm all three)',
          size=12.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 6.9, result_y + 0.4, 5.9, 1.15,
        ['Fixed normalization is what prevents the collapse (V1, V3 both',
         'stable). Whether neighbours are resampled or fixed doesn\'t matter',
         'either way (V2 collapses even with fixed neighbours) — matching',
         'the theoretical argument at left. --fixed_neighbors is a free',
         'simplification going forward, not a substitute for the fix.'],
        size=9.5, title_size=9.5, align=PP_ALIGN.LEFT)

prs.save('/workspace/decks/phase2_relational_training_diagram.pptx')
print('saved /workspace/phase2_relational_training_diagram.pptx')
