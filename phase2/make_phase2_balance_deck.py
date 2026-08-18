"""Builds a minimal, non-colorful 2-slide deck explaining the vertex-balance loss
(modules/vertex_balance_loss.py, --balance_weight) -- same plain style as
make_phase2_relational_deck.py / make_phase2_cycle_consistency_deck.py (white background,
thin gray borders, one muted accent color used only to mark the thing that trains --
the Encoder).

This loss is PARKED (--balance_weight defaults to 0.0, Tal's call, 2026-08-15) -- both
slides say so explicitly rather than presenting it as a live/adopted part of the recipe.
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


# ============================= SLIDE 1 — block diagram =============================
s1 = prs.slides.add_slide(BLANK)
add_title(s1, 'Vertex-Balance Loss — Block Diagram',
          'An additive term that discourages many pixels collapsing onto the same handful of vertices')

# --- trunk: top lane photo->DINOv2 (frozen), bottom lane mesh->Encoder (trains) ---
add_box(s1, 0.5, 1.1, 1.6, 0.7, ['Photo', 'n=2048 centers'], fill=FILL_FROZEN, size=9.5)
add_arrow(s1, 2.1, 1.45, 2.65, 1.45)
add_box(s1, 2.65, 1.1, 1.9, 0.7, ['DINOv2', 'frozen'], fill=FILL_FROZEN, size=10)

add_box(s1, 0.5, 2.0, 1.6, 0.7, ['Mesh', 'V=4036 verts'], fill=FILL_FROZEN, size=9.5)
add_arrow(s1, 2.1, 2.35, 2.65, 2.35)
add_box(s1, 2.65, 2.0, 1.9, 0.7, ['Encoder', 'TRAINS'], accent=True, size=10)

add_arrow(s1, 4.55, 1.45, 5.1, 1.55)
add_arrow(s1, 4.55, 2.35, 5.1, 2.2)

# --- shared forward pass, same call every other loss this session already makes ---
add_box(s1, 5.1, 1.05, 3.5, 1.1,
        ['weights[i,k] = softmax_k∈top10( cos(pixel_i, vertex_k) )',
         '(reused from relational_loss — no extra forward pass)'],
        size=9.3, title_size=9.7)

add_arrow(s1, 6.05, 2.15, 4.6, 2.75, elbow=True)
add_arrow(s1, 7.65, 2.15, 9.1, 2.75, elbow=True)

# --- two branches: differentiable soft mass P vs. no-grad hard-argmax fraction f ---
add_box(s1, 2.5, 2.75, 4.2, 1.1,
        ['P_v  (differentiable)',
         'P_v = (1/n) Σ_i weights[i,k]·1[topk_idx[i,k]=v]',
         '(scatter_add — on the gradient path)'],
        size=9.3, title_size=10.5)
add_box(s1, 7.05, 2.75, 4.2, 1.1,
        ['f_v  (no_grad)',
         'f_v = (1/n) Σ_i 1[ argmax_k weights[i,·] = v ]',
         '(hard top-1 bincount — index only)'],
        size=9.3, title_size=10.5)

add_arrow(s1, 4.6, 3.85, 6.5, 4.35, elbow=True)
add_arrow(s1, 9.15, 3.85, 7.3, 4.35, elbow=True)

add_box(s1, 4.6, 4.35, 4.15, 0.85,
        ['L_balance = V · Σ_v  f_v · P_v',
         '(optimum ≈ 1.0 when f_v = P_v = 1/V for every v)'],
        size=10, title_size=11, border_pt=1.4)

add_box(s1, 9.15, 4.35, 3.7, 0.85,
        ['∂L/∂P_v = V·f_v',
         '— overloaded v (f_v large) gets',
         'pushed down in proportion'],
        size=8.7, title_size=10, fill=FILL_FROZEN, align=PP_ALIGN.LEFT)

add_arrow(s1, 6.68, 5.2, 6.68, 5.55)

add_box(s1, 4.85, 5.55, 3.6, 0.65,
        ['Total = relational_weight·relational_loss', '+ balance_weight·L_balance'],
        size=10, title_size=10)

# feedback: loss -> backward -> Encoder only, same routing as the other two decks
MARGIN_X = 0.3
BELOW_ENCODER_Y = 2.85
add_line(s1, 4.85, 5.87, MARGIN_X, 5.87, color=ACCENT, width=1.5, dashed=True)
add_line(s1, MARGIN_X, 5.87, MARGIN_X, BELOW_ENCODER_Y, color=ACCENT, width=1.5, dashed=True)
add_line(s1, MARGIN_X, BELOW_ENCODER_Y, 3.6, BELOW_ENCODER_Y, color=ACCENT, width=1.5, dashed=True)
add_arrow(s1, 3.6, BELOW_ENCODER_Y, 3.6, 2.7, color=ACCENT, width=1.5, dashed=True)
add_label(s1, 0.4, 6.0, 4.3, 0.3, 'loss.backward() — updates the Encoder only',
          size=9, italic=True, color=ACCENT, align=PP_ALIGN.LEFT, bold=True)

add_label(s1, 8.7, 5.55, 4.3, 0.75,
          'Purely additive — changes nothing about the shared\nweights[i,k], so it can be weighted to 0 and walked\naway from cleanly at any time.',
          size=9, italic=True, color=MUTED, align=PP_ALIGN.LEFT)

# ============================= SLIDE 2 — formulas & sweep comparison =============================
s2 = prs.slides.add_slide(BLANK)
add_title(s2, 'Formulas & the balance_weight Sweep',
          'balance_weight swept at 1e-4 / 3e-4 / 1e-3, on top of --geodesic --mu_3d_scale 1')

# --- left: formulas ---
add_label(s2, 0.5, 1.25, 6.1, 0.35, 'Per-vertex soft mass and hard usage',
          size=12.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 0.5, 1.65, 6.1, 1.15,
        ['P_v = (1/n) Σ_i Σ_k∈K  weights[i,k] · 1[topk_idx[i,k] = v]',
         '      (scatter_add — differentiable, gradient → weights → encoder)',
         'f_v = (1/n) Σ_i  1[ argmax_k weights[i,k] → v ]',
         '      (hard top-1 bincount — no_grad, index only)'],
        size=10, title_size=10, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 3.0, 6.1, 0.35, 'Load-balancing loss',
          size=12.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 0.5, 3.4, 6.1, 0.85,
        ['L_balance = V · Σ_v  f_v · P_v',
         'optimum L_balance ≈ 1.0 when f_v = P_v = 1/V for every v (perfect spread)'],
        size=10, title_size=10, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 4.55, 6.1, 1.0,
          'Why it still trains despite f being detached: for whichever vertex is '
          'over-used this batch (f_v large), ∂L/∂P_v = V·f_v — a real, proportional '
          'push DOWN on that vertex\'s soft probability. No need to differentiate '
          'through the hard argmax at all — same "index only, never on the gradient '
          'path" convention as topk_idx everywhere else in this codebase.',
          size=9.5, italic=True, color=MUTED, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 5.75, 6.1, 1.1,
          'Known blind spot: only rewards EVEN usage across vertices — never checks '
          'whether the vertices being used are anywhere near geometrically correct. '
          'relational_loss is still the only thing checking that.',
          size=9.5, italic=True, color=MUTED, align=PP_ALIGN.LEFT)

# --- right: balance_weight sweep table ---
add_label(s2, 6.9, 1.25, 5.9, 0.35, 'balance_weight sweep  (--geodesic --mu_3d_scale 1)', size=13,
          italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_label(s2, 6.9, 1.58, 5.9, 0.3, 'coverage = unique vertices touched, step300, out of 4036',
          size=9, italic=True, bold=False, color=MUTED, align=PP_ALIGN.LEFT)

NAME_X, NAME_W = 6.9, 2.0
C0_X, C0_W = 8.9, 1.15
C1_X, C1_W = 10.05, 1.15
C2_X, C2_W = 11.2, 1.15
row_h = 0.46
top = 2.0

header = ('', '0 (base)', '1e-4', '3e-4 / 1e-3')
rows_top = [
    ('coverage @300', '603/4036', '629/4036', '812 / 869 / 4036'),
    ('drift @300', '~0.89', '0.975 – 0.989 (all three, range)', ''),
]

add_box(s2, NAME_X, top, NAME_W, row_h, [header[0]], fill=FILL_FROZEN, size=9, title_size=9,
        radius=False, border_pt=0.75, align=PP_ALIGN.LEFT)
add_box(s2, C0_X, top, C0_W, row_h, [header[1]], fill=FILL_FROZEN, size=8.5, title_size=8.5,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)
add_box(s2, C1_X, top, C1_W, row_h, [header[2]], fill=FILL_FROZEN, size=8.5, title_size=8.5,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)
add_box(s2, C2_X, top, C2_W, row_h, [header[3]], fill=FILL_FROZEN, size=7.7, title_size=7.7,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)

y1 = top + row_h
add_box(s2, NAME_X, y1, NAME_W, row_h, ['coverage @300'], fill=FILL_FROZEN, size=8.3, title_size=8.3,
        radius=False, border_pt=0.75, align=PP_ALIGN.LEFT)
add_box(s2, C0_X, y1, C0_W, row_h, ['603/4036'], fill=FILL, size=8, title_size=8,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)
add_box(s2, C1_X, y1, C1_W, row_h, ['629/4036'], fill=FILL, size=8, title_size=8,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)
add_box(s2, C2_X, y1, C2_W, row_h, ['812 / 869'], fill=FILL, size=7.7, title_size=7.7,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)

y2 = top + 2 * row_h
add_box(s2, NAME_X, y2, NAME_W, row_h, ['drift @300'], fill=FILL_FROZEN, size=8.3, title_size=8.3,
        radius=False, border_pt=0.75, align=PP_ALIGN.LEFT)
add_box(s2, C0_X, y2, C0_W, row_h, ['~0.89'], fill=FILL, size=8, title_size=8,
        radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)
add_box(s2, C1_X, y2, C1_W + C2_W + 0.15, row_h, ['0.975 – 0.989 (healthy, all three configs)'], fill=FILL,
        size=8, title_size=8, radius=False, border_pt=0.75, align=PP_ALIGN.CENTER)

y3 = top + 3 * row_h
add_box(s2, NAME_X, y3, NAME_W, row_h, ['numbers alone'], fill=FILL_FROZEN, size=8.3, title_size=8.3,
        radius=False, border_pt=0.75, align=PP_ALIGN.LEFT)
add_box(s2, C0_X, y3, C0_W + C1_W + C2_W + 0.3, row_h, ['best result so far — coverage up, drift stays healthy'],
        fill=FILL, size=8.3, title_size=8.3, radius=(False), border_pt=1.4, align=PP_ALIGN.CENTER,
        title_color=None)

y4 = top + 4 * row_h
add_box(s2, NAME_X, y4, NAME_W, row_h, ['visual review'], fill=FILL_FROZEN, size=8.3, title_size=8.3,
        radius=False, border_pt=0.75, align=PP_ALIGN.LEFT)
add_box(s2, C0_X, y4, C0_W + C1_W + C2_W + 0.3, row_h, ['correspondence quality worse despite better spread'],
        fill=FILL, size=8.3, title_size=8.3, radius=False, border_pt=1.4, align=PP_ALIGN.CENTER)

prs.save('/workspace/decks/phase2_balance_diagram.pptx')
print('saved /workspace/decks/phase2_balance_diagram.pptx')
