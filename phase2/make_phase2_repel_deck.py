"""Builds a minimal, non-colorful 2-slide deck explaining the map+repel
phase-2 training process (the combined attractive/repulsive mapping loss,
as run in encoder_dino_mapping_repel_wide300/). Same plain style as
make_phase2_deck.py (white background, thin gray borders, one muted accent
color used only to mark the thing that trains -- the Encoder): deliberately
not colorful.
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
REPEL = RGBColor(0x8A, 0x8A, 0x8A)       # repulsive branch stays neutral gray, not a second color
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
    a box. (MSO_CONNECTOR.ELBOW auto-picks its own midpoint for the turn, which
    can land inside an unrelated box on longer routes -- straight, explicit
    waypoints avoid that.)"""
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
add_title(s1, 'Map + Repel Training — Block Diagram',
          'Combined attractive/repulsive mapping loss: fine-tuning the DINO shape encoder')

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

# --- split into two branches: near ring (attractive) / far ring (repulsive) ---
# Manually routed via a waypoint row at y=3.35, safely below soft_map_to_vertices'
# bottom edge (3.1) -- an auto-elbow here can pick a midpoint that cuts through
# that box instead of going around it.
WP_Y = 3.35
add_line(s1, 9.8, 2.5, 9.8, WP_Y)
add_line(s1, 9.8, WP_Y, 4.15, WP_Y)
add_arrow(s1, 4.15, WP_Y, 4.15, 3.65)
add_arrow(s1, 9.8, WP_Y, 9.15, 3.65)

add_box(s1, 2.0, 3.65, 4.3, 0.85,
        ['Near ring — attractive',
         '8 px, stride 1 (~1.2px), same fine segment'],
        size=9.5, title_size=11.5, fill=FILL_FROZEN)
add_box(s1, 7.0, 3.65, 4.3, 0.85,
        ['Far ring — repulsive',
         '8 px, stride 100 (~120px), any segment'],
        size=9.5, title_size=11.5, fill=FILL_FROZEN)

add_arrow(s1, 4.15, 4.5, 4.15, 5.15)
add_arrow(s1, 9.1, 4.5, 9.1, 5.15)

add_box(s1, 2.0, 5.15, 4.3, 0.85,
        ['mapping_loss()',
         'mean ||ŷ_center − ŷ_near||², valid pairs'],
        size=9.5, title_size=11.5)
add_box(s1, 7.0, 5.15, 4.3, 0.85,
        ['repulsion_loss()',
         'mean relu(margin − ||ŷ_center − ŷ_far||)²'],
        size=9.5, title_size=11.5)

add_arrow(s1, 4.15, 6.0, 5.9, 6.45, elbow=True)
add_arrow(s1, 9.1, 6.0, 7.4, 6.45, elbow=True)

add_box(s1, 4.85, 6.45, 3.6, 0.6,
        ['Combined Loss  =  λ_map · L_map  +  λ_repel · L_repel'],
        size=11.5, title_size=11.5, accent=False, border_pt=1.4)

# feedback: combined loss -> backward -> Encoder only. Routed up the far-left
# margin and into the Encoder box's BOTTOM edge, not its left edge -- entering
# from the left would mean a horizontal run straight through the Mesh box,
# which sits directly in that corridor.
MARGIN_X = 0.3
BELOW_ENCODER_Y = 3.05
add_line(s1, 4.85, 6.75, MARGIN_X, 6.75, color=ACCENT, width=1.5, dashed=True)
add_line(s1, MARGIN_X, 6.75, MARGIN_X, BELOW_ENCODER_Y, color=ACCENT, width=1.5, dashed=True)
add_line(s1, MARGIN_X, BELOW_ENCODER_Y, 3.65, BELOW_ENCODER_Y, color=ACCENT, width=1.5, dashed=True)
add_arrow(s1, 3.65, BELOW_ENCODER_Y, 3.65, 2.9, color=ACCENT, width=1.5, dashed=True)
add_label(s1, 0.4, 6.85, 3.6, 0.3, 'loss.backward() — updates the Encoder only',
          size=9.5, italic=True, color=ACCENT, align=PP_ALIGN.LEFT, bold=True)

add_label(s1, 4.85, 7.2, 8.0, 0.3,
          'Frozen throughout: DINOv2, pixel features, vertex positions, mesh, both rings’ coordinates.',
          size=9.5, italic=False, color=MUTED, align=PP_ALIGN.LEFT)

# ============================= SLIDE 2 — formulas & run config =============================
s2 = prs.slides.add_slide(BLANK)
add_title(s2, 'Formulas & Run Configuration',
          'What each term computes, and the numbers behind the wide300 run')

# --- left: formulas ---
add_label(s2, 0.5, 1.25, 6.1, 0.35, 'Per-pixel soft point (DPC eq. 1–3, unchanged)',
          size=12.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 0.5, 1.65, 6.1, 1.15,
        ['s_ij = cos(pixel_i feature, vertex_j feature)',
         'w_ij = softmax over j ∈ top-10 nearest by s_ij',
         'ŷ_i = Σ_j w_ij · position_j'],
        size=11, title_size=11, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 3.0, 6.1, 0.35, 'Attractive term — near ring', size=12.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 0.5, 3.4, 6.1, 0.85,
        ['L_map = mean over valid (i, l) of ||ŷ_i − ŷ_l||²',
         'l ∈ 8 pixels adjacent to i, stride 1, same fine segment'],
        size=10.5, title_size=10.5, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 4.45, 6.1, 0.35, 'Repulsive term — far ring', size=12.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 0.5, 4.85, 6.1, 1.15,
        ['L_repel = mean over valid (i, f) of relu(margin − ||ŷ_i − ŷ_f||)²',
         'f ∈ 8 pixels far from i, stride 100, any segment',
         'true distance (not squared) — needs ε inside sqrt for stable',
         'gradients when a far pair starts near-zero apart'],
        size=10.5, title_size=10.5, align=PP_ALIGN.LEFT)

add_box(s2, 0.5, 6.2, 6.1, 0.65,
        ['L = λ_map · L_map  +  λ_repel · L_repel'],
        size=13, title_size=13, border_pt=1.4)

# --- right: technical data for this run ---
add_label(s2, 7.0, 1.25, 5.8, 0.35, 'Run configuration', size=13, italic=False, bold=True,
          color=DARK, align=PP_ALIGN.LEFT)
add_label(s2, 7.0, 1.58, 5.8, 0.3, 'experiments/guitar/encoder_dino_mapping_repel_wide300/',
          size=9.5, italic=True, bold=False, color=MUTED, align=PP_ALIGN.LEFT)

param_lines = [
    ('center pixels', '8192  (sampled once, fixed)'),
    ('vertices / pixel (k)', '10, nearest in feature space'),
    ('near ring', 'stride 1  ≈ 1.21 px, same segment'),
    ('far ring', 'stride 100  ≈ 120 px, any segment'),
    ('repel margin', '0.05  (mesh bbox diagonal ≈ 1.89)'),
    ('λ_map  /  λ_repel', '0.5  /  0.5'),
    ('learning rate', '1e-4,  adam_eps 1e-12'),
    ('steps', '300,  checkpoint every 5'),
    ('init', 'phase-1 checkpoint (photo-consistency MSE)'),
]
row_h = 0.4
top = 1.95
for i, (k, v) in enumerate(param_lines):
    y = top + i * row_h
    add_box(s2, 7.0, y, 2.7, row_h, [k], fill=FILL_FROZEN, size=10, title_size=10,
            radius=False, border_pt=0.75, align=PP_ALIGN.LEFT)
    add_box(s2, 9.7, y, 3.1, row_h, [v], fill=FILL, size=10, title_size=10,
            radius=False, border_pt=0.75, align=PP_ALIGN.LEFT)

results_y = top + len(param_lines) * row_h + 0.25
add_label(s2, 7.0, results_y, 5.8, 0.35, 'Result (step 0 → step 300)',
          size=12.5, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_box(s2, 7.0, results_y + 0.4, 5.8, 0.95,
        ['loss:  0.000135  →  0.000012',
         'drift (mean cosine vs. phase-1 features):  1.000  →  0.761'],
        size=10.5, title_size=10.5, align=PP_ALIGN.LEFT)

prs.save('/workspace/decks/phase2_repel_training_diagram.pptx')
print('saved /workspace/phase2_repel_training_diagram.pptx')
