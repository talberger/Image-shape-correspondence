"""Builds a minimal, non-colorful 2-slide deck explaining the phase-2
(mapping-loss) training process. Deliberately plain: white background, thin
gray borders, one muted accent color used only to mark the thing that trains
(the Encoder), nothing else colored.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

DARK = RGBColor(0x33, 0x33, 0x33)
BORDER = RGBColor(0x8A, 0x8A, 0x8A)
FILL = RGBColor(0xFF, 0xFF, 0xFF)
FILL_FROZEN = RGBColor(0xF6, 0xF6, 0xF6)
ACCENT = RGBColor(0x3B, 0x5B, 0x92)   # the only color used, and only for "trains"
MUTED = RGBColor(0x77, 0x77, 0x77)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_title(slide, text, subtitle=None):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.6))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = DARK
    run.font.name = 'Calibri'
    if subtitle:
        tb2 = slide.shapes.add_textbox(Inches(0.5), Inches(0.72), Inches(12.3), Inches(0.4))
        tf2 = tb2.text_frame
        p2 = tf2.paragraphs[0]
        run2 = p2.add_run()
        run2.text = subtitle
        run2.font.size = Pt(13)
        run2.font.color.rgb = MUTED
        run2.font.name = 'Calibri'
        run2.font.italic = True


def add_box(slide, x, y, w, h, lines, fill=FILL, border=BORDER, border_pt=1.0,
            accent=False, size=12, title_size=13, radius=True):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = ACCENT if accent else border
    shape.line.width = Pt(2.25 if accent else 1.0)
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.1)
    tf.margin_right = Inches(0.1)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = line
        run.font.size = Pt(title_size if i == 0 else size)
        run.font.bold = (i == 0)
        run.font.color.rgb = ACCENT if (accent and i == 0) else DARK
        run.font.name = 'Calibri'
    return shape


def _set_arrowhead(line_elm, tag, kind='triangle'):
    ln = line_elm
    el = ln.makeelement(qn(tag), {'type': kind, 'w': 'med', 'len': 'med'})
    ln.append(el)


def add_arrow(slide, x1, y1, x2, y2, color=BORDER, width=1.25, dashed=False, elbow=False):
    ctype = MSO_CONNECTOR.ELBOW if elbow else MSO_CONNECTOR.STRAIGHT
    conn = slide.shapes.add_connector(ctype, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    conn.line.color.rgb = color
    conn.line.width = Pt(width)
    ln = conn.line._get_or_add_ln()
    _set_arrowhead(ln, 'a:tailEnd')
    if dashed:
        prstDash = ln.makeelement(qn('a:prstDash'), {'val': 'dash'})
        ln.append(prstDash)
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


# ============================= SLIDE 1 =============================
s1 = prs.slides.add_slide(BLANK)
add_title(s1, 'Phase 2 Training — Block Diagram',
          'DPC-style mapping loss: fine-tuning the DINO shape encoder')

# --- top lane: photo -> DINOv2 (frozen) ---
add_box(s1, 0.5, 1.35, 1.9, 0.85, ['Photo'], fill=FILL_FROZEN, size=11)
add_arrow(s1, 2.4, 1.775, 3.15, 1.775)
add_box(s1, 3.15, 1.35, 2.35, 0.85, ['DINOv2', 'frozen · [H,W,1024]'], fill=FILL_FROZEN, size=10.5)

# --- bottom lane: mesh -> Encoder (trains) ---
add_box(s1, 0.5, 3.15, 1.9, 0.85, ['Mesh', '4036 vertices'], fill=FILL_FROZEN, size=10.5)
add_arrow(s1, 2.4, 3.575, 3.15, 3.575)
add_box(s1, 3.15, 3.15, 2.35, 0.85, ['Encoder', 'TRAINS · init = phase-1'], accent=True, size=10.5)

# both lanes converge into the central box
add_arrow(s1, 5.5, 1.775, 6.35, 2.55)
add_arrow(s1, 5.5, 3.575, 6.35, 2.95)

# --- central box: soft_map_to_vertices ---
add_box(s1, 6.35, 1.95, 3.05, 2.05,
        ['soft_map_to_vertices()',
         'cosine similarity (pixel feat., all 4036 vertex feats.)',
         '→ top-10 nearest vertices, in FEATURE space',
         '→ softmax over their similarities',
         '→ weighted average of vertex POSITIONS'],
        size=10.5, title_size=12.5)

add_arrow(s1, 9.4, 2.975, 10.05, 2.975)
add_box(s1, 10.05, 2.55, 1.75, 0.85, ['y-hat', 'soft point on surface,', 'per pixel'], size=9.5, title_size=11.5)

# split to center / neighbor
add_arrow(s1, 11.8, 2.7, 12.55, 1.5, elbow=True)
add_arrow(s1, 11.8, 3.25, 12.55, 4.05, elbow=True)

add_box(s1, 10.35, 1.05, 2.45, 0.75, ['y-hat_center  [2048,3]'], size=10)
add_box(s1, 10.35, 3.95, 2.45, 0.9, ['y-hat_neighbor  [2048,8,3]', '8 pixels adjacent in the image'], size=9.5)

# both feed the loss box
add_arrow(s1, 11.55, 1.8, 11.55, 5.0, elbow=True)
add_arrow(s1, 11.55, 4.85, 11.55, 5.0)

add_box(s1, 9.6, 5.0, 3.2, 0.95,
        ['mapping_loss()', 'mean || y-hat_center − y-hat_neighbor ||^2, over valid pairs'],
        size=10, title_size=12)

# feedback: loss -> backward -> Encoder only
add_arrow(s1, 9.6, 5.475, 4.3, 5.475, dashed=True, color=ACCENT, width=1.5)
add_arrow(s1, 4.3, 5.475, 4.3, 4.0, dashed=True, color=ACCENT, width=1.5, elbow=True)
add_label(s1, 4.55, 5.02, 5.6, 0.35, 'loss.backward()  —  gradients update the Encoder weights only',
          size=10.5, italic=True, color=ACCENT, align=PP_ALIGN.LEFT, bold=True)

# frozen/trains legend
add_label(s1, 0.5, 6.75, 12.3, 0.5,
          'Frozen throughout: DINOv2, pixel features, vertex positions, mesh.   '
          'The only thing phase-2 changes is the Encoder\'s weights.',
          size=11, italic=False, color=MUTED, align=PP_ALIGN.LEFT)

# ============================= SLIDE 2 =============================
s2 = prs.slides.add_slide(BLANK)
add_title(s2, 'The Mapping Loss, in Detail',
          'What gets compared to what, and what that means in practice')

# left: the pixel-neighborhood picture
add_label(s2, 0.5, 1.3, 5.6, 0.35, 'Pixel neighborhood (image space, fixed once)', size=13, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)

cx, cy = 2.6, 3.15
cell = 0.55
offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
for (dr, dc) in offsets:
    add_box(s2, cx + dc * cell - cell / 2, cy + dr * cell - cell / 2, cell, cell,
            ['n'], fill=FILL_FROZEN, size=9, radius=False, border_pt=0.75)
add_box(s2, cx - cell / 2, cy - cell / 2, cell, cell, ['i'], accent=True, size=9, radius=False)

add_label(s2, 0.5, 4.15, 5.6, 1.0,
          'Center pixel i is compared separately against each of its 8\n'
          'surrounding pixels l (mean spacing 1.21 px, same fine segment\n'
          'only) — 8 independent terms, not neighbor-vs-neighbor.',
          size=11, italic=False, color=DARK, align=PP_ALIGN.LEFT)

add_label(s2, 0.5, 5.35, 5.6, 0.55,
          '2048 center pixels x 8 neighbors = 16384 pairs per step\n'
          '(a handful invalid: off-image or across a segment boundary)',
          size=10, italic=True, color=MUTED, align=PP_ALIGN.LEFT)

# right: formula box + facts
add_box(s2, 6.5, 1.3, 6.3, 1.55,
        ['L  =  mean over valid (i, l) pairs of  || y-hat_i  −  y-hat_l ||^2',
         'y-hat_i, y-hat_l are each a softmax-weighted average of the',
         '10 nearest vertices to that pixel in FEATURE space (recomputed every step)'],
        size=11.5, title_size=13.5)

add_label(s2, 6.5, 3.1, 6.3, 0.35, 'Two neighborhoods, two different spaces', size=13, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)

add_box(s2, 6.5, 3.5, 3.05, 1.15,
        ['10 vertex neighbors', 'FEATURE space (cosine sim.)', 'recomputed every step —', 'vertex features change'],
        fill=FILL_FROZEN, size=10, title_size=11.5)
add_box(s2, 9.75, 3.5, 3.05, 1.15,
        ['8 pixel neighbors', 'IMAGE space (row, col)', 'computed once —', 'pixel coordinates never change'],
        fill=FILL_FROZEN, size=10, title_size=11.5)

add_label(s2, 6.5, 4.85, 6.3, 0.35, 'What this means in practice', size=13, italic=False, bold=True, color=DARK, align=PP_ALIGN.LEFT)
add_label(s2, 6.5, 5.2, 6.3, 1.9,
          '•  Purely attractive: the loss only ever pulls y-hat_i and y-hat_l\n'
          '   together, never pushes anything apart.\n'
          '•  Adjacent pixels share a DINO patch by construction, so at\n'
          '   1px spacing ~31% of pairs already share an identical top-10\n'
          '   vertex set before training — the residual signal is mostly\n'
          '   top-k boundary flips, not smoothness as such.\n'
          '•  Nothing here keeps vertex features matching the photo —\n'
          '   that was phase-1\'s job.',
          size=11, italic=False, color=DARK, align=PP_ALIGN.LEFT)

prs.save('/workspace/decks/phase2_training_diagram.pptx')
print('saved /workspace/phase2_training_diagram.pptx')
