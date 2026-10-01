# Tal 2026-09-22: bear3 comparison -- SHIC top1 vs our soft-snap arms (anchor+L_dist, no-LORA merged, LORA per-seg, LORA merged).
import os
from PIL import Image, ImageDraw, ImageFont
S = 'demo/shic/'
BP = '/analyze/back_projection/bear3_OURS_soft_backproj_checker16_uv_k1_withbg_side_by_side.png'
L = 'bear3_sdsquare_truediag2d_mu_aniso_perseg_cycv2T0.05_c1.676e-05_retstride5_'
panels = [
    ('SHIC (their weights) -- top1', S + 'bear3/back_projection/bear3_top1_backproj_checker16_uv_k1_withbg_side_by_side.png'),
    ('anchor+L_dist(1/80), sdist 4.3e-4, iso -- soft',
     S + 'bear3_sdsquare_truediag2d_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_anchorsoftT0.01_w3.5e-7_distT1_g2.7951e-06_sdist_w4.3e-4_T0.01_seed0_phase2' + BP),
    ('no LORA, MERGED, dir per-seg 5e-2 -- soft', S + L + 'T0.01_seed0_merge23_mulr5e-2_phase2' + BP),
    ('LORA, dir per-seg 1e-2 -- soft', S + L + 'loraW_all_r4_lr1e-3_T0.01_seed0_phase2/alpha0.5' + BP),
    ('LORA, dir per-seg 5e-2 -- soft', S + L + 'loraW_all_r4_lr1e-3_T0.01_seed0_mulr5e-2_phase2/alpha0.5' + BP),
    ('LORA, MERGED, dir per-seg 5e-2 -- soft', S + L + 'loraW_all_r4_lr1e-3_T0.01_seed0_merge23_mulr5e-2_phase2/alpha0.5' + BP),
]
H, BAND, PAD, COLS = 480, 34, 12, 3
try: font = ImageFont.truetype('DejaVuSans-Bold.ttf', 17)
except OSError: font = ImageFont.load_default()
def fit(im): return im.resize((int(im.width * H / im.height), H))
top = [fit(Image.open(p).convert('RGB')) for p in (S + 'images/bear3.png', S + 'colored_mesh/checker16_uv_azim180.png',
       S + 'colored_mesh/checker16_uv_azim90.png', S + 'colored_mesh/checker16_uv_azim135.png')]
cells = []
for t, p in panels:
    if not os.path.exists(p): print('MISSING (skipped):', t); continue
    im = Image.open(p).convert('RGB'); im = im.crop((im.width // 2, 0, im.width, im.height))  # right half = back-projection
    cells.append((t, fit(im)))
cw = max(im.width for _, im in cells)
W = max(COLS * (cw + PAD), sum(im.width + PAD for im in top))
rows = (len(cells) + COLS - 1) // COLS
c = Image.new('RGB', (W, (rows + 1) * (BAND + H + PAD)), 'white'); d = ImageDraw.Draw(c)
d.text((8, 8), 'photo  |  mesh side (azim 180)  |  mesh front (azim 90)  |  mesh 3/4 (azim 135, SHIC view)', fill='black', font=font)
x = 0
for im in top: c.paste(im, (x, BAND)); x += im.width + PAD
for i, (t, im) in enumerate(cells):
    x0, y0 = (i % COLS) * (cw + PAD), (1 + i // COLS) * (BAND + H + PAD)
    d.text((x0 + 8, y0 + 8), f'{i + 1}. {t}', fill='black', font=font); c.paste(im, (x0, y0 + BAND))
out = S + 'comparison_bear3_anchor_options/bear3_anchor_options_soft.png'; c.save(out); print('saved', out, c.size)
