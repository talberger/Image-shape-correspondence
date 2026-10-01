# Tal 2026-09-22: LORA directional per-seg, mu_lr progression, bear3 + lvis3044 -- soft back-projection + ellipses.
from PIL import Image, ImageDraw, ImageFont
H, BAND, PAD = 400, 30, 10
try: font = ImageFont.truetype('DejaVuSans-Bold.ttf', 15)
except OSError: font = ImageFont.load_default()
B3 = 'demo/shic/bear3_sdsquare_truediag2d_mu_aniso_perseg_cycv2T0.05_c1.676e-05_retstride5_loraW_all_r4_lr1e-3_T0.01_seed0{}_phase2/alpha0.5/analyze/back_projection/bear3_OURS_soft_backproj_checker16_uv_k1_withbg_side_by_side.png'
L3 = 'demo/shic/lvis3044_mu/mu_aniso_perseg_cycv2T0.05_c1.676e-05_retstride3_loraW_all_r4_lr1e-3_T0.01_seed0_mulr{}/alpha0.5/backproj_lvis3044/lvis3044_OURS_soft_backproj_side_by_side.png'
def right(p, aspect):
    im = Image.open(p).convert('RGB'); w = round(im.height * aspect); return im.crop((im.width - w, 0, im.width, im.height))
def fit(im): return im.resize((int(im.width * H / im.height), H))
b_as = 808 / 750; l_as = 270 / 400
rows = [
  [(f'bear3 mu_lr {l} -- soft', right(B3.format('' if l == '1e-2' else '_mulr' + l), b_as)) for l in ('1e-2', '5e-2', '1e-1', '5e-1')],
  [(f'bear3 ellipses {l}', Image.open(f'demo/shic/comparison_bear3_anchor_options/ellipses_bear3_lora_aniso_mulr{l}.png').convert('RGB')) for l in ('1e-2', '5e-2', '1e-1', '5e-1')],
  [(f'3044 mu_lr {l} -- soft', right(L3.format(l), l_as)) for l in ('5e-2', '1e-1', '5e-1')]
  + [(f'3044 ellipses {l}', Image.open(f'demo/shic/lvis3044_mu/figures/ellipses_aniso{"" if l == "5e-2" else "_mulr" + l}.png').convert('RGB')) for l in ('5e-2', '1e-1', '5e-1')],
]
rows = [[(t, fit(im)) for t, im in r] for r in rows]
W = max(sum(im.width + PAD for _, im in r) for r in rows)
c = Image.new('RGB', (W, 30 + len(rows) * (BAND + H + PAD)), 'white'); d = ImageDraw.Draw(c)
d.text((8, 6), 'LORA-W directional per-seg, mu_lr progression. Big-segment ratio: bear3 seg2 1.38/1.42/1.41/1.46; 3044 seg1 1.41/1.42/1.44 -- SATURATED', fill='black', font=font)
y = 30
for r in rows:
    x = 0
    for t, im in r: d.text((x + 6, y + 6), t, fill='black', font=font); c.paste(im, (x, y + BAND)); x += im.width + PAD
    y += BAND + H + PAD
out = 'demo/shic/comparison_bear3_anchor_options/lora_dir_perseg_mulr_progression.png'; c.save(out); print('saved', out, c.size)
