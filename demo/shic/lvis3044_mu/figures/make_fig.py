# Tal 2026-09-22: directional mu on LVIS 3044 (all LORA-W, mu_lr 5e-2) -- SHIC argmax + our 4 arms soft-snap + ellipses.
import json, sys
from PIL import Image, ImageDraw, ImageFont
# argv (2026-09-22): [ann_id] [class]; default 3044 bear. Horse/elephant use their own SHIC back-projection + renders.
ANN = sys.argv[1] if len(sys.argv) > 1 else '3044'; CLS = sys.argv[2] if len(sys.argv) > 2 else 'bear'
TAG = f'lvis{ANN}'
B = f'demo/shic/{TAG}_mu/'; T = 'cycv2T0.05_c1.676e-05_retstride3_loraW_all_r4_lr1e-3_T0.01_seed0'
PHOTO = f'data/densepose_lvis/runs/{TAG}/{TAG}.png'
arms = [('iso per-seg (baseline)', f'mu_learn_perseg_{T}_mulr5e-2', None), ('directional per-seg', f'mu_aniso_perseg_{T}_mulr5e-2', 'aniso'),
        ('directional grid 6x6', f'mu_grid_perseg_{T}_g6_mulr5e-2', 'grid6'), ('directional global', f'mu_globalaniso_perseg_{T}_mulr5e-2', 'global')]
H, BAND, PAD = 520, 52, 12
try: font = ImageFont.truetype('DejaVuSans-Bold.ttf', 16)
except OSError: font = ImageFont.load_default()
ph = Image.open(PHOTO).convert('RGB'); pw = ph.width / ph.height
def right(p):  # side_by_side = [mesh render | photo-shaped back-projection] -> keep the back-projection
    im = Image.open(p).convert('RGB'); w = round(im.height * pw); return im.crop((im.width - w, 0, im.width, im.height))
def fit(im): return im.resize((int(im.width * H / im.height), H))
row1 = [('photo', ph), ('SHIC -- argmax', right(f'demo/shic/lvis_vs_shic/backproj_lvis3044/shic/lvis3044_SHIC_argmax_backproj_side_by_side.png' if CLS == 'bear' else
         f'{B}{arms[0][1]}/alpha0.5/backproj_{TAG}/shic/{TAG}_SHIC_argmax_backproj_side_by_side.png'))]
row2 = []
for t, r, e in arms:
    j = json.load(open(f'{B}{r}/alpha0.5/{TAG}_vs_shic.json'))
    row1.append((f'{t} -- soft\nGT err soft {j["mean_ours_soft"]:.1f} (SHIC {j["mean_shic"]:.1f})', right(f'{B}{r}/alpha0.5/backproj_{TAG}/{TAG}_OURS_soft_backproj_side_by_side.png')))
    if e: row2.append((f'ellipses: {t}', Image.open(f'{B}figures/ellipses_{e}.png').convert('RGB')))
# Mesh views on the SAME side as the photo (Tal, 2026-09-22): bear photo faces right -> azim 135/180; the horse/elephant
# instances face left -> the mirrored azim 45/0. Plus the fine SAM segmentation the per-seg mu is fit on.
AZ = (135, 180) if CLS == 'bear' else (45, 0)
CM = 'demo/shic/colored_mesh/' + ('' if CLS == 'bear' else CLS + '_')
row2.insert(0, (f'mesh 3/4 (azim {AZ[0]})', Image.open(f'{CM}checker16_uv_azim{AZ[0]}.png').convert('RGB')))
row2.insert(1, (f'mesh side (azim {AZ[1]})', Image.open(f'{CM}checker16_uv_azim{AZ[1]}.png').convert('RGB')))
fs = Image.open(f'data/densepose_lvis/runs/{TAG}/{TAG}_fine_segments.png').convert('RGB')
row2.insert(2, ('fine segments (SAM)', fs))
rows = [[(t, fit(im)) for t, im in r] for r in (row1, row2)]
W = max(sum(im.width + PAD for _, im in r) for r in rows)
c = Image.new('RGB', (W, len(rows) * (BAND + H + PAD) + 30), 'white'); d = ImageDraw.Draw(c)
d.text((8, 6), f'LVIS {ANN} ({CLS}) -- directional mu, all LORA-W (all r4, alpha 0.5), mu_lr 5e-2, cyc stride 3. GT err = mean geodesic over the 3 GT points (noise!)', fill='black', font=font)
y = 30
for r in rows:
    x = 0
    for t, im in r: d.multiline_text((x + 6, y + 4), t, fill='black', font=font); c.paste(im, (x, y + BAND)); x += im.width + PAD
    y += BAND + H + PAD
# Caption UNDER the figure (Tal, 2026-09-22): what each line is, and every drawn segment's fill colour + size.
import numpy as np, torch
seg = torch.load(f'data/densepose_lvis/runs/{TAG}/{TAG}_fine_segments.pth').numpy()
msk = torch.load(f'data/densepose_lvis/runs/{TAG}/{TAG}_mask.pth').numpy().astype(bool)
pal = (np.random.RandomState(3).rand(int(seg.max()) + 1, 3) * 200 + 40).astype(np.uint8)   # = plot_mu_ellipses.py's palette
fr = {g: (seg[msk] == g).mean() for g in range(1, int(seg.max()) + 1)}
shown = [g for g in fr if fr[g] >= 0.01]
cap = Image.new('RGB', (c.width, 70), 'white'); dc = ImageDraw.Draw(cap)
dc.text((8, 4), 'ELLIPSES: red = learned directional mu (radius = mu in that direction) | white = mu 1 reference | '
        'cyan = mu of the ISOTROPIC run on the same segment. Per-seg: segments < 1% of the animal are not drawn '
        f'({len(fr) - len(shown)} slivers hidden). Grid: one ellipse per node on the animal. Global: one ellipse for all.',
        fill='black', font=font)
x = 8
for g in shown:
    dc.rectangle((x, 36, x + 22, 58), fill=tuple(int(v) for v in pal[g]), outline='black')
    t = f'seg {g}: {100 * fr[g]:.1f}%'; dc.text((x + 28, 38), t, fill='black', font=font); x += 40 + int(dc.textlength(t, font=font)) + 20
full = Image.new('RGB', (c.width, c.height + cap.height), 'white'); full.paste(c, (0, 0)); full.paste(cap, (0, c.height)); c = full
out = B + f'figures/{TAG}_directional_mu_lora.png'; c.save(out); print('saved', out, c.size)
