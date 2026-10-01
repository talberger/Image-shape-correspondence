# Tal 2026-09-22: horse leg VERTEX -> mesh similarity, iso vs directional, azim 45/135, red sphere (r=0.015) = query vertex.
from PIL import Image, ImageDraw, ImageFont
O='demo/shic/lvis164135_mu/figures/heatmap_legs_vertex'
try: f=ImageFont.truetype('DejaVuSans-Bold.ttf',16)
except OSError: f=ImageFont.load_default()
H=300
vs=((4471,'front leg, side x+'),(4384,'front leg, side x-'),(4209,'hind leg, side x+'),(4364,'hind leg, side x-'))
def fit(im): return im.resize((int(im.width*H/im.height),H))
rows=[[(f'vertex {v} ({nm}) -- iso: azim 45 | 135',fit(Image.open(f'{O}/iso/v{v}_az45_135.png').convert('RGB'))),
       ('directional per-seg: azim 45 | 135',fit(Image.open(f'{O}/aniso/v{v}_az45_135.png').convert('RGB')))] for v,nm in vs]
B=26; W=max(sum(im.width+14 for _,im in r) for r in rows)
cv=Image.new('RGB',(W,34+len(rows)*(H+B+10)+34),'white'); d=ImageDraw.Draw(cv)
d.text((8,6),'Horse 164135, LORA, mu_lr 5e-2 -- feature similarity of ONE leg vertex (RED sphere) to every mesh vertex',fill='black',font=f)
y=34
for r in rows:
    x=0
    for t,im in r: d.text((x+6,y+4),t,fill='black',font=f); cv.paste(im,(x,y+B)); x+=im.width+14
    y+=H+B+10
d.text((8,y+6),'Colour: viridis (yellow = most similar). Query vertices at ~22% of the mesh height (lower leg); front = +z (head side).',fill='black',font=f)
cv.save(f'{O}/heatmap_legs_vertex.png'); print(cv.size)
