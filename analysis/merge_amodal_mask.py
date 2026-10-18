"""Step 3/3 of the amodal mask (bsb env): pix2gestalt's samples -> ONE mask in the run crop's pixel grid.
  vote   : a pixel is object if it is in at least --min_votes of the samples (the model is a diffusion model: each sample
           is another guess of the hidden part); default 1 = the union of the samples (2026-10-08);
  domain = vote | visible mask (the visible mask is never removed);
  fallback: if the visible pixels are still not in one piece (>= 99.5%), the geometric gap fill joins the rest, so the
           result is ALWAYS one connected domain.
Out: <run_dir>/<tag>_amodal_mask.pth (bool [H, W], same grid as <tag>_mask.pth), _amodal_stats.json, _amodal_mask.png.
Usage: python analysis/merge_amodal_mask.py --run_dir <run dir> --tag lvis7107"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from PIL import Image
from scipy import ndimage
from gapfill_preview import gap_fill
S8 = np.ones((3, 3))

def connected_share(vis, full):
    lab, _ = ndimage.label(full, structure=S8); c = np.bincount(lab[vis]); c[0] = 0
    return float(c.max() / vis.sum())

ap = argparse.ArgumentParser(); ap.add_argument('--run_dir', required=True); ap.add_argument('--tag', required=True)
ap.add_argument('--min_votes', type=int, default=1,
                help='samples that must agree on a pixel. 1 since 2026-10-08 (Tal, after demo/shic/amodal_votes_preview: closes the thin fence / branch '
                     'holes the majority vote left open); 2 = the majority of 3, every amodal run before')
a = ap.parse_args(); D, T = a.run_dir, a.tag
vis = torch.load(f'{D}/{T}_mask.pth').numpy().astype(bool)
g = np.load(f'{D}/{T}_amodal_input.npz'); smp = np.load(f'{D}/{T}_amodal_samples.npz')['amodal']
sq, box, size = g['square'], g['box'], g['size']
assert vis.shape == (size[1], size[0]), (vis.shape, size)
votes = np.zeros(vis.shape, np.int32); side = int(sq[2] - sq[0])
for am in smp:                                                           # 256 -> square crop px -> bbox -> run crop
    full = np.asarray(Image.fromarray((am * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR)) > 127
    sub = full[box[1] - sq[1]:box[3] - sq[1], box[0] - sq[0]:box[2] - sq[0]]
    votes += np.asarray(Image.fromarray((sub * 255).astype(np.uint8)).resize((int(size[0]), int(size[1])), Image.BILINEAR)) > 127
need = a.min_votes
model = votes >= need; domain = model | vis
before = connected_share(vis, domain); radius = 0
if before < 0.995:
    domain, radius, _ = gap_fill(domain)
lab, _ = ndimage.label(domain, structure=S8); main = lab == np.bincount(lab[vis]).argmax()
stats = dict(tag=T, samples=int(len(smp)), min_votes=int(need), visible_px=int(vis.sum()), domain_px=int(main.sum()),
             added_frac=float((main & ~vis).sum() / vis.sum()), model_covers_visible=float((model & vis).sum() / vis.sum()),
             connected_today=connected_share(vis, vis), connected_model_only=before, gapfill_radius=int(radius),
             connected_final=float((vis & main).sum() / vis.sum()))
torch.save(torch.from_numpy(main), f'{D}/{T}_amodal_mask.pth'); json.dump(stats, open(f'{D}/{T}_amodal_stats.json', 'w'), indent=1)
img = np.full(vis.shape + (3,), 255, np.uint8); img[main] = (60, 90, 230); img[vis] = (235, 150, 150); img[vis & main] = (160, 215, 160)
Image.fromarray(img).save(f'{D}/{T}_amodal_mask.png')
print(json.dumps(stats))
