"""Measure how different a center pixel's DINO feature is from its 7 chosen
neighbours' -- the open question from 2026-08-02 (see FINE_TUNING_PLAN.md,
"Open question: is the pixel neighbourhood below DINO's resolution?").

If neighbour cosine comes out ~0.99 and close to the random-pair baseline, the
mapping loss is asking near-identical feature vectors to map to the same point,
and flattening the softmax (= contraction) is the cheapest way to comply.

Run from /workspace/bsb. Needs ~8GB RAM for the 3.6GB feature file.
"""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.mapping_loss import sample_center_pixels, build_pixel_neighbors

mask = torch.load('./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2.pth')
seg  = torch.load('./demo/guitar/phase2/guitar_01_fine_segments.pth')
mask = np.asarray(mask.cpu() if torch.is_tensor(mask) else mask).astype(bool).squeeze()
seg  = np.asarray(seg.cpu() if torch.is_tensor(seg) else seg).squeeze()

coords, _, seg_ids = sample_center_pixels(mask, seg, 2048, seed=0)
nbr, valid = build_pixel_neighbors(coords, seg_ids, 7)

print('loading 3.6GB pixel features...')
pf = torch.load('./demo/guitar/bsb/guitar_01_pix_dino_features.pth', map_location='cpu')
feats = pf[coords[:, 0], coords[:, 1]].float()
rng = np.random.RandomState(1)
rr, cc = np.nonzero(mask)
ridx = rng.choice(len(rr), 4096, replace=False)
rand_feats = pf[rr[ridx], cc[ridx]].float()
del pf

f  = torch.nn.functional.normalize(feats, dim=1)
rf = torch.nn.functional.normalize(rand_feats, dim=1)
cos = (f.unsqueeze(1) * f[nbr]).sum(-1)
cos_v = cos[valid].numpy()
rand_cos = (rf[:2048] * rf[2048:]).sum(-1).numpy()

print('\n--- cosine: center pixel vs its 7 euclidean neighbours ---')
for r in range(7):
    m = valid[:, r]
    print('  rank %d: cos %.4f' % (r + 1, cos[:, r][m].mean()))
print('  ALL neighbour pairs : mean %.4f  median %.4f  p5 %.4f'
      % (cos_v.mean(), np.median(cos_v), np.percentile(cos_v, 5)))
print('  RANDOM pairs in obj : mean %.4f  median %.4f'
      % (rand_cos.mean(), np.median(rand_cos)))
print('  neighbour pairs with cos > 0.99: %.2f%%' % (100 * (cos_v > 0.99).mean()))
print('  neighbour pairs with cos > 0.95: %.2f%%' % (100 * (cos_v > 0.95).mean()))

print('\n--- effective feature resolution ---')
print('  distinct DINO patches covering the object: ~45 (67.5px footprint)')
print('  distinct feature rows among 2048 centers:')
u = torch.unique(torch.round(f * 1e4).long(), dim=0)
print('    %d unique (rounded to 1e-4)' % len(u))
