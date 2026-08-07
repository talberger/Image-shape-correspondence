import torch, numpy as np, sys
sys.path.insert(0, '/workspace/bsb')
from modules.mapping_loss import sample_center_pixels, build_pixel_neighbors

mask = torch.load('./demo/guitar/bsb/guitar_01_click_130_140_sam_mask_2.pth')
seg  = torch.load('./demo/guitar/phase2/guitar_01_fine_segments.pth')
mask = np.asarray(mask.cpu() if torch.is_tensor(mask) else mask).astype(bool).squeeze()
seg  = np.asarray(seg.cpu() if torch.is_tensor(seg) else seg).squeeze()
print('mask', mask.shape, mask.sum(), '| seg', seg.shape, 'labels', len(np.unique(seg)))

coords, ncoords, seg_ids = sample_center_pixels(mask, seg, 2048, seed=0)
nbr, valid = build_pixel_neighbors(coords, seg_ids, 7)

cf = coords.float()
d = torch.cdist(cf, cf)
gathered = torch.gather(d, 1, nbr)
gathered[~valid] = float('nan')

print('\n--- neighbour distance in PIXELS (1080x810 image, bbox diag ~1231px) ---')
per_rank = np.nanmean(gathered.numpy(), axis=0)
for r, v in enumerate(per_rank):
    print(f'  rank {r+1}: mean {v:8.2f}')
allv = gathered.numpy()[valid.numpy()]
print(f'  ALL: mean {allv.mean():.2f}  median {np.median(allv):.2f}  '
      f'p5 {np.percentile(allv,5):.2f}  p95 {np.percentile(allv,95):.2f}  max {allv.max():.2f}')

print('\n--- validity ---')
nv = valid.sum(1).numpy()
print(f'  pixels with all 7 neighbours: {(nv==7).sum()}/2048   mean valid {nv.mean():.2f}')

print('\n--- segment sizes (center pixels per segment) ---')
u, c = np.unique(seg_ids.numpy(), return_counts=True)
order = np.argsort(-c)
for i in order:
    print(f'  seg {u[i]:3d}: {c[i]:5d} centers')
