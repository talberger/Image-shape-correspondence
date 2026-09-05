"""SAM-segment a SHIC demo image, the same way demo/guitar's mask was made.

SHIC's demo PNGs are RGBA but their alpha is fully opaque -- there is no cutout in them, so
`shic_infer_correspondence.py`'s alpha path would return the WHOLE frame (measured on bear3:
606000/606000 pixels) and the back-projection would paint the grass as well as the bear. This
produces a real object mask instead, with a SAM click, matching how
demo/guitar/segment_masks_pixel270_370/ was built.

Saves all three candidate masks plus a preview so the right mask_idx can be chosen by eye, exactly
as was done for the guitar.

Run:
  python analysis/segment_shic_image.py --image_path demo/shic/images/bear3.png \
    --click 300 400 --save_dir demo/shic/masks
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SAM_repo.segment_anything import sam_model_registry, SamPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_path', required=True)
    ap.add_argument('--click', type=int, nargs='+', required=True,
                    help='one or more (row col) pairs, all POSITIVE clicks')
    ap.add_argument('--negative_click', type=int, nargs='*', default=[],
                    help='optional (row col) pairs marked as background')
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--sam_checkpoint', default='SAM_repo/model_checkpoints/sam_vit_h_4b8939.pth')
    ap.add_argument('--sam_type', default='vit_h')
    args = ap.parse_args()

    if len(args.click) % 2 or len(args.negative_click) % 2:
        raise SystemExit('clicks must be (row col) pairs')
    pos = np.array(args.click).reshape(-1, 2)
    neg = np.array(args.negative_click).reshape(-1, 2) if args.negative_click else np.zeros((0, 2), int)

    os.makedirs(args.save_dir, exist_ok=True)
    img = np.array(Image.open(args.image_path).convert('RGB'))
    H, W = img.shape[:2]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint).to(device)
    pred = SamPredictor(sam)
    pred.set_image(img)

    # SAM wants (x, y) = (col, row); the clicks are given (row, col) to match this project's
    # convention everywhere else, so they are swapped exactly once, here.
    pts = np.concatenate([pos, neg]) if len(neg) else pos
    xy = pts[:, ::-1].astype(np.float32)
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int32)
    masks, scores, _ = pred.predict(point_coords=xy, point_labels=labels, multimask_output=True)

    base = os.path.splitext(os.path.basename(args.image_path))[0]
    tag = '_'.join(f'{r}_{c}' for r, c in pos)
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    ax[0].imshow(img); ax[0].set_title('image + clicks', fontsize=12)
    ax[0].scatter(xy[labels == 1][:, 0], xy[labels == 1][:, 1], c='lime', s=90, edgecolor='white')
    if len(neg):
        ax[0].scatter(xy[labels == 0][:, 0], xy[labels == 0][:, 1], c='red', s=90, edgecolor='white')
    for i in range(3):
        m = masks[i]
        out = os.path.join(args.save_dir, f'{base}_click_{tag}_sam_mask_{i}.pth')
        torch.save(torch.from_numpy(m), out)
        frac = 100 * m.sum() / (H * W)
        ax[i + 1].imshow(img)
        ax[i + 1].imshow(np.dstack([np.ones((H, W, 3)) * np.array([0.1, 0.6, 1.0]), m * 0.6]))
        ax[i + 1].set_title(f'mask_idx {i}  score {scores[i]:.3f}  {frac:.1f}% of frame', fontsize=12)
        print(f'mask_idx {i}: score {scores[i]:.4f}  {int(m.sum())} px ({frac:.1f}%)  -> {out}')
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    prev = os.path.join(args.save_dir, f'{base}_click_{tag}_sam_masks_preview.png')
    fig.suptitle(f'{args.image_path}  ({H}x{W})  -- SAM candidate masks', fontsize=14)
    fig.tight_layout(); fig.savefig(prev, dpi=110, bbox_inches='tight')
    print(f'preview: {prev}')


if __name__ == '__main__':
    main()
