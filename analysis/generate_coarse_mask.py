"""
Standalone coarse-mask generator: runs SAM's single-click predictor for a given
(pixel, mask_idx) and saves the resulting object mask, without running the rest
of the image_shape_correspondence.py pipeline (no mesh/decoder/DINO loading).
Mirrors BestSegmentationBuddies.prepare_data()'s coarse-mask save convention
(<img_name>_click_<x>_<y>_sam_mask_<idx>.pth) so downstream --mask_path
consumers can use the output directly.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch

from modules.best_segmentation_buddies import BestSegmentationBuddies
from modules.sam import load_img_seg_model
from modules.utils import get_img_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./images/guitar_01.png')
    parser.add_argument('--pixel', nargs=2, type=int, required=True, help='Click pixel (x, y)')
    parser.add_argument('--mask_idx', type=int, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--sam_dir', type=str, default='./SAM_repo/model_checkpoints/')
    parser.add_argument('--sam_fname', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_model_type', type=str, default='vit_h')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    image = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    sam = load_img_seg_model(args.sam_dir, args.sam_fname, args.sam_model_type)

    mask = BestSegmentationBuddies.test_seg_click(
        args.img_path, args.pixel, args.mask_idx,
        mask_color=[30, 144, 255, 0.6], show_test_click=True,
        save_dir=args.save_dir, image=image, sam=sam,
    )
    # match prepare_data()'s own save convention (torch bool tensor, not numpy)
    mask = torch.as_tensor(mask, dtype=torch.bool)

    img_name = get_img_name(args.img_path)
    click_str = 'click_%d_%d_sam_mask_%d' % (args.pixel[0], args.pixel[1], args.mask_idx)
    save_path = os.path.join(args.save_dir, '_'.join([img_name, click_str]) + '.pth')
    torch.save(mask, save_path)

    coverage = mask.float().mean().item()
    print('Coarse mask saved to %s' % save_path)
    print('Coverage: %.1f%% of image (%d px)' % (coverage * 100, mask.sum().item()))
