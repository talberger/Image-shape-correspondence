"""
Phase-2 relational-loss training, initialized from an L2-normalized (EncoderL2Norm) phase-1
checkpoint instead of the baseline Tanh-based Encoder. New, additive script -- does not modify
phase2/train_relational_loss.py or phase2/train_mapping_loss.py.

Why this checkpoint needs a different loader: phase2/train_mapping_loss.py's
load_phase1_encoder() hardcodes modules/encoder.py's Encoder (Tanh-terminated). The phase-1
checkpoint this script targets (experiments/guitar/encoder_dino_sd_frontcentered_l2norm_phase1/, see
train_encoder_dino_sd_l2norm.py) was trained with modules/encoder_l2norm.py's EncoderL2Norm
instead (F.normalize(dim=-1) as the final op, no Tanh). Since Tanh() and L2Normalize are both
parameter-free, the baseline loader's state_dict.load_state_dict would succeed with NO error --
but every forward pass afterward would then run the wrong final activation, silently training on
wrong features. (train_relational_loss.py's own assert_resume_is_sane resume-gate would likely
catch this via a large cosine mismatch against pred_f.pth, but that's a symptom check, not a fix.)

How this avoids duplicating train_relational_loss.py's ~350-line train() (or its own ~230-line,
heavily-documented argparse block): rather than copying either, this script pre-patches
phase2.train_mapping_loss.load_phase1_encoder to an EncoderL2Norm-aware version, THEN runs
phase2/train_relational_loss.py's own __main__ block completely unmodified via
runpy.run_path(..., run_name='__main__'). Because Python's import system checks sys.modules
before re-executing a module, train_relational_loss.py's own
`from phase2.train_mapping_loss import (..., load_phase1_encoder, ...)` -- which runs as part of
that runpy execution -- picks up the already-patched name instead of the original. The actual
training logic (train(), the loss computation, checkpointing, the CLI surface) is reused 100%
unduplicated; only the encoder-construction function is swapped.

Run: identical CLI to train_relational_loss.py (this script forwards sys.argv to it unchanged),
e.g. the L2-norm counterpart of the current-best recipe:
cd $DISC_ROOT && python phase2/train_relational_loss_l2norm.py \
  --init_encoder_dir ./experiments/guitar/encoder_dino_sd_frontcentered_l2norm_phase1 \
  --object_mask_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_click_270_370_sam_mask_2.pth \
  --fine_seg_path ./demo/guitar/segment_masks_pixel270_370/guitar_01_fine_segments.pth \
  --pix_dino_features_path ./demo/guitar/features_analyze/dino_sd_poc/guitar_01_pix_dino_sd_features.pth \
  --width 2304 --n_classes 2304 --geodesic --mu_3d_scale 1 \
  --geo_table_2d_path ./experiments/guitar/geodesic_tables/guitar_01_click_270_370_sam_mask_2_geo_table_2d_seed0.pth \
  --out_dir ./experiments/guitar/encoder_dino_sd_l2norm_relational_geodesic_mu3dscale1_phase2
"""
import runpy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from analysis.verify_encoder_resume import seed_everything
from modules.encoder_l2norm import EncoderL2Norm
from modules.utils import device, load_state_dict
import phase2.train_mapping_loss as train_mapping_loss


def load_phase1_encoder_l2norm(args):
    """Mirrors phase2/train_mapping_loss.py's load_phase1_encoder exactly, except it builds
    EncoderL2Norm instead of Encoder -- see this file's module docstring for why that's required
    for a checkpoint trained by train_encoder_dino_sd_l2norm.py.
    """
    seed_everything(args.seed)
    encoder = EncoderL2Norm(args.depth, args.width, out_dim=args.n_classes,
                            positional_encoding=args.positional_encoding,
                            sigma=args.sigma, network_verbose=0).to(device)
    checkpoint = torch.load(args.init_encoder_checkpoint, map_location=device)
    load_state_dict(encoder, checkpoint['encoder_state_dict'])
    return encoder


train_mapping_loss.load_phase1_encoder = load_phase1_encoder_l2norm

if __name__ == '__main__':
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    runpy.run_path(os.path.join(_this_dir, 'train_relational_loss.py'), run_name='__main__')
