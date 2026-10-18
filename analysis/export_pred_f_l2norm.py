"""
L2-normalized counterpart to analysis/export_pred_f.py -- new, additive script, does not modify
that file.

Why this is needed: export_pred_f.py hardcodes `from modules.encoder import Encoder` (the
Tanh-terminated architecture). A checkpoint trained by train_encoder_dino_sd_l2norm.py /
phase2/train_relational_loss_l2norm.py uses modules/encoder_l2norm.py's EncoderL2Norm instead
(F.normalize(dim=-1) as the final op, no Tanh). Since Tanh() and L2Normalize are both
parameter-free, export_pred_f.py's load_state_dict would succeed with NO error against an
EncoderL2Norm checkpoint -- but every forward pass would then run the wrong final activation,
silently exporting wrong per-step features. Same failure mode already documented and worked
around in phase2/train_relational_loss_l2norm.py; this file applies the identical fix to
export_pred_f.py instead.

How this avoids duplicating export_pred_f.py's batch-export logic: rather than copying it, this
script patches modules.encoder.Encoder (the name export_pred_f.py imports via
`from modules.encoder import Encoder`) to EncoderL2Norm, THEN runs export_pred_f.py's own
__main__ block unmodified via runpy.run_path(..., run_name='__main__'). Because Python's import
system checks sys.modules before re-executing a module, export_pred_f.py's own
`from modules.encoder import Encoder` -- which runs as part of that runpy execution -- picks up
the already-patched name instead of the original.

Run: identical CLI to export_pred_f.py (single-checkpoint or --checkpoints/--outputs batch mode),
e.g. for the SD-DINO L2norm phase-2 run's step-progression:
cd $DISC_ROOT && python analysis/export_pred_f_l2norm.py \
  --checkpoints <comma-separated step checkpoints> --outputs <comma-separated pred_f paths> \
  --width 2304 --n_classes 2304
"""
import runpy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.encoder as encoder_module
from modules.encoder_l2norm import EncoderL2Norm

encoder_module.Encoder = EncoderL2Norm

if __name__ == '__main__':
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    runpy.run_path(os.path.join(_this_dir, 'export_pred_f.py'), run_name='__main__')
