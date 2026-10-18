"""Guitar eval v2 along one two-stage LORA-W run: ZS, every kept checkpoint (step dumps and/or stageA_end), final.
One process: each checkpoint is merged at --alpha exactly as analysis/export_pred_f_lora_alpha.py does, then scored by
analysis/eval_geodesic_error.main with the photo features loaded once.
"""
import argparse
import glob
import json
import os
import re
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analysis.eval_geodesic_error as evalmod  # noqa: E402
from analysis.verify_encoder_resume import seed_everything  # noqa: E402
from modules.encoder_l2norm import EncoderL2Norm  # noqa: E402
from modules.lora import merge_lora_into_state_dict  # noqa: E402
from modules.utils import device, load_state_dict, loadmesh  # noqa: E402
from phase2.train_mapping_loss import set_fourier_basis  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--run_dir', required=True)
ap.add_argument('--phase1_dir', required=True)
ap.add_argument('--obj_path', required=True)
ap.add_argument('--name', required=True)
ap.add_argument('--features_path', required=True)
ap.add_argument('--eval_pairs_path', required=True)
ap.add_argument('--geo_table_path', required=True)
ap.add_argument('--alpha', type=float, default=0.5)
ap.add_argument('--temperature', type=float, default=0.01)
ap.add_argument('--out_dir', required=True)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

ckpts = []
for f in glob.glob(os.path.join(a.run_dir, 'encoder_checkpoint_step*.pth')):
    ckpts.append((int(re.search(r'step(\d+)\.pth', f).group(1)), 'step', f))
f = os.path.join(a.run_dir, 'encoder_checkpoint_stageA_end.pth')
if os.path.exists(f):
    ckpts.append((torch.load(f, map_location='cpu')['step'], 'stageA_end', f))
ckpts.sort()
ckpts.append((10 ** 9, 'final', os.path.join(a.run_dir, 'encoder_checkpoint.pth')))

vertices = loadmesh(dir=a.obj_path, name=a.name, load_rings=False).vertices
p1 = torch.load(os.path.join(a.phase1_dir, 'encoder_checkpoint.pth'), map_location=device)
arms = [('ZS', os.path.join(a.phase1_dir, 'pred_f.pth'))]
with torch.no_grad():
    for step, kind, f in ckpts:
        lc = torch.load(f, map_location=device)
        seed_everything(0)
        enc = EncoderL2Norm(4, 2304, out_dim=2304, positional_encoding=True, sigma=5.0, network_verbose=0).to(device)
        set_fourier_basis(enc, lc.get('fourier_basis'))
        load_state_dict(enc, merge_lora_into_state_dict(p1['encoder_state_dict'], lc['lora_state'], a.alpha))
        label = 'final' if kind == 'final' else ('stageA_end(step%d)' % step if kind == 'stageA_end' else 'step%d' % step)
        out = os.path.join(a.out_dir, 'pred_f_%s.pth' % label)
        torch.save(enc(vertices), out)
        arms.append((label, out))
        del enc, lc

feats = evalmod.load_features(a.features_path)
evalmod.load_features = lambda _p: feats
rows = []
for label, pred in arms:
    save = os.path.join(a.out_dir, 'eval_%s.pth' % label)
    evalmod.main(SimpleNamespace(
        replot_from=None, eval_pairs_path=a.eval_pairs_path, features_path=a.features_path,
        vertex_features_path=pred, run_label=label, geo_table_path=a.geo_table_path, obj_path=a.obj_path, name=a.name,
        k_vertices_per_pixel=10, temperature=a.temperature, eps_max=0.20, eps_step=0.01, variant='A',
        plot_path=None, save_path=save))
    d = torch.load(save, map_location='cpu')
    rows.append({'arm': label, 'auc_snap': d['auc']['geometric snap'], 'auc_argmax': d['auc']['feature argmax'],
                 'auc_soft': d['auc']['soft mean distance'], 'err_snap': d['mean_error_hard_snap'],
                 'err_argmax': d['mean_error_argmax'], 'err_soft': d['mean_error_expected']})
with open(os.path.join(a.out_dir, 'ladder.json'), 'w') as f:
    json.dump(rows, f, indent=1)
print('LADDER (AUC x100 snap/argmax/soft | mean err %% of diameter snap/argmax/soft), read T=%g, alpha %g' % (a.temperature, a.alpha))
for r in rows:
    print('%-22s %5.1f %5.1f %5.1f | %5.2f %5.2f %5.2f' % (r['arm'], 100 * r['auc_snap'], 100 * r['auc_argmax'], 100 * r['auc_soft'],
                                                       100 * r['err_snap'], 100 * r['err_argmax'], 100 * r['err_soft']))
print('===== LADDER DONE =====')
