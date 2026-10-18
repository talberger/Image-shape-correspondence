"""Step 2/3 of the amodal mask (pix2gestalt venv): <run_dir>/<tag>_amodal_input.npz -> <tag>_amodal_samples.npz
(the synthesized whole objects + their masks = threshold on the white background, as in the paper's evaluation).
Run from pix2gestalt_repo/pix2gestalt:  python ../run_amodal.py --run_dir <dir> --tags lvis7107 [lvis...] --n_samples 3 --steps 50
Memory fixes vs their loader (see PROJECT_NOTES 2026-10-07): checkpoint loaded on the CPU, EMA weights copied in once."""
import argparse, gc, os, sys, time
import numpy as np, torch, cv2
from omegaconf import OmegaConf
sys.path.insert(0, '.')
from inference import run_pix2gestalt
from ldm.util import instantiate_from_config

ap = argparse.ArgumentParser(); ap.add_argument('--run_dirs', nargs='+', required=True, help='one run dir per tag')
ap.add_argument('--tags', nargs='+', required=True); ap.add_argument('--n_samples', type=int, default=3)
ap.add_argument('--wait_for_input', type=float, default=0, help='2026-10-08 (speed): started BEFORE the input exists, so the model loads while SAM 3 '
                'builds the mask. After loading, wait up to this many seconds per tag for <tag>_amodal_input.ready; <tag>_amodal.done appearing first '
                '(scripts/make_amodal_mask.sh decided the mask is in one piece) = nothing to do for that tag. 0 = the input is already there')
ap.add_argument('--steps', type=int, default=50); ap.add_argument('--scale', type=float, default=2.0); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args(); t0 = time.time()
# 2026-10-08 (speed): ckpt/slim_ema.pt = the same model after the EMA copy, without the EMA duplicate / optimizer state of the 15.5 GB
# training checkpoint. Written once from the full checkpoint, then loaded instead of it (same weights -> same samples).
SLIM = './ckpt/slim_ema.pt'
model = instantiate_from_config(OmegaConf.load('./configs/sd-finetune-pix2gestalt-c_concat-256.yaml').model)
if os.path.exists(SLIM):
    model.use_ema = False; del model.model_ema
    missing, unexpected = model.load_state_dict(torch.load(SLIM, map_location='cpu'), strict=False); assert not missing, missing[:5]
else:
    sd = torch.load('./ckpt/epoch=000005.ckpt', map_location='cpu')['state_dict']; model.load_state_dict(sd, strict=False); del sd
    if getattr(model, 'use_ema', False):
        model.model_ema.copy_to(model.model); model.use_ema = False; del model.model_ema
    torch.save(model.state_dict(), SLIM + '.tmp'); os.replace(SLIM + '.tmp', SLIM); print('wrote', SLIM, flush=True)
gc.collect(); model = model.to('cuda').eval(); print(f'model load {time.time() - t0:.0f} s', flush=True)
for D, T in zip(a.run_dirs, a.tags):
    if a.wait_for_input > 0:
        tw = time.time()
        while not os.path.exists(f'{D}/{T}_amodal_input.ready') and not os.path.exists(f'{D}/{T}_amodal.done') and time.time() - tw < a.wait_for_input:
            time.sleep(1)
        if not os.path.exists(f'{D}/{T}_amodal_input.ready'):
            print(f'{T}: no input after {time.time() - tw:.0f} s (mask in one piece, or timeout) -- nothing sampled', flush=True); continue
    d = np.load(f'{D}/{T}_amodal_input.npz'); rgb_mask = np.repeat(d['mask'][:, :, None], 3, axis=2).astype(np.float64)
    torch.manual_seed(a.seed); np.random.seed(a.seed); t = time.time(); preds = []   # fixed seed -> the same mask every run
    for _ in range(a.n_samples):
        preds += run_pix2gestalt(model, 'cuda', d['image'], rgb_mask, scale=a.scale, n_samples=1, ddim_steps=a.steps)
        gc.collect(); torch.cuda.empty_cache()
    np.savez(f'{D}/{T}_amodal_samples.npz', preds=np.stack(preds), amodal=np.stack([cv2.cvtColor(p, cv2.COLOR_RGB2GRAY) < 250 for p in preds]))
    print(f'{T}: {a.n_samples} samples x {a.steps} steps in {time.time() - t:.1f} s', flush=True)
    open(f'{D}/{T}_amodal_samples.done', 'w').close()
print('AMODAL SAMPLES DONE')
