"""pix2gestalt on occluded DensePose-LVIS instances (Tal, 2026-10-07): is a model-predicted amodal mask worth it?
Input: lvis_test/inputs/<cls>_<id>.npz (256x256 square crop + the visible LVIS mask, built in the bsb env).
Output per instance: the synthesized whole object(s), the amodal mask (threshold on the white background, as the
paper's evaluation does), timing. Run from pix2gestalt_repo/pix2gestalt with the venv active:
  python ../run_lvis_test.py --steps 50 --n_samples 3
"""
import argparse, gc, glob, json, os, sys, time
import numpy as np, torch, cv2
from omegaconf import OmegaConf
sys.path.insert(0, '.')
from inference import load_model_from_config, run_pix2gestalt

ap = argparse.ArgumentParser()
ap.add_argument('--inputs', default='../lvis_test/inputs'); ap.add_argument('--out', default='../lvis_test/out')
ap.add_argument('--steps', type=int, default=50); ap.add_argument('--n_samples', type=int, default=3)
ap.add_argument('--scale', type=float, default=2.0); ap.add_argument('--only', nargs='*', default=None)
a = ap.parse_args()
out = f'{a.out}_steps{a.steps}'; os.makedirs(out, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
t0 = time.time()
# Their loader does torch.load(ckpt, map_location='cuda'): the whole 15.5 GB checkpoint (optimizer state included) lands on the GPU
# and a 16 GB card runs out before the model is even built. Load on the CPU, keep only the weights, then move the model.
from ldm.util import instantiate_from_config
cfg = OmegaConf.load('./configs/sd-finetune-pix2gestalt-c_concat-256.yaml')
sd = torch.load('./ckpt/epoch=000005.ckpt', map_location='cpu')['state_dict']
model = instantiate_from_config(cfg.model); print('load_state_dict:', [len(x) for x in model.load_state_dict(sd, strict=False)], '(missing, unexpected)')
del sd
# ema_scope() clones the whole UNet on every call to swap the EMA weights in (third copy -> OOM on 16 GB). Same weights, done once:
if getattr(model, 'use_ema', False):
    model.model_ema.copy_to(model.model); model.use_ema = False; del model.model_ema
gc.collect(); model = model.to('cuda').eval()
print(f'GPU after model load: {torch.cuda.memory_allocated() / 1e9:.1f} GB', flush=True)
t_load = time.time() - t0; print(f'model load {t_load:.0f} s', flush=True)
times = {}
for f in sorted(glob.glob(f'{a.inputs}/*.npz')):
    name = os.path.basename(f)[:-4]
    if a.only and name not in a.only: continue
    d = np.load(f); img = d['image']; vm = d['mask']
    rgb_mask = np.repeat(vm[:, :, None], 3, axis=2).astype(np.float64)
    torch.cuda.reset_peak_memory_stats(); t = time.time()
    preds = []
    for k in range(a.n_samples):          # one sample per call + cache flush: 3 at once peaks at 12.9 GB and the next instance OOMs on 16 GB
        preds += run_pix2gestalt(model, 'cuda', img, rgb_mask, scale=a.scale, n_samples=1, ddim_steps=a.steps)
        gc.collect(); torch.cuda.empty_cache()
    dt = time.time() - t; times[name] = dt
    masks = [(cv2.cvtColor(p, cv2.COLOR_RGB2GRAY) < 250) for p in preds]
    np.savez(f'{out}/{name}.npz', preds=np.stack(preds), amodal=np.stack(masks))
    print(f'{name}: {dt:.1f} s for {a.n_samples} samples x {a.steps} steps, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB', flush=True)
json.dump({'model_load_s': t_load, 'per_instance_s': times, 'steps': a.steps, 'n_samples': a.n_samples}, open(f'{out}/timing.json', 'w'), indent=1)
print('P2G TEST DONE', out)
