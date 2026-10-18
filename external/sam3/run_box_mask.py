"""SAM 3 object mask for one image, box prompt = the whole image, no text (its SAM-1-style interactive task).
Out: bool [H, W] tensor + a preview png next to it. Called by /workspace/bsb/scripts/make_sam3_mask.sh. Runs in /workspace/sam3_repo/env."""
import argparse
import numpy as np, torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 2026-10-08: several --image / --out pairs in one call = the model is loaded once (scripts/run_lvis_sam3_batch.sh)
ap = argparse.ArgumentParser(); ap.add_argument('--image', nargs='+', required=True); ap.add_argument('--out', nargs='+', required=True); a = ap.parse_args()
assert len(a.image) == len(a.out)
model = build_sam3_image_model(enable_inst_interactivity=True); proc = Sam3Processor(model)
for image, out in zip(a.image, a.out):
    pil = Image.open(image).convert('RGB'); W, H = pil.size
    with torch.autocast('cuda', dtype=torch.bfloat16), torch.inference_mode():
        masks, scores, _ = model.predict_inst(proc.set_image(pil), point_coords=None, point_labels=None,
                                              box=np.array([0, 0, W - 1, H - 1], np.float32)[None], multimask_output=False)
    m = np.asarray(masks).reshape(-1, H, W)[0] > 0
    torch.save(torch.from_numpy(m), out)
    img = np.asarray(pil).astype(np.float32) * 0.35; img[m] = 0.45 * np.asarray(pil)[m] + 0.55 * np.array((255, 90, 220))
    Image.fromarray(img.astype(np.uint8)).save(out.replace('.pth', '.png'))
    print(f'sam3 mask {int(m.sum())} px ({100.0 * m.mean():.1f}% of the crop), score {float(np.asarray(scores).reshape(-1)[0]):.2f} -> {out}', flush=True)
