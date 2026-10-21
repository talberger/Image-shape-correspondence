# Run VLPart on one image and save one panel per detected part (mask overlay + label + score + area %).
# Runs inside /workspace/vlpart_repo (env: vlpart). Usage:
#   python dump_instances.py --image X.png --vocabulary partimagenet --save_dir OUT [--thr 0.3]
import argparse, os, sys, math
import numpy as np, cv2, torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "demo"))
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from demo import setup_cfg
from predictor import VisualizationDemo

p = argparse.ArgumentParser()
p.add_argument("--image", required=True); p.add_argument("--vocabulary", required=True)
p.add_argument("--save_dir", required=True); p.add_argument("--thr", type=float, default=0.3)
p.add_argument("--config", default="configs/joint_in/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.yaml")
p.add_argument("--weights", default="models/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth")
a = p.parse_args()
ns = argparse.Namespace(config_file=a.config, confidence_threshold=a.thr, vocabulary=a.vocabulary,
                        custom_vocabulary="", opts=["MODEL.WEIGHTS", a.weights, "VIS.BOX", "False"])
cfg = setup_cfg(ns)
demo = VisualizationDemo(cfg, ns)
img = read_image(a.image, format="BGR")
pred, _ = demo.run_on_image(img)
inst = pred["instances"].to("cpu")
names = demo.metadata.thing_classes
os.makedirs(a.save_dir, exist_ok=True)
fg = None
sam = a.image.replace(".png", "_sam3_mask.pth")
if os.path.exists(sam):
    m = torch.load(sam); m = m.numpy() if torch.is_tensor(m) else np.asarray(m); fg = m.squeeze() > 0
panels = []
order = np.argsort(-inst.scores.numpy())
for i in order:
    mk = inst.pred_masks[i].numpy(); s = float(inst.scores[i]); n = names[int(inst.pred_classes[i])]
    frac = mk.sum() / (fg.sum() if fg is not None else mk.size)
    v = img.copy(); v[mk] = (0.45 * v[mk] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.putText(v, f"{n} {s:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(v, f"{100*frac:.1f}% of {'horse' if fg is not None else 'image'}", (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    panels.append(v); print(f"{n:30s} score {s:.2f}  area {100*frac:5.1f}%")
cols = min(6, len(panels)); rows = math.ceil(len(panels) / cols)
h, w = img.shape[:2]; grid = np.zeros((rows * h, cols * w, 3), np.uint8)
for k, v in enumerate(panels): grid[(k // cols) * h:(k // cols + 1) * h, (k % cols) * w:(k % cols + 1) * w] = v
cv2.imwrite(os.path.join(a.save_dir, "instances_grid.png"), grid)
torch.save({"masks": inst.pred_masks, "scores": inst.scores, "classes": [names[int(c)] for c in inst.pred_classes]},
           os.path.join(a.save_dir, "instances.pth"))
