"""Geodesic error on DensePose-LVIS (SHIC Table 1 protocol) -- currently for SHIC's released weights.

Protocol (SHIC arXiv 2407.18907, Table 1; see PROJECT_NOTES 2026-09-19):
  * test set  : ds2_val, instances of the class that carry dp_x/dp_y/dp_vertex (bear: 132 instances, 105 images)
  * GT points : dp_x/dp_y live in a 256x256 frame over the instance bbox -> image px = bbox_xy + dp/256 * bbox_wh
  * prediction: embedding at the GT pixel -> argmax vertex (dot product, as in SHIC's own inference)
  * error     : geodesic(GT vertex, predicted vertex), heat-method geodesics, max geodesic normalized to 228
  * reported  : mean over all points (and mean-of-instance-means for reference)

Input variants (the paper does not say how the test image is fed; SHIC resizes to 224x224, not
aspect-preserving):
  bbox        : crop to the bbox, resize to 224x224 (aspect distorted)
  square      : square crop centred on the bbox (side = max(w,h), zero-padded outside the image)
  square_mask : as square, background (outside the LVIS polygon) set to black

Run:
  python analysis/eval_densepose_lvis.py --class_name bear --out_dir data/densepose_lvis/eval_shic_bear
"""
import argparse, json, os, sys, urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shic_infer_correspondence import load_shic  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), 'data', 'densepose_lvis')
TF = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def load_instances(class_name, split='ds2_val'):
    d = json.load(open(os.path.join(DATA, f'densepose_lvis_v1_{split}_v1.json')))
    cat = {c['id']: c['name'] for c in d['categories']}
    imgs = {im['id']: im for im in d['images']}
    anns = [a for a in d['annotations'] if cat[a['category_id']] == class_name and a.get('dp_x')]
    return anns, imgs


def get_image(im):
    path = os.path.join(DATA, 'coco_imgs', f"{im['id']}.jpg")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(im['coco_url'], path)
    return Image.open(path).convert('RGB')


def polygon_mask(seg, H, W):
    m = Image.new('L', (W, H), 0)
    for poly in seg:
        if len(poly) >= 6:
            ImageDraw.Draw(m).polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=255)
    return np.array(m) > 0


def make_crop(img, ann, variant):
    """Return (crop PIL, (x0, y0, side_w, side_h)) -- the crop's box in image coords."""
    x, y, w, h = ann['bbox']
    if variant == 'bbox':
        box = (x, y, w, h)
    else:
        s = max(w, h)
        box = (x + w / 2 - s / 2, y + h / 2 - s / 2, s, s)
    arr = np.array(img)
    if variant == 'square_mask':
        arr = arr * polygon_mask(ann['segmentation'], *arr.shape[:2])[..., None]
    # PIL crop zero-pads outside the image
    crop = Image.fromarray(arr.astype(np.uint8)).crop(
        (round(box[0]), round(box[1]), round(box[0] + box[2]), round(box[1] + box[3])))
    return crop, (round(box[0]), round(box[1]), crop.size[0], crop.size[1])


def heat_geodesics(verts, faces):
    import potpourri3d as pp3d
    solver = pp3d.MeshHeatMethodDistanceSolver(verts, faces)
    G = np.stack([solver.compute_distance(i) for i in range(len(verts))])
    G = 0.5 * (G + G.T)
    return G * (228.0 / G.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class_name', default='bear')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--variants', default='bbox,square,square_mask')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)

    anns, imgs = load_instances(args.class_name)
    print(f'{args.class_name}: {len(anns)} instances, {len({a["image_id"] for a in anns})} images, '
          f'{sum(len(a["dp_x"]) for a in anns)} points')

    enc, shape_feats, verts, faces = load_shic(args.class_name, dev)
    gpath = os.path.join(args.out_dir, f'{args.class_name}_heat_geodesics_228.npy')
    if os.path.exists(gpath):
        G = np.load(gpath)
    else:
        G = heat_geodesics(verts.astype(np.float64), faces.astype(np.int64))
        np.save(gpath, G)

    results = {}
    for variant in args.variants.split(','):
        errs, per_inst = [], []
        for a in anns:
            img = get_image(imgs[a['image_id']])
            crop, (cx, cy, cw, ch) = make_crop(img, a, variant)
            with torch.no_grad():
                feats = enc(TF(crop.resize((224, 224), Image.BILINEAR)).unsqueeze(0).to(dev))  # [1,16,224,224]
            bx, by, bw, bh = a['bbox']
            px = bx + np.array(a['dp_x']) / 256.0 * bw          # image coords
            py = by + np.array(a['dp_y']) / 256.0 * bh
            u = (px - cx) / cw * 2 - 1                           # crop coords in [-1, 1]
            v = (py - cy) / ch * 2 - 1
            grid = torch.tensor(np.stack([u, v], -1), dtype=torch.float32, device=dev)[None, None]
            emb = F.grid_sample(feats, grid, mode='bilinear', align_corners=False)[0, :, 0].T  # [P,16]
            pred = (emb @ shape_feats.T).argmax(1).cpu().numpy()
            e = G[np.array(a['dp_vertex']) - 0, pred]
            errs.extend(e.tolist()); per_inst.append(float(e.mean()))
        results[variant] = {'mean_point': float(np.mean(errs)), 'mean_instance': float(np.mean(per_inst)),
                            'median_point': float(np.median(errs)), 'n_points': len(errs),
                            'per_instance': {str(a['id']): m for a, m in zip(anns, per_inst)}}
        print(f'{variant:12s}  geodesic error mean {np.mean(errs):6.2f}  (median {np.median(errs):6.2f}, '
              f'mean of instance means {np.mean(per_inst):6.2f}, n={len(errs)})')
    json.dump(results, open(os.path.join(args.out_dir, 'results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
