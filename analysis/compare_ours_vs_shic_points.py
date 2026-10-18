"""Side-by-side of OUR predicted vertex vs SHIC's, per GT point, on one DensePose-LVIS instance.

Tal's ask, 2026-09-20: "to check where the points fell, and their geodesic distance to the labels,
what we get versus SHIC". analysis/eval_lvis_ours.py already produces the NUMBERS; this produces the
PICTURE, because a mean over 2-3 points says nothing about whether a miss is a small local slip or a
left/right flip -- and on this dataset the mean is dominated by flips (median error 18.7 vs mean 33).

What it draws, per GT point i:
  * the photo, with point i marked and numbered, coloured by a fixed per-point palette
  * the mesh, with THREE spheres per point in that same point-colour, distinguished by size:
      GT vertex    -- largest sphere
      OURS         -- medium
      SHIC         -- smallest
    so a point where all three coincide reads as one blob, and a flip reads as two blobs on
    opposite sides of the animal.
  * a printed table of geodesic(GT, pred) for both methods, in the SAME 228-normalized units as
    SHIC Table 1.

The two predictions are NOT computed the same way, deliberately -- each method is asked for its own
prediction, exactly as it would make it:
  * SHIC: their released encoder on their own input (bbox crop resized to 224x224, no mask), GT
    pixel sampled bilinearly from their [1,16,224,224] embedding, argmax over their CSE shape
    features. Identical to analysis/eval_densepose_lvis.py's 'bbox' variant, which reproduces their
    published 33.0 as 33.09.
  * OURS: our per-pixel DINO+SD features at the GT pixel in OUR crop, soft-snap against our phase-2
    pred_f (read T=0.01, k=10) -- the rule PROJECT_NOTES records as ours.
Vertex indices are shared: SHIC's bear.obj IS our demo/shic/shapes/bear.obj (4936v, ref_model
bear_4936), which is why one geodesic matrix serves both.

Run:
cd $DISC_ROOT && python analysis/compare_ours_vs_shic_points.py --ann_id 117683 \
  --run_dir data/densepose_lvis/runs/lvis117683 \
  --our_pred_f data/densepose_lvis/runs/lvis117683/encoder_loraW_all_r4_phase2/alpha/alpha0.5/pred_f.pth \
  --out_dir demo/shic/lvis_vs_shic
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from analysis.color_mesh_from_soft_mapping import compute_soft_snap
from analysis.eval_densepose_lvis import load_instances, get_image, make_crop
from modules.utils import add_spheres, loadmesh, write_ply
from shic_infer_correspondence import load_shic

TF = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# One hue per GT point, reused across the photo and all three spheres of that point.
PALETTE = np.array([[228, 26, 28], [55, 126, 184], [77, 175, 74], [255, 127, 0],
                    [152, 78, 163], [166, 86, 40], [247, 129, 191]], dtype=np.float64)

# Mesh marker colours, by ROLE. GT is the label, so it gets the unambiguous green.
ROLE_GT, ROLE_OURS, ROLE_SHIC = (60, 180, 75), (0, 110, 230), (230, 30, 30)


def split_sphere(center, radius, role_colours, resolution=40):
    """One sphere at `center`, coloured in equal longitude sectors (about the mesh's +y up axis),
    one sector per colour in `role_colours` (RGB 0-255). Seen from any side, 2+ colours show."""
    import open3d as o3d
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    v = np.asarray(s.vertices)
    ang = (np.arctan2(v[:, 2], v[:, 0]) + np.pi) / (2 * np.pi)          # [0, 1) around y
    idx = np.minimum((ang * len(role_colours)).astype(int), len(role_colours) - 1)
    s.vertex_colors = o3d.utility.Vector3dVector(np.asarray(role_colours, dtype=np.float64)[idx] / 255.0)
    s.translate(center)
    return s


def text_marker(text, anchor, normal, radius, height, colour=(0.05, 0.05, 0.05)):
    """Extruded 3D text, `height` tall, placed just outside a marker of `radius` along the surface
    normal and facing outward, upright w.r.t. the mesh's +y axis. Lives in the PLY itself, so it
    shows in any viewer, not only in our renders."""
    import open3d as o3d
    t = o3d.t.geometry.TriangleMesh.create_text(text, depth=1.0).to_legacy()
    tv = np.asarray(t.vertices)
    lo, hi = tv.min(0), tv.max(0)
    scale = height / max(hi[1] - lo[1], 1e-9)
    tv = (tv - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])) * scale
    n = normal / max(np.linalg.norm(normal), 1e-9)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, n)
    if np.linalg.norm(x) < 1e-3:                 # normal parallel to up: any horizontal axis
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.stack([x, y, n], axis=1)              # text x->x, text y->y, extrusion->normal
    t.vertices = o3d.utility.Vector3dVector(tv @ R.T + anchor + n * (radius + 0.02) + y * (radius + height / 2))
    t.vertex_colors = o3d.utility.Vector3dVector(np.tile(colour, (len(tv), 1)))
    return t


def shic_predict(ann, imgs, device, class_name):
    """SHIC's own prediction at each GT pixel: their weights, their input frame, their argmax."""
    enc, shape_feats, _verts, _faces = load_shic(class_name, device)
    img = get_image(imgs[ann['image_id']])
    crop, (cx, cy, cw, ch) = make_crop(img, ann, 'bbox')
    with torch.no_grad():
        feats = enc(TF(crop.resize((224, 224), Image.BILINEAR)).unsqueeze(0).to(device))
    bx, by, bw, bh = ann['bbox']
    px = bx + np.array(ann['dp_x']) / 256.0 * bw
    py = by + np.array(ann['dp_y']) / 256.0 * bh
    u = (px - cx) / cw * 2 - 1
    v = (py - cy) / ch * 2 - 1
    grid = torch.tensor(np.stack([u, v], -1), dtype=torch.float32, device=device)[None, None]
    emb = F.grid_sample(feats, grid, mode='bilinear', align_corners=False)[0, :, 0].T
    return (emb @ shape_feats.T).argmax(1).cpu().numpy()


def shic_cached(ann, imgs, device, class_name, tag):
    """2026-10-08 (speed): SHIC's prediction at the GT points depends only on the image + bbox, so it is cached per instance
    (loading SHIC is most of this script's time). Delete the file to recompute."""
    cache = os.path.join('data/densepose_lvis', f'eval_shic_{class_name}', 'point_cache', f'{tag}.json')
    if os.path.exists(cache):
        return np.array(json.load(open(cache))['shic_vertex'])
    shic_v = shic_predict(ann, imgs, device, class_name)
    os.makedirs(os.path.dirname(cache), exist_ok=True); json.dump({'ann_id': ann['id'], 'shic_vertex': shic_v.tolist()}, open(cache, 'w'))
    return shic_v


def our_predict(run_dir, tag, our_pred_f, obj_path, name, rows, cols, k, temperature, device):
    """Our prediction at the same GT pixels: our features in OUR crop, soft-snap + argmax."""
    pix = torch.load(os.path.join(run_dir, f'{tag}_pix_sdsquare_fuse16.pth'), map_location='cpu')
    pixel_features = pix[rows, cols]
    del pix
    vf = torch.load(our_pred_f, map_location='cpu')
    if isinstance(vf, dict):
        vf = vf.get('pred_f', vf)
    mesh = loadmesh(dir=obj_path, name=name)
    vpos = mesh.vertices.detach().cpu()
    soft, _ = compute_soft_snap(pixel_features, vf, vpos, k=k, chunk_size=4096,
                                device=device, temperature=temperature)
    pf = F.normalize(pixel_features.detach().to(device).float(), dim=-1)
    argmax = (pf @ vf.to(device).float().T).argmax(1).cpu().numpy()
    return soft.numpy(), argmax, mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann_id', type=int, required=True)
    ap.add_argument('--class_name', default='bear')
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--tag', default=None, help='defaults to lvis<ann_id>')
    ap.add_argument('--our_pred_f', default=None)
    ap.add_argument('--shic_only', action='store_true', help='only compute + cache SHIC\'s prediction at the GT points, then exit')
    ap.add_argument('--obj_path', default='./demo/shic/shapes/bear.obj')
    ap.add_argument('--name', default='shic_bear')
    ap.add_argument('--geodesics_path', default='data/densepose_lvis/eval_shic_bear/bear_heat_geodesics_228.npy')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--k_vertices_per_pixel', type=int, default=10)
    ap.add_argument('--read_temperature', type=float, default=0.01)
    ap.add_argument('--sphere_radius', type=float, default=0.055)
    ap.add_argument('--label_height', type=float, default=0.07,
                    help='height of the 3D point-number text next to each marker (mesh units; bear is ~1.1 tall)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    tag = args.tag or f'lvis{args.ann_id}'
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    gt = torch.load(os.path.join(args.run_dir, f'{tag}_gt.pth'), map_location='cpu')
    rows, cols = gt['rows'].numpy(), gt['cols'].numpy()
    gt_vertex = gt['vertex'].numpy()
    n = len(gt_vertex)

    anns, imgs = load_instances(args.class_name)
    ann = next(a for a in anns if a['id'] == args.ann_id)

    if args.shic_only:   # fill the cache while our model still trains (scripts/debug_twostage_instance.sh, early chain)
        shic_cached(ann, imgs, device, args.class_name, tag); print('SHIC point cache ready'); return
    ours_soft, ours_argmax, mesh = our_predict(
        args.run_dir, tag, args.our_pred_f, args.obj_path, args.name, rows, cols,
        args.k_vertices_per_pixel, args.read_temperature, device)
    shic_v = shic_cached(ann, imgs, device, args.class_name, tag)

    G = np.load(args.geodesics_path)
    err_ours = G[gt_vertex, ours_soft]
    err_ours_am = G[gt_vertex, ours_argmax]
    err_shic = G[gt_vertex, shic_v]

    # --- the photo, GT points numbered in the per-point palette ---
    photo = Image.open(os.path.join(args.run_dir, f'{tag}.png')).convert('RGB')
    draw = ImageDraw.Draw(photo)
    r = max(4, int(0.012 * max(photo.size)))
    for i in range(n):
        c = tuple(int(v) for v in PALETTE[i % len(PALETTE)])
        draw.ellipse([cols[i] - r, rows[i] - r, cols[i] + r, rows[i] + r], outline=c, width=max(2, r // 3))
        draw.text((cols[i] + r + 2, rows[i] - r), str(i + 1), fill=c)
    photo_path = os.path.join(args.out_dir, f'{tag}_photo_gt_points.png')
    photo.save(photo_path)

    # --- the mesh, three spheres per point (GT biggest, ours medium, SHIC smallest) ---
    # Coloured by ROLE, not by point: with only 2-3 GT points per instance, "which method put its
    # sphere where" is the question, and a per-point palette made that unreadable (first attempt).
    # GT green / ours blue / SHIC red, with size as a secondary cue (GT largest).
    # Markers that land on the SAME vertex (e.g. an exact hit, GT == ours == SHIC) are merged into ONE
    # split sphere, coloured in longitude sectors per role -- otherwise the GT sphere, being largest,
    # hides the others completely (Tal, 2026-09-21, ann 3044 pt 3). Every marker also gets its GT
    # point number as 3D text just outside the surface, so each sphere can be tied to its label.
    verts = mesh.vertices.detach().cpu().numpy().astype(np.float64)
    groups = {}                                  # vertex -> list of (role colour, scale, point number)
    for i in range(n):
        for vtx, scale, c in ((gt_vertex[i], 1.0, ROLE_GT),
                              (ours_soft[i], 0.70, ROLE_OURS),
                              (shic_v[i], 0.70, ROLE_SHIC)):
            groups.setdefault(int(vtx), []).append((c, scale, i + 1))
    import open3d as o3d
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces.detach().cpu().numpy().astype(np.int32))
    o3d_mesh.compute_vertex_normals()
    normals = np.asarray(o3d_mesh.vertex_normals).copy()
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.full_like(verts, 0.78))
    combined = o3d_mesh
    for vtx, members in groups.items():
        roles = [ROLE_GT, ROLE_OURS, ROLE_SHIC]
        present = [r for r in roles if any(m[0] == r for m in members)]
        radius = args.sphere_radius * max(m[1] for m in members)
        if len(present) == 1:
            combined = add_spheres(combined, verts[[vtx]], np.array([np.asarray(present[0]) / 255.0]),
                                   sphere_radius=radius, sphere_resolution=12)
        else:
            combined += split_sphere(verts[vtx], radius, present)
        label = ','.join(str(p) for p in sorted({m[2] for m in members}))
        combined += text_marker(label, verts[vtx], normals[vtx], radius, args.label_height)
    # modules.utils.write_ply, NOT o3d.io.write_triangle_mesh: open3d names the face property
    # 'vertex_indices' while analysis/render_colored_mesh.py's reader expects 'vertex_index', so an
    # open3d-written PLY fails to load with "no field of name vertex_index".
    ply_path = os.path.join(args.out_dir, f'{tag}_markers.ply')
    write_ply(ply_path,
              np.asarray(combined.vertices),
              np.asarray(combined.triangles),
              (np.asarray(combined.vertex_colors) * 255).astype(np.uint8))

    result = {'ann_id': args.ann_id, 'n_points': n,
              'gt_vertex': gt_vertex.tolist(),
              'ours_soft_vertex': ours_soft.tolist(), 'ours_argmax_vertex': ours_argmax.tolist(),
              'shic_vertex': shic_v.tolist(),
              'err_ours_soft': err_ours.tolist(), 'err_ours_argmax': err_ours_am.tolist(),
              'err_shic': err_shic.tolist(),
              'mean_ours_soft': float(err_ours.mean()), 'mean_ours_argmax': float(err_ours_am.mean()),
              'mean_shic': float(err_shic.mean()),
              'photo': photo_path, 'ply': ply_path}
    json.dump(result, open(os.path.join(args.out_dir, f'{tag}_vs_shic.json'), 'w'), indent=1)

    print(f'--- {tag} (ann {args.ann_id}), geodesic error, 228-normalized ---')
    print(f'{"pt":>3} {"GT v":>6} {"ours v":>7} {"SHIC v":>7} | {"ours":>7} {"SHIC":>7}   winner')
    for i in range(n):
        w = 'ours' if err_ours[i] < err_shic[i] else ('SHIC' if err_shic[i] < err_ours[i] else 'tie')
        print(f'{i+1:>3} {gt_vertex[i]:>6} {ours_soft[i]:>7} {shic_v[i]:>7} | '
              f'{err_ours[i]:>7.2f} {err_shic[i]:>7.2f}   {w}')
    print(f'{"mean":>3} {"":>6} {"":>7} {"":>7} | {err_ours.mean():>7.2f} {err_shic.mean():>7.2f}')
    print(f'(ours argmax, for reference: {err_ours_am.mean():.2f})')
    print(f'photo -> {photo_path}\nply   -> {ply_path}')


if __name__ == '__main__':
    main()
