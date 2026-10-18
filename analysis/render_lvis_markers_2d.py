"""The mesh row of the LVIS summary figure (Tal, 2026-10-05; second version the same day).

The first figure (compare_ours_vs_shic_points.py -> <tag>_markers.ply) had real spheres plus 3D digit geometry: the
digits were unreadable. A first rewrite drew flat 2D discs over the render with "dashed ring = behind the mesh":
Tal -- not clear enough, you cannot tell what is in front of the animal and what is behind it.

This version keeps the markers IN 3D, so the renderer's own occlusion answers front/behind:
  * spheres on the surface (GT green, largest / ours blue / SHIC red), the point NUMBER written on each in 2D;
  * a sphere the animal HIDES in a view is still drawn, FADED (semi-transparent over the mesh, Tal's idea): solid =
    on this side, faded = on the far side. It comes from a second render of the spheres alone, same camera.
Geodesic label->prediction paths as tubes on the surface were tried the same day and dropped (Tal: they clutter).

Reads <base>/<tag>_vs_shic.json (vertex ids), writes <base>/<tag>_mesh_views_2d.png.

  python analysis/render_lvis_markers_2d.py --ann_id 159218 --base demo/shic/lvis_vs_shic_twostage
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import torch
import kaolin as kal
import open3d as o3d

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from modules.utils import device, loadmesh, setcolor_mesh_batched, get_camera_from_view2  # noqa: E402
from modules.render import Renderer  # noqa: E402
from analysis.render_colored_mesh import RenderMesh  # noqa: E402

# brighter than the legend swatches on purpose: the renderer's lighting darkens them
ROLE = {'gt': (0.25, 1.0, 0.35), 'ours': (0.15, 0.55, 1.0), 'shic': (1.0, 0.2, 0.2)}


def render_view(renderer, mesh, elev_deg, azim_deg):
    """-> (H x W x 3 image, H x W index of the face drawn at each pixel (-1 = background))."""
    elev = torch.tensor([elev_deg], dtype=torch.float32).to(device) / 180 * np.pi
    azim = torch.tensor([azim_deg], dtype=torch.float32).to(device) / 360 * 2 * np.pi
    images, _, _ = renderer.render_views(mesh, num_views=1, show=False, random_views=False,
                                         center_elev=elev, center_azim=azim, lighting=True,
                                         background=torch.ones(3).to(device), return_views=True)
    img = images[0].permute(1, 2, 0).detach().cpu().numpy()
    H, W = img.shape[:2]
    # the same camera render_views built, so face_idx is exactly what was drawn
    cam = get_camera_from_view2(elev[0], azim[0], r=renderer.radius).to(device)
    fv_cam, fv_img, f_normals = kal.render.mesh.prepare_vertices(
        mesh.vertices.to(device), mesh.faces.to(device), renderer.camera_projection, camera_transform=cam)
    _, _, face_idx = kal.render.mesh.dibr_rasterization(
        H, W, fv_cam[:, :, :, -1], fv_img, torch.ones((1, mesh.faces.shape[0], 3, 1), device=device), f_normals[:, :, -1])
    return img, face_idx[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann_id', type=int, required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--class_name', default='bear')
    ap.add_argument('--obj_path', default=None)
    ap.add_argument('--render_res', type=int, default=700)
    ap.add_argument('--n_views', type=int, default=4)
    ap.add_argument('--cand_elev', nargs='+', type=float, default=[15, -20],
                    help='candidate elevations; a negative one looks up at the belly / inner legs')
    ap.add_argument('--sphere_radius', type=float, default=0.04, help='GT sphere, mesh units (predictions are 0.85x)')
    ap.add_argument('--hidden_alpha', type=float, default=0.38, help='opacity of a sphere that the mesh hides in a view')
    ap.add_argument('--output_path', default=None)
    args = ap.parse_args()
    tag = f'lvis{args.ann_id}'
    res = json.load(open(os.path.join(args.base, f'{tag}_vs_shic.json')))
    out = args.output_path or os.path.join(args.base, f'{tag}_mesh_views_2d.png')

    mesh = loadmesh(args.obj_path or f'./demo/shic/shapes/{args.class_name}.obj', name=f'shic_{args.class_name}')
    V = mesh.vertices.detach().cpu().numpy().astype(np.float64)
    F = mesh.faces.detach().cpu().numpy().astype(np.int32)
    base = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    base.compute_vertex_normals()
    N = np.asarray(base.vertex_normals).copy()
    base.paint_uniform_color((0.80, 0.80, 0.80))
    combined = base
    n_base = len(F)

    n = res['n_points']
    marks = []                                     # (point number, role, first face, one-past-last face)
    stacked = {}                                   # vertex -> how many spheres already sit on it
    for i in range(n):
        g = int(res['gt_vertex'][i])
        for role, key, scale in (('gt', 'gt_vertex', 1.0), ('ours', 'ours_soft_vertex', 0.85), ('shic', 'shic_vertex', 0.85)):
            vtx = int(res[key][i])
            k = stacked.get(vtx, 0); stacked[vtx] = k + 1
            r = args.sphere_radius * scale
            # several markers on ONE vertex (an exact hit) are stacked outwards along the normal so all stay visible
            centre = V[vtx] + N[vtx] * (k * 1.5 * args.sphere_radius)
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=14).translate(centre)
            sph.paint_uniform_color(ROLE[role])
            f0 = len(combined.triangles)
            combined += sph
            marks.append((i + 1, role, f0, len(combined.triangles), r))

    rmesh = RenderMesh(torch.from_numpy(np.asarray(combined.vertices)).float().to(device),
                       torch.from_numpy(np.asarray(combined.triangles)).long().to(device))
    rmesh.frame_locked = getattr(mesh, 'frame_locked', False)
    setcolor_mesh_batched(rmesh, torch.from_numpy(np.asarray(combined.vertex_colors)).float().to(device).unsqueeze(0))
    # the spheres WITHOUT the animal, in the very same (already normalized) frame: same vertex tensor, sphere faces only
    smesh = RenderMesh(rmesh.vertices, rmesh.faces[n_base:])
    smesh.face_attributes = rmesh.face_attributes[:, n_base:]
    # VIEW CHOICE (2026-10-05): fixed azimuths left markers on the chest/belly hidden in all four views. Score a grid
    # of candidate cameras at low resolution by how much of each sphere shows, then greedily keep the --n_views that
    # together show every marker best (a view only earns what it adds over the views already chosen).
    lo = Renderer(dim=(240, 240))
    cands = [(e, a) for e in args.cand_elev for a in np.arange(0, 360, 30)]
    full = np.array([np.pi * (r * 240 / (2 * lo.radius * np.tan(np.pi / 6))) ** 2 for *_, r in marks])
    score = np.zeros((len(cands), len(marks)))
    for c, (e, a) in enumerate(cands):
        _, fi = render_view(lo, rmesh, e, a)
        score[c] = [min(((fi >= f0) & (fi < f1)).sum() / full[m], 1.0) for m, (_, _, f0, f1, _) in enumerate(marks)]
    chosen, best = [], np.zeros(len(marks))
    for _ in range(args.n_views):
        gain = np.maximum(score, best).sum(1) - best.sum()
        gain[chosen] = -1
        c = int(np.argmax(gain)); chosen.append(c); best = np.maximum(best, score[c])
    chosen.sort(key=lambda c: cands[c][1])
    print('views (elev, azim):', [cands[c] for c in chosen], '| worst marker visibility %.2f' % best.min())

    renderer = Renderer(dim=(args.render_res, args.render_res))
    views = []
    for c in chosen:
        img, fi = render_view(renderer, rmesh, *cands[c])
        simg, sfi = render_view(renderer, smesh, *cands[c])
        sfi = np.where(sfi >= 0, sfi + n_base, -1)                       # back to the combined mesh's face numbering
        hidden = (sfi >= 0) & (fi >= 0) & (fi < n_base)                  # a sphere is there, but the animal was drawn
        img = img.copy()
        img[hidden] = (1 - args.hidden_alpha) * img[hidden] + args.hidden_alpha * simg[hidden]
        views.append((img, fi, np.where(hidden, sfi, -1)))
    px_per_unit = args.render_res / (2 * renderer.radius * np.tan(np.pi / 6))      # at the mesh centre
    # every view cropped to its OWN footprint, all crops the same size, so the animal fills each panel
    boxes = []
    for _, fi, _ in views:
        ys, xs = np.nonzero(fi >= 0)
        boxes.append((xs.min(), xs.max(), ys.min(), ys.max()))
    half_w = max(b[1] - b[0] for b in boxes) / 2 * 1.08
    half_h = max(b[3] - b[2] for b in boxes) / 2 * 1.08

    fig, axs = plt.subplots(1, len(views), figsize=(5.2 * len(views), 5.2 * half_h / half_w), squeeze=False)
    for ax, (img, fi, hfi), b in zip(axs[0], views, boxes):
        cx, cy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
        ax.imshow(img); ax.axis('off')
        ax.set_xlim(cx - half_w, cx + half_w); ax.set_ylim(cy + half_h, cy - half_h)
        for num, role, f0, f1, r in marks:
            yy, xx = np.nonzero((fi >= f0) & (fi < f1))
            area = np.pi * (r * px_per_unit) ** 2
            if len(yy) >= 0.35 * area:                 # enough of the sphere shows: the number goes on its visible part
                ax.text(xx.mean() + 0.5, yy.mean() + 0.5, str(num), color='white', fontsize=9.5 if role == 'gt' else 8,
                        weight='bold', ha='center', va='center',
                        path_effects=[pe.withStroke(linewidth=1.4, foreground='black')])
                continue
            yy, xx = np.nonzero((hfi >= f0) & (hfi < f1))
            if len(yy) >= 0.35 * area:                 # hidden behind the animal: faded sphere, faded number
                ax.text(xx.mean() + 0.5, yy.mean() + 0.5, str(num), color='white', fontsize=9.5 if role == 'gt' else 8,
                        weight='bold', ha='center', va='center', alpha=0.75,
                        path_effects=[pe.withStroke(linewidth=1.2, foreground='0.35')])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    fig.savefig(out, dpi=150, facecolor='white')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
