"""
Joins Tal's hand-labeled MeshLab point-picking file (a `.pp`, PickedPoints XML -- each <point>
carries a `name` attribute matching sample_eval_pixels.py's pixel label_id 0-99, plus a raw
(x, y, z) mesh-space position) against pixel_labels.pth (produced by sample_eval_pixels.py: FPS
pixel sample, label_id 0-99 -> row/col/mask_index) to build the ground-truth pixel<->vertex
evaluation set (see PROJECT_NOTES.md's 2026-08-24 "hand-labeled pixel<->vertex evaluation set"
entry). Never assumed the .pp schema matched the older `.pp` tutorials describe -- read the
actual file directly first (confirmed 2026-08-25: standard PickedPoints XML, `name`/`x`/`y`/`z`
attributes per <point>, a `templateName` pointing at the .pptpl MeshLab 2025.07 actually saves
templates as -- so `.pp` is the per-mesh instance file, `.pptpl` the reusable point-set template;
both are real MeshLab formats, not a version quirk).

Vertex lookup is a geometric nearest-vertex snap (Euclidean, brute-force argmin over all mesh
vertices) -- same family of operation as color_mesh_from_soft_mapping.py's own geometric snap,
just point-to-point instead of pixel-to-K-vertices.

COORDINATE SPACE -- read this before pointing --obj_path at a different mesh. The .pp file's
(x,y,z) is MeshLab's RAW display of whatever .obj was opened. modules.utils.loadmesh() then
bounding-sphere-normalizes the mesh in place, and `Normalizer.get_bounding_sphere_normalizer`
centers on `mean(vertices)` -- the VERTEX CENTROID, which depends on how the mesh is sampled, not
only on its shape. For meshes/guitar.obj that normalizer happens to be the identity (shift
[0,0,0], scale 1.0), which is why this script originally compared the raw .pp point straight
against the normalized vertices and got snap distances < 1e-3.

That assumption BROKE on meshes/guitar_isotropic.obj (2026-09-02): the remesh moved ~1000
vertices off the headstock, so the vertex centroid moved by 0.328 in Y and the scale by 1.132x,
and snapping raw .pp points against normalized vertices gave a mean snap distance of 0.130
instead of 0.015 -- a silently wrong evaluation set, not an error. The picked points are now put
through the SAME normalizer loadmesh() applies to the mesh, which is a verified no-op for
meshes/guitar.obj and the correct transform for any other mesh.

Run:
cd /workspace/bsb && python analysis/join_eval_pixel_labels.py \
  --pp_path demo/guitar/eval_labels/guitar_picked_points.pp \
  --pixel_labels_path demo/guitar/eval_labels/pixel_labels.pth \
  --save_dir demo/guitar/eval_labels
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.Normalization.Normalizer import Normalizer
from modules.utils import load_mesh_normalizer_spec, loadmesh


def read_raw_obj_vertices(obj_path):
    """Vertices exactly as they sit in the file, before loadmesh's normalization."""
    verts = []
    with open(obj_path) as fh:
        for line in fh:
            if line.startswith('v '):
                verts.append([float(v) for v in line.split()[1:4]])
    return torch.tensor(verts, dtype=torch.float32)


def parse_pp(pp_path):
    tree = ET.parse(pp_path)
    root = tree.getroot()
    points = {}
    for pt in root.findall('point'):
        name = pt.get('name')
        points[name] = (float(pt.get('x')), float(pt.get('y')), float(pt.get('z')))
    return points


def main(args):
    picked = parse_pp(args.pp_path)
    print(f'{len(picked)} picked points in {args.pp_path}: names = {sorted(picked.keys(), key=int)}')

    labels = torch.load(args.pixel_labels_path)
    label_id_to_idx = {int(lid): i for i, lid in enumerate(labels['label_id'].tolist())}

    mesh = loadmesh(dir=args.obj_path, name=args.name)
    vertices = mesh.vertices  # [num_vertices, 3], already bounding-sphere normalized
    print(f'mesh: {vertices.shape[0]} vertices (matching --obj_path {args.obj_path})')

    # Same normalizer loadmesh() just applied to the mesh, rebuilt from the RAW file so the .pp's
    # raw MeshLab coordinates can be carried into the same space (see this module's docstring).
    raw_vertices = read_raw_obj_vertices(args.obj_path)
    if raw_vertices.shape[0] != vertices.shape[0]:
        raise ValueError(f'raw .obj has {raw_vertices.shape[0]} vertices but loadmesh returned '
                         f'{vertices.shape[0]} -- loadmesh is expected to change coordinates, '
                         f'never vertex count')
    # MUST match whatever loadmesh just did to the mesh, including a `<mesh>.normalizer.json`
    # sidecar -- rebuilding the default normalizer here would put the picked points in a different
    # frame from the vertices they are snapped against, which is silent (2026-09-02, twice).
    raw_vertices = raw_vertices.to(vertices.device)
    if load_mesh_normalizer_spec(args.obj_path) == 'bbox':
        normalizer = Normalizer.get_bounding_sphere_normalizer_bbox(raw_vertices)
    else:
        normalizer = Normalizer.get_bounding_sphere_normalizer(raw_vertices)
    shift = normalizer._shift.tolist()
    scale = float(normalizer._scale)
    is_identity = max(abs(v) for v in shift) < 1e-6 and abs(scale - 1.0) < 1e-6
    print(f'picked-point normalizer for this mesh: shift {[round(v, 6) for v in shift]}, '
          f'scale {scale:.6f}' + ('  (identity -- raw .obj was already normalized)' if is_identity
                                  else '  (NOT identity: raw .pp coordinates are being transformed)'))

    missing = [name for name in picked if int(name) not in label_id_to_idx]
    if missing:
        raise ValueError(f'.pp point name(s) {missing} have no matching label_id in {args.pixel_labels_path} '
                          f'(valid range 0-{labels["label_id"].max().item()})')

    matched_label_id, matched_row, matched_col, matched_vertex_idx, matched_dist, matched_xyz = [], [], [], [], [], []
    for name, (x, y, z) in sorted(picked.items(), key=lambda kv: int(kv[0])):
        label_id = int(name)
        i = label_id_to_idx[label_id]
        row, col = int(labels['row'][i]), int(labels['col'][i])

        raw_pos = torch.tensor([x, y, z], dtype=vertices.dtype, device=vertices.device)
        pos = normalizer(raw_pos[None, :])[0]
        dists = torch.norm(vertices - pos[None, :], dim=1)
        vertex_idx = int(torch.argmin(dists))
        dist = float(dists[vertex_idx])

        matched_label_id.append(label_id)
        matched_row.append(row)
        matched_col.append(col)
        matched_vertex_idx.append(vertex_idx)
        matched_dist.append(dist)
        matched_xyz.append((x, y, z))

        print(f'  label {label_id}: pixel (row={row}, col={col})  ->  vertex {vertex_idx} '
              f'at {tuple(vertices[vertex_idx].tolist())}  (snap distance {dist:.6f})')

    out = {
        'label_id': torch.tensor(matched_label_id),
        'row': torch.tensor(matched_row),
        'col': torch.tensor(matched_col),
        'vertex_idx': torch.tensor(matched_vertex_idx),
        'snap_distance': torch.tensor(matched_dist),
        'picked_xyz': torch.tensor(matched_xyz),
        'pp_path': args.pp_path,
        'pixel_labels_path': args.pixel_labels_path,
        'obj_path': args.obj_path,
    }
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, 'pixel_vertex_eval_pairs.pth')
    torch.save(out, out_path)
    print(f'saved {len(matched_label_id)} pixel<->vertex pairs to {out_path}')

    max_dist = max(matched_dist)
    if max_dist > 0.01:
        print(f'WARNING: max snap distance {max_dist:.6f} is large relative to the mesh\'s '
              f'normalized scale (~1.0 bounding-sphere radius) -- point may not have been clicked '
              f'precisely on the mesh surface, worth a visual sanity check')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pp_path', type=str, required=True)
    parser.add_argument('--pixel_labels_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')
    main(parser.parse_args())
