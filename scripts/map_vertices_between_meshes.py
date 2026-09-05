"""Map vertex indices from one mesh to the nearest vertex of another (2026-09-02).

Needed because several analysis calls carry HARD-CODED vertex ids -- e.g. the project's standard
similarity-heatmap query set `--query_vertices 1527 1545 2760 642`. Those ids mean nothing on a
remeshed surface, and every one of them is still < 3886, so they stay silently "valid" instead of
raising: the heatmap would render happily for whatever anatomy now sits at index 1527.

COORDINATE SPACE is the whole difficulty. modules.utils.loadmesh normalizes by `mean(vertices)`,
the vertex CENTROID -- which depends on how the surface is sampled, not only on its shape. So two
triangulations of the SAME object do not share a frame: meshes/guitar_isotropic.obj sits 0.328
(in Y) away from meshes/guitar.obj in loadmesh space, purely because the remesh moved ~1000
vertices off the headstock. Both meshes' RAW file coordinates ARE in a common frame (pymeshlab
does not transform), so this maps raw -> raw and only normalizes for reporting.
"""
import argparse, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.Normalization.Normalizer import Normalizer


def read_raw_obj_vertices(obj_path):
    verts = []
    with open(obj_path) as fh:
        for line in fh:
            if line.startswith('v '):
                verts.append([float(v) for v in line.split()[1:4]])
    return torch.tensor(verts, dtype=torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_obj_path', required=True, help='mesh the indices were minted on')
    ap.add_argument('--dst_obj_path', required=True, help='mesh to map them onto')
    ap.add_argument('--vertex_indices', type=int, nargs='+', required=True)
    args = ap.parse_args()

    src = read_raw_obj_vertices(args.src_obj_path)
    dst = read_raw_obj_vertices(args.dst_obj_path)
    print(f'src {args.src_obj_path}: {len(src)} vertices')
    print(f'dst {args.dst_obj_path}: {len(dst)} vertices')

    bad = [i for i in args.vertex_indices if i >= len(src)]
    if bad:
        raise ValueError(f'indices {bad} are out of range for the source mesh')

    # scale of the DESTINATION mesh's loadmesh frame, for reporting the gap in the units every
    # downstream metric actually uses
    dst_scale = float(Normalizer.get_bounding_sphere_normalizer(dst.float())._scale)

    mapped = []
    print(f'\n{"src idx":>8} {"->":^4} {"dst idx":>8}   {"gap (raw)":>10} {"gap (dst frame)":>16}')
    for i in args.vertex_indices:
        d = torch.norm(dst - src[i][None, :], dim=1)
        j = int(torch.argmin(d))
        gap = float(d[j])
        mapped.append(j)
        print(f'{i:>8} {"->":^4} {j:>8}   {gap:>10.6f} {gap / dst_scale:>16.6f}')
    print('\n--query_vertices ' + ' '.join(str(v) for v in mapped))


if __name__ == '__main__':
    main()
