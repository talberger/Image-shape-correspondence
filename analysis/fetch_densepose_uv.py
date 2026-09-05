"""Download the OFFICIAL per-vertex UV coordinates DensePose CSE publishes for its canonical
animal/human meshes -- the real parameterization used for their own textured-mesh figures
(Neverova et al., "Discovering relationships between object categories via universal canonical
maps", CVPR 2021 / arXiv 2011.12438), not an auto-unwrap approximation.

WHY THIS EXISTS (2026-09-03, comparing against SHIC, arXiv 2407.18907). SHIC's own released
`.obj` files (demo/shic/shapes/*.obj) carry ZERO `vt` entries -- no UV at all, confirmed by grep.
A checkerboard built from raw xyz spirals around limbs on a voluminous mesh (see
compute_lbo_coords's docstring in mesh_coloring.py for the full diagnosis); a Laplace-Beltrami
substitute (--checker_coords lbo) fixes the spiral but still isn't a real checkerboard. The actual
fix: DensePose's own mesh catalog (`densepose/data/meshes/builtin.py` in
facebookresearch/detectron2) publishes real per-vertex UV for these EXACT meshes, keyed by the
same name/vertex-count convention SHIC itself uses (bear_4936, horse_5004, elephant_5002, ...) --
confirmed the SHIC and DensePose bear share vertex correspondence: >94% of mesh EDGES have small
UV distance (median 0.008) when this file's UV is indexed by SHIC's own vertex order, which would
be near-random if the correspondence didn't hold.

TRAP FOUND AND WORKED AROUND: the matching geometry file (`<mesh>.pkl`, holds 'vertices'/'faces')
downloads fine (verified byte-identical on redownload, correct HTTP 200 + content-length) but
UNPICKLES TO ALL-ZERO arrays in this environment -- correct shape, correct dtype, contentless data.
Root cause not chased down (likely a numpy pickle-protocol/version mismatch on Facebook's end, not
a download issue). WORKAROUND: don't use it. Apply the downloaded UV directly onto SHIC's own
`.obj` geometry by vertex INDEX -- verified valid above, and everything downstream normalizes/loads
through modules.utils.loadmesh exactly like every other mesh in this project.

A checkerboard built from this UV STILL needs clipping at cell boundaries in UV space (see
analysis/mesh_coloring.py's split_faces_on_uv_cells) -- coloring each triangle by its UV centroid
alone gives jagged, sawtooth boundaries (the same reason the xyz checker clips in xyz space).

Run:
  python analysis/fetch_densepose_uv.py --mesh_name bear_4936 --save_dir demo/shic/densepose_cse
"""
import argparse
import os
import pickle
import urllib.request

BASE_URL = 'https://dl.fbaipublicfiles.com/densepose/meshes/'

# Every animal/human mesh DensePose CSE ships as of the 2021-03 release (facebookresearch/
# detectron2, densepose/data/meshes/builtin.py). Vertex counts are part of the name by their own
# convention. Not all of these have a matching SHIC weight file (SHIC only ships bear/horse/
# elephant) -- the rest are here for completeness/future reuse.
KNOWN_MESHES = [
    'smpl_27554', 'chimp_5029', 'cat_5001', 'cat_7466', 'sheep_5004', 'zebra_5002',
    'horse_5004', 'giraffe_5002', 'elephant_5002', 'dog_5002', 'dog_7466', 'cow_5002',
    'bear_4936',
]


def fetch(mesh_name, save_dir):
    if mesh_name not in KNOWN_MESHES:
        raise ValueError(f'{mesh_name!r} is not in the known DensePose CSE mesh list: {KNOWN_MESHES}')
    os.makedirs(save_dir, exist_ok=True)

    uv_url = f'{BASE_URL}texcoords/texcoords_{mesh_name}.pkl'
    uv_path = os.path.join(save_dir, f'texcoords_{mesh_name}.pkl')
    if not os.path.isfile(uv_path):
        print(f'downloading {uv_url}')
        urllib.request.urlretrieve(uv_url, uv_path)
    else:
        print(f'already have {uv_path}')

    with open(uv_path, 'rb') as f:
        uv = pickle.load(f, encoding='latin1')
    print(f'UV: shape {uv.shape}, dtype {uv.dtype}, range [{uv.min():.4f}, {uv.max():.4f}]')
    if uv.min() == 0 and uv.max() == 0:
        raise RuntimeError(f'{uv_path} unpickled to an all-zero array -- see this script\'s '
                           'docstring for the known geometry-file version of this trap; report '
                           'if it now also affects texcoords')
    return uv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh_name', required=True, choices=KNOWN_MESHES)
    ap.add_argument('--save_dir', required=True)
    args = ap.parse_args()
    path = fetch(args.mesh_name, args.save_dir)
    print(f'saved {path}')


if __name__ == '__main__':
    main()
