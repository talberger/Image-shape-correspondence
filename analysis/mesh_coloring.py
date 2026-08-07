"""
mesh coloring: color every vertex of the mesh from the mesh alone -- no image, no
mask, no correspondence involved. This is the mesh-side counterpart of
pixel_coloring.py, and the opposite direction from color_mesh_from_knn.py (which
paints the mesh with colors that came from the *photo*, through the learned
pixel->vertex correspondence). Use these colors as a *source* signal: paint the
mesh here, then push the colors through a correspondence to see where they land.

--mode position (default): xyz -> RGB, per-axis min-max to [0, 1] over the mesh's
own bounding box. This is DPC's create_colormap verbatim
(github.com/dvirginz/DPC/blob/main/visualization/mesh_visualization_utils.py),
the function pixel_coloring.py's `position` mode was adapted *from* -- there the
third channel had to be faked (B fixed at 128) because pixels have only 2
coordinates; here all three channels carry real geometry.

--mode pca: per-vertex features [V, F] -> PCA(3) -> RGB, exactly the reduction
pixel_coloring.py --mode pca does on per-pixel DINO features (Amir et al.,
"Deep ViT Features as Dense Visual Descriptors", arXiv 2112.05814): PCA fit on
the raw features, then independent per-channel min-max to [0, 255].

--mode checker: a colored checkerboard wrapped onto the surface. xyz is quantized
into a grid of *cubic* cells and each cell gets one flat color, with the
guarantee that no two face-adjacent cells share a color -- so every cell boundary
is a visible seam. Unlike `position`, which is smooth by construction and
therefore hides small errors in whatever consumes it, this one turns a coherent
map into large contiguous patches and an incoherent one into speckle. (Same
reasoning as pixel_coloring.py's `grid` palette, in 3D.)

The colors follow a linear rule, (i - j + 3k) mod 8, so cells along a diagonal
share a color and the whole thing repeats like a tablecloth -- matching the
regular look of the textured template in SHIC (arXiv 2407.18907, Fig. 6: "we map
a textured mesh over the image using the predicted dense correspondences"). SHIC
gets it from a checkerboard *texture* on a canonical UV atlas; our meshes have no
UV parameterization at all (meshes/guitar.obj has zero `vt` entries and
modules/mesh.py leaves face_uvs None), and auto-unwrapping would put chart seams
and stretch into the pattern -- contaminating exactly the regularity the
checkerboard exists to show. Quantizing 3D space needs no atlas.

Run (position, the default):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/mesh_coloring.py \
  --obj_path ./meshes/guitar.obj

Run (checkerboard, coarser cells):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/mesh_coloring.py \
  --obj_path ./meshes/guitar.obj --mode checker --grid 8

Run (PCA of per-vertex features -- --vertex_features_path is mandatory, there is
deliberately no default: the analysis pipeline has more than one pred_f.pth
(BSB authors' demo features vs our phase-1 retrain vs phase-2 checkpoints) and
they are not interchangeable):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/mesh_coloring.py \
  --obj_path ./meshes/guitar.obj --mode pca \
  --vertex_features_path ./experiments/guitar/encoder_dino/pred_f.pth

Each run writes both, into --save_dir (default ./demo/guitar/colored_mesh/,
created if missing -- this coloring belongs to neither the mask-driven pipeline
in demo/guitar/bsb/ nor to phase-2 in demo/guitar/phase2/):
  <obj_name>_mesh_<mode>_color.ply  -- viewable in MeshLab / render_colored_mesh.py
  <obj_name>_mesh_<mode>_color.pth  -- {'vertex_colors': uint8 [V, 3], ...} for
                                       code that needs the colors, not a mesh

To render one (slow, >120s of kaolin JIT warm-up -- run it in the background;
note render_colored_mesh.py still defaults --save_dir to demo/guitar/bsb/, since
that default is shared with the older pipeline, so pass it explicitly):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/render_colored_mesh.py \
  --ply_path ./demo/guitar/colored_mesh/guitar_mesh_checker12_color.ply \
  --save_dir ./demo/guitar/colored_mesh/ --render_res 512
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pixel_coloring import pca_colors
from modules.mesh import Mesh
from modules.utils import write_ply


def load_vertex_features(vertex_features_path, num_vertices):
    features = torch.load(vertex_features_path, map_location='cpu')
    if features.dtype != torch.float32:
        raise ValueError(f'expected float32 vertex features, got {features.dtype}')
    if features.dim() != 2:
        raise ValueError(f'expected vertex features of shape [V, F], got {tuple(features.shape)}')
    if features.shape[0] != num_vertices:
        raise ValueError(f'vertex features have {features.shape[0]} rows, '
                          f'but the mesh has {num_vertices} vertices')
    return features.detach()


def position_colors(verts):
    """DPC's create_colormap, unchanged: per-axis min-max of xyz to [0, 1].

    Their code divides by (max - min) with no guard; an axis of zero extent
    (a perfectly flat mesh) would give inf/nan there. We raise instead, since a
    silent nan would be written into the .ply as garbage.
    """
    verts = np.double(verts)
    min_xyz = verts.min(axis=0)
    max_xyz = verts.max(axis=0)
    extent = max_xyz - min_xyz
    if np.any(extent < 1e-12):
        raise ValueError(f'mesh bounding box is degenerate along some axis (extent={extent}); '
                          'xyz -> RGB is undefined there')

    colors = (verts - min_xyz) / extent
    return (colors * 255).astype(np.uint8)


# Sampled from the reference checkerboard-on-animals figure: muted, roughly
# equal in lightness, so no cell reads as "the important one" and the seams stay
# visible from any angle.
#
# Chosen by maximizing the *minimum* CIELAB distance between any two entries
# (iterated maximin over the sRGB cube in steps of 4, restricted to L 40-78 and
# chroma 30-65 to stay in figure-palette territory rather than neon). The colors
# first sampled off SHIC's figure had two blues, a teal and two greens among
# eight, i.e. a minimum separation of only 26.3 -- neighbouring patches read as
# shades of each other. This set's minimum separation is 47.6.
#
# The *order* is load-bearing too, not just the set: under the diagonal rule
# below, two spatially adjacent cells always differ by one step in this list, so
# consecutive entries are the ones that end up touching on the mesh. This is the
# cyclic permutation maximizing the minimum distance between consecutive entries
# (87.3). Re-run both optimizations if the palette ever changes --
# scratchpad/palette_search.py in the session notes, or just redo the two loops:
# maximin over candidates, then brute force over the 7! cyclic orders.
CHECKER_PALETTE = [
    (0, 112, 4),       # green
    (0, 108, 216),     # blue
    (152, 136, 76),    # khaki
    (164, 44, 140),    # magenta
    (32, 192, 236),    # cyan
    (164, 64, 0),      # rust
    (0, 216, 164),     # spring green
    (252, 136, 148),   # salmon
]

CHECKER_NEIGHBOR_OFFSETS = [(-1, 0, 0), (1, 0, 0), (0, -1, 0),
                            (0, 1, 0), (0, 0, -1), (0, 0, 1)]


def checker_grid(verts, grid):
    """The cell lattice: *cubic* cells, one cell size for all three axes derived
    from the longest side of the bounding box, so the patches come out square on
    the surface instead of stretched along the mesh's long axis. `grid` is
    therefore the number of cells across that longest axis.

    Returns (min_xyz, cell_size, max_cell) so vertices and face centroids can be
    quantized against the same lattice."""
    verts = np.asarray(verts, dtype=np.float64)
    min_xyz = verts.min(axis=0)
    extent = verts.max(axis=0) - min_xyz
    cell_size = extent.max() / grid
    if cell_size < 1e-12:
        raise ValueError('mesh bounding box has zero extent; cannot build a checkerboard')

    max_cell = np.floor(extent / cell_size).astype(np.int64)
    return min_xyz, cell_size, max_cell


def checker_cell_index(points, min_xyz, cell_size, max_cell):
    cell = np.floor((np.asarray(points, dtype=np.float64) - min_xyz) / cell_size).astype(np.int64)
    # Points exactly on the max edge land in cell `grid`; fold them back.
    return np.clip(cell, 0, max_cell)


# color index = (i - j + 3k) mod len(palette). Every coefficient is nonzero mod 8,
# which is exactly the condition for two face-adjacent cells never to share a
# color; and because the i and j coefficients are +1/-1, cells along a diagonal
# (i+1, j+1) land on the *same* color. That is what gives the regular,
# tablecloth-like repeat SHIC's textured template shows, rather than a scatter.
# The k coefficient (3, coprime with 8) keeps the diagonal bands from repeating
# too soon through the mesh's depth.
CHECKER_DIAGONAL_COEFFS = (1, -1, 3)


def diagonal_cell_colors(cell_ids):
    """One palette color per cell from a linear rule -- constant along diagonals."""
    num_colors = len(CHECKER_PALETTE)
    a, b, c = CHECKER_DIAGONAL_COEFFS
    return {cell: int((a * cell[0] + b * cell[1] + c * cell[2]) % num_colors)
            for cell in cell_ids}


def assign_cell_colors(cell_ids, seed):
    """One palette color per occupied cell, such that no two face-adjacent cells
    share a color.

    Greedy graph coloring in raster order: each cell takes the first color, in
    its own seeded random order, that none of its already-colored 6-neighbors
    used. With 8 colors and at most 6 neighbors this can never fail. Gives an
    irregular patchwork -- no repeating structure at all, which is the opposite
    of what diagonal_cell_colors is for.
    """
    rng = np.random.RandomState(seed)
    num_colors = len(CHECKER_PALETTE)
    cell_color = {}

    for cell in sorted(cell_ids):
        taken = set()
        for di, dj, dk in CHECKER_NEIGHBOR_OFFSETS:
            neighbor = (cell[0] + di, cell[1] + dj, cell[2] + dk)
            if neighbor in cell_color:
                taken.add(cell_color[neighbor])
        for candidate in rng.permutation(num_colors):
            if candidate not in taken:
                cell_color[cell] = int(candidate)
                break
        else:  # unreachable with 8 colors and 6 neighbors; loud if it ever isn't
            raise RuntimeError(f'ran out of palette colors at cell {cell}')

    return cell_color


def _clip_polygon_below(poly, axis, plane):
    """Sutherland-Hodgman: returns (below, above), the parts of a convex polygon
    on each side of the axis-aligned plane. Either may be empty."""
    below, above = [], []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        cur_in, nxt_in = cur[axis] <= plane, nxt[axis] <= plane
        if cur_in:
            below.append(cur)
        else:
            above.append(cur)
        if cur_in != nxt_in:
            t = (plane - cur[axis]) / (nxt[axis] - cur[axis])
            crossing = cur + t * (nxt - cur)
            crossing[axis] = plane  # exact, so the two pieces share the edge
            below.append(crossing)
            above.append(crossing)
    return below, above


def split_faces_on_cells(verts, faces, min_xyz, cell_size, min_area=1e-12):
    """Cut every triangle along the cell-boundary planes it crosses, so that no
    output triangle spans two cells.

    Without this, a face is colored by the cell its *centroid* happens to fall
    in, so a boundary between two cells can only follow existing triangle edges:
    it comes out as a sawtooth of whole triangles poking into the neighboring
    patch (visible as ragged/bleeding edges once the patches are flat-colored).
    Clipping makes each boundary the actual plane -- a straight line -- at the
    cost of some extra triangles.

    Returns (new_verts [3T, 3], new_faces [T, 3]) with vertices already split per
    face, ready to carry one flat color each.
    """
    tris = [np.asarray(t, dtype=np.float64) for t in np.asarray(verts)[faces]]

    for axis in range(3):
        pieces = []
        for poly in tris:
            coords = np.array([p[axis] for p in poly])
            k_lo = int(np.floor((coords.min() - min_xyz[axis]) / cell_size))
            k_hi = int(np.floor((coords.max() - min_xyz[axis]) / cell_size))
            remainder = list(poly)
            for k in range(k_lo + 1, k_hi + 1):
                if len(remainder) < 3:
                    break
                plane = min_xyz[axis] + k * cell_size
                below, remainder = _clip_polygon_below(remainder, axis, plane)
                if len(below) >= 3:
                    pieces.append(below)
            if len(remainder) >= 3:
                pieces.append(remainder)
        tris = pieces

    out_tris = []
    for poly in tris:
        poly = np.asarray(poly, dtype=np.float64)
        for i in range(1, len(poly) - 1):  # fan triangulation of a convex piece
            tri = np.stack([poly[0], poly[i], poly[i + 1]])
            area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            if area > min_area:  # drop slivers created exactly on a plane
                out_tris.append(tri)

    new_verts = np.concatenate(out_tris, axis=0).astype(np.float32)
    new_faces = np.arange(new_verts.shape[0], dtype=np.int64).reshape(-1, 3)
    return new_verts, new_faces


def checker_colors(verts, faces, grid, pattern, seed, split_faces):
    """Returns (vertex_colors [V, 3], face_colors [F, 3], split), all colors
    uint8. `split` is None unless `split_faces`, in which case it is
    (split_verts, split_faces, split_colors): the clipped display geometry with
    one flat color per triangle.

    A face's color comes from the cell its *centroid* falls in, and every set of
    colors comes from one shared cell->color assignment, so the flat, split and
    smooth versions of the same run agree wherever they can.
    """
    min_xyz, cell_size, max_cell = checker_grid(verts, grid)

    def cells_of(points):
        cell = checker_cell_index(points, min_xyz, cell_size, max_cell)
        return [tuple(int(c) for c in row) for row in cell]

    vertex_ids = cells_of(verts)
    face_ids = cells_of(np.asarray(verts)[faces].mean(axis=1))

    split_verts = split_face_idx = split_ids = None
    if split_faces:
        split_verts, split_face_idx = split_faces_on_cells(verts, faces, min_xyz, cell_size)
        split_ids = cells_of(split_verts[split_face_idx].mean(axis=1))

    # Every cell that any of the three sets touches has to be colored before the
    # greedy `patchwork` assignment runs, or a clipped sliver could land in a
    # cell nothing else occupies and have no color.
    occupied = set(vertex_ids) | set(face_ids) | set(split_ids or [])
    cell_color = (diagonal_cell_colors(occupied) if pattern == 'diagonal'
                  else assign_cell_colors(occupied, seed))

    palette = np.array(CHECKER_PALETTE, dtype=np.uint8)
    paint = lambda ids: palette[np.array([cell_color[c] for c in ids], dtype=np.int64)]

    print(f'checkerboard ({pattern}): {len(cell_color)} occupied cells across {len(palette)} palette colors, '
          f'{grid} cells along the longest bounding-box axis '
          f'(cells per axis: {max_cell + 1}, cell size {cell_size:.4f})')

    split = None
    if split_faces:
        print(f'cell-boundary clipping: {faces.shape[0]} faces -> {split_face_idx.shape[0]} '
              f'({split_face_idx.shape[0] / faces.shape[0]:.2f}x), each fully inside one cell')
        split = (split_verts, split_face_idx, paint(split_ids))

    return paint(vertex_ids), paint(face_ids), split


def main(args):
    mesh = Mesh(args.obj_path)
    verts = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    num_vertices = verts.shape[0]

    obj_name = os.path.splitext(os.path.basename(args.obj_path))[0]
    if args.mode != 'checker':
        mode_tag = args.mode
    else:
        mode_tag = (f'checker{args.grid}'
                    + ('' if args.checker_pattern == 'diagonal' else f'_{args.checker_pattern}')
                    + ('' if args.checker_shading == 'flat' else '_smooth'))
    output_path = args.output_path or os.path.join(
        args.save_dir, f'{obj_name}_mesh_{mode_tag}_color.ply')
    colors_output_path = args.colors_output_path or os.path.splitext(output_path)[0] + '.pth'
    # This script's default save_dir is its own, and is not created by anything
    # else in the pipeline.
    for path in (output_path, colors_output_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    face_colors = None
    split = None
    if args.mode == 'position':
        vertex_colors = position_colors(verts)
    elif args.mode == 'pca':
        if args.vertex_features_path is None:
            raise ValueError('--mode pca requires --vertex_features_path (no default on purpose: '
                              'demo/, phase-1 and phase-2 pred_f.pth files are not interchangeable)')
        features = load_vertex_features(args.vertex_features_path, num_vertices)
        if num_vertices < args.n_components:
            raise ValueError(f'mesh has only {num_vertices} vertices, '
                              f'fewer than n_components={args.n_components}')
        vertex_colors = pca_colors(features, args.n_components, args.random_state)
    else:
        vertex_colors, face_colors, split = checker_colors(
            verts, faces, args.grid, args.checker_pattern, args.seed,
            split_faces=args.split_faces and args.checker_shading == 'flat')

    print(f'mesh: {num_vertices} vertices, {faces.shape[0]} faces, mode: {args.mode}')

    if split is not None:
        # Clipped geometry: every triangle lies inside a single cell, so the
        # boundaries are the cell planes themselves rather than a sawtooth of
        # whole triangles. Vertices are already one set per triangle.
        split_verts, split_face_idx, split_colors = split
        ply_verts, ply_faces = split_verts, split_face_idx
        ply_colors = np.repeat(split_colors, 3, axis=0)
        print(f'flat shading: .ply written with {ply_verts.shape[0]} split vertices '
              f'(3 per clipped face)')
    elif face_colors is not None and args.checker_shading == 'flat':
        # Per-vertex colors are interpolated across each triangle by every
        # renderer, so a cell boundary that cuts through a triangle comes out as
        # a gradient -- the smear. Giving each face its own three corners, all
        # three carrying that face's single color, makes every triangle flat and
        # every boundary a hard edge. Costs 3x the vertices in the .ply (display
        # geometry only; the .pth keeps colors indexed by the original vertices).
        ply_verts = verts[faces].reshape(-1, 3)
        ply_faces = np.arange(ply_verts.shape[0], dtype=np.int64).reshape(-1, 3)
        ply_colors = np.repeat(face_colors, 3, axis=0)
        print(f'flat shading: .ply written with {ply_verts.shape[0]} split vertices '
              f'(3 per face) so cell boundaries stay hard edges')
    else:
        ply_verts, ply_faces, ply_colors = verts, faces, vertex_colors

    write_ply(output_path, ply_verts, ply_faces, ply_colors.astype(np.int32))
    print(f'saved to {output_path}')

    torch.save({
        # Always indexed by the *original* mesh vertices, whatever the .ply holds,
        # so downstream code can index it with vertex ids from the correspondence.
        'vertex_colors': torch.from_numpy(vertex_colors),  # uint8 [V, 3]
        'face_colors': torch.from_numpy(face_colors) if face_colors is not None else None,  # uint8 [F, 3]
        'mode': args.mode,
        'obj_path': args.obj_path,
        'grid': args.grid if args.mode == 'checker' else None,
        'seed': args.seed if args.mode == 'checker' else None,
        'shading': args.checker_shading if args.mode == 'checker' else None,
        'pattern': args.checker_pattern if args.mode == 'checker' else None,
        'split_faces': split is not None,
        'vertex_features_path': args.vertex_features_path if args.mode == 'pca' else None,
    }, colors_output_path)
    print(f'vertex colors saved to {colors_output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['position', 'pca', 'checker'], default='position',
                         help="position (default): xyz -> RGB, DPC's create_colormap. "
                              'pca: per-vertex features -> PCA(3) -> RGB, as pixel_coloring.py --mode pca. '
                              'checker: colored checkerboard of cubic cells wrapped on the surface.')
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--save_dir', type=str, default='./demo/guitar/colored_mesh/',
                         help='created if missing. Its own directory, not shared with the '
                              'mask-driven pipeline (demo/guitar/bsb/) or with phase-2 '
                              '(demo/guitar/phase2/): this coloring is image-independent and '
                              'belongs to neither.')
    parser.add_argument('--output_path', type=str, default=None,
                         help='defaults to <save_dir>/<obj_name>_mesh_<mode>_color.ply')
    parser.add_argument('--colors_output_path', type=str, default=None,
                         help='defaults to the --output_path with a .pth extension; holds '
                              "{'vertex_colors': uint8 [V, 3], ...} for code that wants the colors "
                              'without parsing a .ply')
    parser.add_argument('--vertex_features_path', type=str, default=None,
                         help='--mode pca only, and mandatory there: per-vertex features [V, F] '
                              '(a pred_f.pth). No default -- demo/guitar/encoder_dino/pred_f.pth '
                              '(BSB authors), experiments/guitar/encoder_dino/pred_f.pth (phase-1) '
                              'and the phase-2 exports are different features.')
    parser.add_argument('--n_components', type=int, default=3, help='--mode pca only')
    parser.add_argument('--random_state', type=int, default=0, help='--mode pca only')
    parser.add_argument('--grid', type=int, default=16,
                         help='--mode checker only. Number of cells along the longest side of the '
                              'bounding box; cells are cubes of that size on all axes. Larger = '
                              'smaller squares. Keep cells comfortably above the mesh edge length: '
                              'once a cell is only a few triangles wide the patches stop reading as '
                              'squares whatever the shading.')
    parser.add_argument('--checker_pattern', type=str, choices=['diagonal', 'patchwork'],
                         default='diagonal',
                         help='--mode checker only. diagonal (default): color index = '
                              '(i - j + 3k) mod 8, so cells along a diagonal share a color and the '
                              'pattern repeats like a tablecloth -- the regular look of SHIC\'s '
                              'textured template. patchwork: greedy random coloring, no repeating '
                              'structure (writes to <...>_checker<grid>_patchwork_color.ply). Both '
                              'guarantee that face-adjacent cells differ.')
    parser.add_argument('--checker_shading', type=str, choices=['flat', 'smooth'], default='flat',
                         help='--mode checker only. flat (default): one color per face, written as '
                              'split vertices (3 per face) so cell boundaries are hard edges, the '
                              'way a UV checkerboard texture would look. smooth: one color per '
                              'vertex, which every renderer interpolates across triangles -- the '
                              'boundaries come out smeared. Both are saved in the .pth regardless; '
                              'smooth writes to <...>_checker<grid>_smooth_color.ply so it never '
                              'clobbers the flat output.')
    parser.add_argument('--no_split_faces', dest='split_faces', action='store_false',
                         help='--mode checker --checker_shading flat only. By default every triangle '
                              'that straddles a cell boundary is clipped along that boundary plane, '
                              'so patch edges are straight lines. Without it a face is colored by '
                              'the cell of its centroid and boundaries can only follow existing '
                              'triangle edges -- a sawtooth of whole triangles poking into the '
                              'neighboring patch. Pass this to keep the original triangles (fewer '
                              'faces, ragged edges).')
    parser.add_argument('--seed', type=int, default=0,
                         help='--mode checker --checker_pattern patchwork only: which palette color '
                              'each cell picks. The adjacency guarantee holds for every seed. '
                              'Unused by the diagonal pattern, which is a closed-form rule.')

    args = parser.parse_args()
    main(args)
