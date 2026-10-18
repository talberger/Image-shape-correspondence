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
cd $DISC_ROOT && python analysis/mesh_coloring.py \
  --obj_path ./meshes/guitar.obj

Run (checkerboard, coarser cells):
cd $DISC_ROOT && python analysis/mesh_coloring.py \
  --obj_path ./meshes/guitar.obj --mode checker --grid 8

Run (PCA of per-vertex features -- --vertex_features_path is mandatory, there is
deliberately no default: the analysis pipeline has more than one pred_f.pth
(BSB authors' demo features vs our phase-1 retrain vs phase-2 checkpoints) and
they are not interchangeable):
cd $DISC_ROOT && python analysis/mesh_coloring.py \
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
cd $DISC_ROOT && python analysis/render_colored_mesh.py \
  --ply_path ./demo/guitar/colored_mesh/guitar_mesh_checker12_color.ply \
  --save_dir ./demo/guitar/colored_mesh/ --render_res 512
"""

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pixel_coloring import pca_colors
from modules.mesh import Mesh
from modules.utils import loadmesh, write_ply


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


# ---------------------------------------------------------------------------
# --checker_coords lbo (2026-09-03, Tal): an INTRINSIC alternative to the xyz
# lattice above.
#
# WHY. The xyz diagonal rule ((i - j + 3k) mod 8) looks clean on a near-planar
# mesh (meshes/guitar.obj) because depth barely varies, so the pattern is
# effectively 2D and lines up with what the eye sees as "diagonal". On a
# voluminous, articulated mesh (demo/shic tested this on SHIC's bear.obj,
# 4936v) it does not: walking around a limb's circular cross-section changes x
# and y quickly over a short physical distance, so the cell index cycles
# several times per limb and the bands SPIRAL around legs/torso instead of
# running as flat diagonal stripes -- confirmed by coarsening the grid
# (6/8/16 all show the same spiraling, so it is not a resolution problem).
#
# FIX: replace raw xyz with the mesh's own low-frequency Laplace-Beltrami
# eigenfunctions as the three coordinates. These are smooth scalar fields
# defined by the mesh's EDGE CONNECTIVITY (solutions of L f = lambda M f for
# the cotangent-weight Laplacian L and the lumped-area mass matrix M), so a
# short walk along the actual surface -- around a limb's circumference, versus
# along its length -- changes them at very different rates by construction:
# low modes vary slowly across a limb's short way round and only pick up
# structure over the mesh's longer geodesic extent. That is exactly the
# "distance along the surface, not straight-line xyz" property the spiral
# diagnosis calls for, at a fraction of the cost of a real geodesic-distance
# field (see analysis/build_geodesic_tables.py's heat-method solve, which this
# deliberately does not reuse: it is built for one V x V table, not a
# smooth low-rank coordinate basis).
#
# COST: sparse partial eigensolve (scipy ARPACK, shift-invert near 0) for the
# 4 smallest eigenpairs (1 trivial + 3 used) -- under 1s on both the guitar
# (3886v) and this bear (4936v). The naive alternative, spharapy's own
# SpharaBasis.basis() (dense scipy.linalg.eigh over ALL V eigenpairs, what
# demo/shic's own weights were likely built with), took 136s on the same bear
# mesh for the same 4 leading eigenvalues -- confirmed matching (7.67 vs 7.68,
# 14.35 vs 14.39): this is the same math, not an approximation, just not
# wasting time on the V-4 eigenpairs nothing here uses.
#
# BINNING: raw eigenfunction values are NOT uniformly distributed (measured on
# the bear: eig3's 30th-70th percentile spans -0.076 to 0.076, i.e. ~40% of
# vertices crowd one narrow band) -- fixed-width cells (checker_grid's
# approach, fine for a roughly-uniform-density mesh in xyz) would dump most of
# the mesh into 1-2 bins. Cells are quantile-binned instead, per coordinate
# independently, so every bin holds a roughly equal share of vertices
# regardless of the eigenfunction's own distribution.
#
# NOT SUPPORTED: --split_faces. split_faces_on_cells clips triangles against
# axis-ALIGNED xyz planes; a per-vertex scalar field has no such plane, and
# clipping against it would mean interpolating (and re-clipping against) all
# three eigenfunction values along every cut edge, not just position. Flat
# per-face coloring works today; the clean split-boundary version is future
# work if this direction is kept (see PROJECT_NOTES.md 2026-09-03).
def compute_lbo_coords(verts, faces, k=3):
    """Returns [V, 3] float64: the `k` smallest-NONTRIVIAL Laplace-Beltrami
    eigenfunctions of the mesh, zero-padded to 3 columns if k < 3 (so the
    diagonal-rule machinery below, which always indexes 3 axes, is unchanged).

    Cotangent-weight Laplacian + barycentric-lumped mass matrix (the standard
    discretization -- Meyer et al. 2003 / Pinkall & Polthier 1993's cotan
    formula); generalized eigenproblem L f = lambda M f solved for the
    smallest `k + 1` eigenpairs via shift-invert ARPACK (sigma near 0), then
    the first (trivial, constant, eigenvalue ~0) eigenvector is dropped.
    """
    if not (1 <= k <= 3):
        raise ValueError(f'--checker_lbo_k must be 1, 2 or 3 (got {k})')
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n = verts.shape[0]

    rows, cols, vals = [], [], []
    area = np.zeros(n)
    for i, j, m in faces:
        vi, vj, vm = verts[i], verts[j], verts[m]

        def cot_angle(a, b, c):  # angle at a, opposite edge b-c
            u, w = b - a, c - a
            cosang = np.dot(u, w)
            sinang = np.linalg.norm(np.cross(u, w))
            return cosang / max(sinang, 1e-12)

        cot_i, cot_j, cot_m = cot_angle(vi, vj, vm), cot_angle(vj, vm, vi), cot_angle(vm, vi, vj)
        for a, b, w in ((j, m, cot_i), (m, i, cot_j), (i, j, cot_m)):
            rows += [a, b]; cols += [b, a]; vals += [0.5 * w, 0.5 * w]
        tri_area = 0.5 * np.linalg.norm(np.cross(vj - vi, vm - vi))
        area[i] += tri_area / 3; area[j] += tri_area / 3; area[m] += tri_area / 3

    W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(W.sum(axis=1)).flatten()) - W
    M = sp.diags(np.maximum(area, 1e-12))

    eigvals, eigvecs = spla.eigsh(L.tocsc(), k=k + 1, M=M.tocsc(), sigma=-1e-6, which='LM')
    order = np.argsort(eigvals)
    coords = eigvecs[:, order[1:k + 1]]  # drop the trivial constant mode (smallest eigenvalue)
    if k < 3:
        coords = np.concatenate([coords, np.zeros((n, 3 - k))], axis=1)
    print(f'LBO coords: k={k}, eigenvalues used = {np.round(eigvals[order[1:k + 1]], 4)} '
          f'(trivial mode {eigvals[order[0]]:.2e} dropped)')
    return coords


def quantile_grid(coords, grid):
    """Per-column quantile bin edges, [grid + 1] each -- the LBO counterpart of
    checker_grid's fixed-width cells (see the module note above for why LBO
    coordinates need quantile rather than linear binning)."""
    edges = []
    for c in range(coords.shape[1]):
        col = coords[:, c]
        if np.allclose(col, col[0]):  # a zero-padded unused axis (k < 3)
            edges.append(np.array([col[0] - 1.0, col[0] + 1.0]))
            continue
        e = np.quantile(col, np.linspace(0, 1, grid + 1))
        e[0] -= 1e-9; e[-1] += 1e-9
        edges.append(e)
    return edges


def quantile_cell_index(points, edges):
    idx = np.zeros(points.shape, dtype=np.int64)
    for c in range(points.shape[1]):
        idx[:, c] = np.clip(np.digitize(points[:, c], edges[c][1:-1]), 0, len(edges[c]) - 2)
    return idx
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# --checker_coords uv (2026-09-03, Tal): the REAL fix for demo/shic's bear.obj, found after
# --checker_coords lbo above turned out to only be a partial one.
#
# THE STORY, kept because this will get reused (Tal: "תזכור איך עשינו את זה"). SHIC's own
# bear.obj/horse.obj/elephant.obj carry ZERO `vt` (UV) entries -- confirmed by grep -- so there is
# no shipped surface parameterization to texture against. But these are DensePose CSE's canonical
# meshes by name and vertex count (bear_4936 etc.), and DensePose's OWN release
# (facebookresearch/detectron2, densepose/data/meshes/builtin.py) publishes real per-vertex UV for
# them at a fixed URL -- see analysis/fetch_densepose_uv.py, which downloads it. Vertex
# correspondence between SHIC's .obj and DensePose's UV was verified before trusting it: indexing
# DensePose's UV by SHIC's own vertex order, >94% of mesh EDGES land within 0.1 UV units of each
# other (median 0.008) -- consistent with a real continuous parameterization, not a coincidence.
# (The matching DensePose GEOMETRY .pkl is a dead end -- unpickles to an all-zero array in this
# environment, correct shape/dtype, no content; documented in fetch_densepose_uv.py so nobody
# re-diagnoses it. Apply the UV directly to SHIC's own geometry instead, which is what this does.)
#
# EVEN WITH REAL UV, A NAIVE PER-FACE-CENTROID CHECKER STILL LOOKED WRONG -- jagged, sawtooth
# boundaries, not the clean squares of the reference figure. The cause was NOT the UV (grid=16 with
# real UV, no clipping, still looked broken) -- it was the same gap that --split_faces already
# exists to close for the xyz mode: coloring a whole triangle by its centroid's cell means any
# triangle straddling a cell boundary paints its ENTIRE area the wrong side's color, and on a UV
# atlas triangles routinely straddle boundaries because chart density is not uniform. The fix is
# `split_faces_on_uv_cells` below -- the direct UV-space counterpart of `split_faces_on_cells`,
# built by reusing `_clip_polygon_below` UNCHANGED (it already indexes generically by axis, so
# clipping a [x, y, z, u, v] point against axis 3 or 4 needed no new clipping math at all).
def load_uv(uv_path, num_vertices):
    """[V, 2] float64 in [0, 1] from a .pkl/.npy/.pth file (a bare array, or a dict with a
    'uv'/'vertices' key -- accepts analysis/fetch_densepose_uv.py's output as-is)."""
    if uv_path.endswith('.npy'):
        uv = np.load(uv_path)
    else:
        import pickle
        with open(uv_path, 'rb') as f:
            try:
                uv = pickle.load(f, encoding='latin1')
            except Exception:
                uv = torch.load(uv_path, map_location='cpu')
    if isinstance(uv, dict):
        uv = uv.get('uv', uv.get('vertices'))
    if torch.is_tensor(uv):
        uv = uv.numpy()
    uv = np.asarray(uv, dtype=np.float64)
    if uv.shape != (num_vertices, 2):
        raise ValueError(f'{uv_path}: expected [{num_vertices}, 2], got {uv.shape} -- wrong mesh, '
                         'or vertex correspondence does not hold (see this module\'s uv note)')
    return uv


def split_faces_on_uv_cells(verts, faces, uv, grid, min_area=1e-14):
    """UV counterpart of split_faces_on_cells: clips every triangle at the UV grid lines k/grid
    (k=1..grid-1) on both axes, so no output triangle straddles a cell boundary. Grid lines are
    fixed at k/grid (UV is already normalized to [0, 1] by convention -- unlike xyz there is no
    mesh-dependent min/cell_size to derive).

    Reuses _clip_polygon_below UNCHANGED against a 5-component [x, y, z, u, v] point per corner, so
    position is carried through every clip for free and the final triangles still have real 3D
    geometry, just cut along UV-space lines.

    Returns (new_verts [3T, 3] float32, new_faces [T, 3], centroid_uv [T, 2]) -- the last one takes
    the place of checker_cell_index's per-piece id lookup, since each clipped triangle already sits
    fully inside one UV cell and its own centroid names that cell exactly.
    """
    tris = [np.concatenate([verts[f], uv[f]], axis=1).astype(np.float64) for f in faces]  # [3,5] each
    for axis in (3, 4):
        pieces = []
        for poly in tris:
            coords = np.array([p[axis] for p in poly])
            k_lo = int(np.floor(coords.min() * grid))
            k_hi = int(np.floor(coords.max() * grid))
            remainder = list(poly)
            for k in range(k_lo + 1, k_hi + 1):
                if len(remainder) < 3:
                    break
                plane = k / grid
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
            pos = tri[:, :3]
            area = 0.5 * np.linalg.norm(np.cross(pos[1] - pos[0], pos[2] - pos[0]))
            if area > min_area:  # drop slivers created exactly on a plane
                out_tris.append(tri)

    out = np.concatenate(out_tris, axis=0)
    new_verts = out[:, :3].astype(np.float32)
    centroid_uv = out[:, 3:5].reshape(-1, 3, 2).mean(axis=1)
    new_faces = np.arange(new_verts.shape[0], dtype=np.int64).reshape(-1, 3)
    return new_verts, new_faces, centroid_uv
# ---------------------------------------------------------------------------


def checker_colors(verts, faces, grid, pattern, seed, split_faces, coord_source='xyz', lbo_k=3, uv=None):
    """Returns (vertex_colors [V, 3], face_colors [F, 3], split), all colors
    uint8. `split` is None unless `split_faces`, in which case it is
    (split_verts, split_faces, split_colors): the clipped display geometry with
    one flat color per triangle.

    A face's color comes from the cell its *centroid* falls in, and every set of
    colors comes from one shared cell->color assignment, so the flat, split and
    smooth versions of the same run agree wherever they can.

    `coord_source`: 'xyz' (default, unchanged original behavior -- fixed-width
    cells over raw vertex position), 'lbo' (see the module note above
    compute_lbo_coords: quantile-binned Laplace-Beltrami eigenfunctions,
    intrinsic to the surface -- fixes the spiral artifact 'xyz' shows on
    voluminous/articulated meshes; `split_faces` is not supported with 'lbo'),
    or 'uv' (see the module note above load_uv/split_faces_on_uv_cells: real
    per-vertex UV, e.g. from analysis/fetch_densepose_uv.py, passed as `uv`
    [V, 2] in [0, 1] -- linear cells, split_faces fully supported and
    strongly recommended, see that note for why the unsplit version looks
    wrong even with correct UV).
    """
    if coord_source == 'lbo':
        if split_faces:
            raise ValueError("--split_faces is not supported with --checker_coords lbo "
                             "(split_faces_on_cells clips against xyz planes only) -- "
                             "pass --no_split_faces, or use --checker_shading smooth")
        coords = compute_lbo_coords(verts, faces, k=lbo_k)
        edges = quantile_grid(coords, grid)

        def cells_of(points_coords):
            cell = quantile_cell_index(points_coords, edges)
            return [tuple(int(c) for c in row) for row in cell]

        vertex_ids = cells_of(coords)
        face_ids = cells_of(coords[faces].mean(axis=1))
        split_verts = split_face_idx = split_ids = None
        max_cell_repr = f'{grid} quantile bins per axis (k={lbo_k})'
    elif coord_source == 'uv':
        if uv is None:
            raise ValueError("--checker_coords uv requires --uv_path (a [V, 2] array in [0, 1])")

        def uv_cell(points_uv):
            i = np.clip((points_uv[:, 0] * grid).astype(np.int64), 0, grid - 1)
            j = np.clip((points_uv[:, 1] * grid).astype(np.int64), 0, grid - 1)
            k = np.zeros_like(i)  # UV is 2D; the diagonal rule's 3rd (depth) axis is unused
            return [tuple(row) for row in np.stack([i, j, k], axis=1).tolist()]

        vertex_ids = uv_cell(uv)
        face_ids = uv_cell(uv[faces].mean(axis=1))
        split_verts = split_face_idx = split_ids = None
        if split_faces:
            split_verts, split_face_idx, centroid_uv = split_faces_on_uv_cells(verts, faces, uv, grid)
            split_ids = uv_cell(centroid_uv)
        max_cell_repr = f'{grid}x{grid} UV cells'
    else:
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
        max_cell_repr = f'cells per axis: {max_cell + 1}, cell size {cell_size:.4f}'

    # Every cell that any of the three sets touches has to be colored before the
    # greedy `patchwork` assignment runs, or a clipped sliver could land in a
    # cell nothing else occupies and have no color.
    occupied = set(vertex_ids) | set(face_ids) | set(split_ids or [])
    cell_color = (diagonal_cell_colors(occupied) if pattern == 'diagonal'
                  else assign_cell_colors(occupied, seed))

    palette = np.array(CHECKER_PALETTE, dtype=np.uint8)
    paint = lambda ids: palette[np.array([cell_color[c] for c in ids], dtype=np.int64)]

    print(f'checkerboard ({pattern}, coords={coord_source}): {len(cell_color)} occupied cells across '
          f'{len(palette)} palette colors, {grid} cells along the longest bounding-box axis ({max_cell_repr})')

    split = None
    if split_faces:
        print(f'cell-boundary clipping: {faces.shape[0]} faces -> {split_face_idx.shape[0]} '
              f'({split_face_idx.shape[0] / faces.shape[0]:.2f}x), each fully inside one cell')
        split = (split_verts, split_face_idx, paint(split_ids))

    return paint(vertex_ids), paint(face_ids), split


def main(args):
    # loadmesh, NOT a bare Mesh(): the bare constructor returns RAW file coordinates and so bypasses
    # both the standard normalization and any `<mesh>.normalizer.json` sidecar, which meant the .ply
    # this writes was in a different frame from everything else built off the same mesh (2026-09-02).
    # A no-op for meshes/guitar.obj, whose normalizer is exactly the identity -- every .ply already
    # on disk is unchanged by this.
    mesh = loadmesh(dir=args.obj_path, name=os.path.splitext(os.path.basename(args.obj_path))[0])
    verts = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    num_vertices = verts.shape[0]

    obj_name = os.path.splitext(os.path.basename(args.obj_path))[0]
    if args.mode != 'checker':
        mode_tag = args.mode
    else:
        coord_tag = ('' if args.checker_coords == 'xyz' else
                    f'_lbo{args.checker_lbo_k}' if args.checker_coords == 'lbo' else '_uv')
        mode_tag = (f'checker{args.grid}' + coord_tag
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
        # --checker_coords lbo never supports the clipped-boundary geometry (see
        # checker_colors' docstring) -- forced off here rather than requiring
        # --no_split_faces to be remembered every time. uv DOES support it (and should
        # basically always use it -- see the uv module note for why the unsplit version
        # looked broken even with correct UV).
        want_split = (args.split_faces and args.checker_shading == 'flat'
                     and args.checker_coords != 'lbo')
        uv = None
        if args.checker_coords == 'uv':
            if not args.uv_path:
                raise ValueError('--checker_coords uv requires --uv_path')
            uv = load_uv(args.uv_path, num_vertices)
        vertex_colors, face_colors, split = checker_colors(
            verts, faces, args.grid, args.checker_pattern, args.seed,
            split_faces=want_split, coord_source=args.checker_coords, lbo_k=args.checker_lbo_k, uv=uv)

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
        'coords': args.checker_coords if args.mode == 'checker' else None,
        'lbo_k': args.checker_lbo_k if (args.mode == 'checker' and args.checker_coords == 'lbo') else None,
        'uv_path': args.uv_path if (args.mode == 'checker' and args.checker_coords == 'uv') else None,
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
    parser.add_argument('--checker_coords', type=str, choices=['xyz', 'lbo', 'uv'], default='xyz',
                         help='--mode checker only. xyz (default): fixed-width cells over raw '
                              'vertex position -- clean diagonal bands on a near-planar mesh '
                              '(meshes/guitar.obj), but SPIRALS around limbs on a voluminous / '
                              'articulated mesh (confirmed on SHIC\'s bear.obj, 4936v). lbo: '
                              'quantile-binned Laplace-Beltrami eigenfunctions instead -- intrinsic '
                              'to the surface, fixes the spiral (see compute_lbo_coords\'s '
                              'docstring for the full reasoning, 2026-09-03). Not compatible with '
                              '--split_faces (forced off automatically). uv: real per-vertex UV '
                              '(--uv_path required, e.g. from analysis/fetch_densepose_uv.py) -- '
                              'the closest match to a real textured-mesh figure; USE WITH '
                              '--split_faces (the default) or boundaries look jagged even with '
                              'correct UV, see split_faces_on_uv_cells\'s docstring.')
    parser.add_argument('--checker_lbo_k', type=int, default=3,
                         help='--checker_coords lbo only. how many Laplace-Beltrami eigenfunctions '
                              '(1-3) to use as coordinates; 3 gives the full diagonal rule, fewer '
                              'zero-pad the unused axis/axes.')
    parser.add_argument('--uv_path', type=str, default=None,
                         help='--checker_coords uv only (required). Path to a [V, 2] UV array in '
                              '[0, 1] -- .pkl (bare array or a dict with a \'uv\' key), .npy, or '
                              '.pth. analysis/fetch_densepose_uv.py produces one for DensePose '
                              'CSE\'s canonical meshes (bear_4936, horse_5004, elephant_5002, ...).')
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
