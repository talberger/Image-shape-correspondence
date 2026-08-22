"""Precompute geodesic distance tables for the relational-loss --geodesic experiment
(see PROJECT_NOTES.md, 2026-08-13 entry, for the full design discussion).

Two independent tables, both fixed geometry -- neither depends on any encoder/checkpoint,
computed once and reused across every geodesic training run on this mesh/mask:

  3D (mesh surface): vertex-to-vertex geodesic distance via the heat method
    (potpourri3d.MeshHeatMethodDistanceSolver, the same "heat solver" SHIC (arxiv 2407.18907)
    uses for its own geodesic table). Computed in loadmesh()'s normalized (bounding-sphere)
    space -- the same space relational_loss() already operates in -- by looping
    compute_distance() over all V vertices (V=4036 here, so V calls, each cheap: the heat
    method is a single sparse linear solve per source, not a full Dijkstra).

  2D (pixel space, whole mask): geodesic (mask-constrained shortest path) distance from each
    of the FIXED sample-center pixels to every other on-object pixel, via
    scipy.sparse.csgraph.dijkstra over an 8-connected pixel graph restricted to on-object
    pixels (edge weight = true Euclidean pixel distance -- 1.0 for a 4-neighbor step, sqrt(2)
    for a diagonal step -- so this approximates true geodesic distance through the mask, not
    just hop count). Only the FIXED centers are needed as sources (not all M masked pixels):
    sample_center_pixels's centers are sampled ONCE and reused for the whole training run
    (only the 16 neighbours per center resample every step), so a [num_centers, H, W] table is
    exactly the granularity training needs, computed once regardless of how many steps run.
    Per Tal's explicit call, "must pass through the segment map" means constrained to the whole
    mask (can cross between segments), NOT constrained to stay within one fine segment.

Run:
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python analysis/build_geodesic_tables.py \\
  --object_mask_path ./demo/guitar/bsb_origin/guitar_01_click_130_140_sam_mask_2.pth \\
  --fine_seg_path ./demo/guitar/phase2/guitar_01_fine_segments.pth \\
  --num_center_pixels 2048 --seed 0 \\
  --out_dir ./experiments/guitar/geodesic_tables
"""
import argparse
import os
import sys
import time

import numpy as np
import potpourri3d as pp3d
import scipy.sparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.mapping_loss import sample_center_pixels, sample_grid_center_pixels
from modules.utils import loadmesh


def build_3d_table(obj_path, name, t_coef, undershoot_tol):
    """t_coef default (20.0) was measured against this specific mesh, not potpourri3d's own
    default (1.0): this mesh's edge lengths span min=0.00062 to max=0.32143 (normalized,
    ~500x ratio -- a fine detailed part mixed with much coarser flat regions), which the heat
    method's default time step (calibrated off the MEAN edge length) handles poorly -- measured
    directly from vertex 0, t_coef=1.0 gave 2193/4036 vertices with geodesic < Euclidean (should
    be mathematically impossible), worst case undershooting by 0.154 in a mesh whose own bbox
    diagonal is ~1.89 (>8% relative error). Swept t_coef in {1, 5, 10, 15, 20, 25, 30, 40, 100}:
    15-25 is the good range (worst-case undershoot 0.008-0.013, ~0.5-0.7% relative), degrading
    again above ~30. 20.0 is a reasonable middle of that range, not a tuned-to-death optimum --
    if this mesh's geometry ever changes, re-sweep rather than assuming 20.0 still holds.
    """
    mesh = loadmesh(dir=obj_path, name=name)
    verts = mesh.vertices.detach().cpu().numpy().astype(np.float64)
    faces = mesh.faces.detach().cpu().numpy().astype(np.int32)
    num_verts = verts.shape[0]

    solver = pp3d.MeshHeatMethodDistanceSolver(verts, faces, t_coef=t_coef)
    table = np.empty((num_verts, num_verts), dtype=np.float32)
    t0 = time.time()
    for v in range(num_verts):
        table[v] = solver.compute_distance(v).astype(np.float32)
        if v % 500 == 0:
            print(f'  3D: {v}/{num_verts} source vertices done ({time.time() - t0:.1f}s so far)')
    print(f'3D geodesic table done in {time.time() - t0:.1f}s (t_coef={t_coef})')

    # Geodesic distance can never be LESS than Euclidean distance, mathematically -- but the heat
    # method is an APPROXIMATION (diffusion-based, not exact), and measured directly against this
    # mesh's full 4036x4036 table: undershoot happens on ~1% of all pairs (concentrated in a
    # small, much finer-triangulated region -- this mesh's edge lengths span a ~500x ratio, see
    # this function's docstring), mean undershoot 0.011, p99 0.021, worst case 0.066 (~3.5% of
    # the mesh's own ~1.89 bbox diagonal) -- no --t_coef in a sweep of {1..100} eliminated this
    # everywhere at once (different mesh regions want different diffusion times). Since the true
    # lower bound is known exactly (Euclidean distance, computed directly, no approximation),
    # CLAMPING is the principled fix: nudge only the erroneous entries up to their proven minimum
    # possible value, changing nothing elsewhere. `undershoot_tol` is now a safety ceiling on the
    # PRE-CLAMP violation, not a per-entry pass/fail -- anything beyond it (well above the ~0.066
    # observed here) would mean something is actually broken (bad --t_coef, bad mesh), not
    # ordinary approximation noise, and still raises loudly instead of silently clamping it away.
    diff2 = ((verts[:, None, :] - verts[None, :, :]) ** 2).sum(-1)
    euclidean = np.sqrt(diff2).astype(np.float32)
    undershoot = euclidean - table
    n_violations = int((undershoot > 0).sum())
    worst = float(undershoot.max())
    if worst > undershoot_tol:
        raise ValueError(f'3D geodesic table undershoots Euclidean by {worst:.5f}, beyond the '
                         f'{undershoot_tol} safety ceiling -- this is larger than ordinary '
                         f'heat-method approximation error, treat as a real bug (try a different '
                         f'--t_coef, or check the mesh geometry) rather than clamping it away')
    print(f'{n_violations}/{num_verts * num_verts} entries undershoot Euclidean (mean '
          f'{float(undershoot[undershoot > 0].mean()):.5f}, worst {worst:.5f}, within the '
          f'{undershoot_tol} safety ceiling) -- clamping those up to the exact Euclidean lower '
          f'bound, since it is known exactly and geodesic can never be less than it')
    table = np.maximum(table, euclidean)

    diagonal = float(table.max())
    return torch.from_numpy(table), diagonal


def build_2d_table(coords, coarse_mask, chunk_size=2000):
    """coords: int64 [n, 2] (row, col) of the fixed sample centers.
    coarse_mask: bool ndarray [H, W].
    Returns (table [n, H, W] float32, diagonal float, geodesic_safe_mask bool [H, W]).

    First real bug found while building this: `coarse_mask` has 3 connected components
    (checked directly, scipy.ndimage.label, 8-connectivity) -- the real object (205778 px) plus
    two tiny SAM-segmentation specks (4px and 1px). `sample_center_pixels` already excludes
    these from being a CENTER (it requires `fine_segments > 0`) -- but checked directly, these
    specks DO carry a real, non-background fine_segment id (16, here) despite being spatially
    disconnected from the rest of that segment, so they remain fully eligible NEIGHBOUR
    candidates via both `sample_in_segment_neighbors` (draws from `build_segment_index`'s
    per-segment pixel lists, built from `fine_segments == s` alone, blind to any mask at all)
    and `sample_out_segment_neighbors` (draws from `mask_coords`, built from the raw
    `coarse_mask`). For Euclidean training this is harmless (still just some finite, if
    meaningless, distance) -- for geodesic it is not: a speck is graph-disconnected from the
    main component, so its distance from every real center is genuinely infinite. Measured:
    ~0.0024% of all mask pixels are specks, which over a 300-step run's ~4.9M neighbour draws
    would hit repeatedly (expected ~100+ times), not a one-in-a-billion edge case safe to ignore.

    Fixed by restricting the whole geodesic pipeline to the mask's LARGEST connected component
    only: this table is built over it directly, and the caller (phase2/train_relational_loss.py)
    additionally zeroes out fine_segments at every dropped pixel before building its own
    seg_idx, since build_segment_index's per-segment lists need that -- passing a restricted
    mask alone is not enough (see that caller for the fuller explanation, found via a real
    smoke-test failure, not anticipated up front). geodesic_safe_mask is returned so the caller
    persists and reuses the identical restricted mask rather than recomputing it (which risks a
    silent mismatch between what this table was built against and what training later samples
    from).
    """
    import scipy.ndimage

    labeled, num_components = scipy.ndimage.label(coarse_mask, structure=np.ones((3, 3), dtype=int))
    if num_components > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # background label, not a real component
        largest_label = int(sizes.argmax())
        geodesic_safe_mask = labeled == largest_label
        dropped = int(coarse_mask.sum() - geodesic_safe_mask.sum())
        print(f'coarse_mask has {num_components} connected components; keeping only the largest '
              f'({int(geodesic_safe_mask.sum())} px), dropping {dropped} px across '
              f'{num_components - 1} disconnected speck(s) -- see build_2d_table\'s docstring')
    else:
        geodesic_safe_mask = coarse_mask.copy()

    H, W = geodesic_safe_mask.shape
    mask_rows, mask_cols = np.nonzero(geodesic_safe_mask)
    num_mask_pixels = mask_rows.shape[0]
    print(f'2D: {num_mask_pixels} on-object pixels, {coords.shape[0]} fixed centers')

    # Map every on-object pixel to a compact graph-node id; background pixels get -1 (never
    # used as a node, since the graph only connects on-object pixels to each other).
    pixel_to_node = -np.ones((H, W), dtype=np.int64)
    pixel_to_node[mask_rows, mask_cols] = np.arange(num_mask_pixels)

    # 8-connected graph, edge weight = true pixel distance (1.0 orthogonal, sqrt(2) diagonal) --
    # NOT just hop count, so shortest-path length approximates true geodesic distance through
    # the mask rather than a coarser grid-step count.
    offsets = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]
    rows_list, cols_list, weights_list = [], [], []
    for dr, dc, w in offsets:
        nbr_rows = mask_rows + dr
        nbr_cols = mask_cols + dc
        valid = (nbr_rows >= 0) & (nbr_rows < H) & (nbr_cols >= 0) & (nbr_cols < W)
        valid_src = np.nonzero(valid)[0]
        nbr_rows_v = nbr_rows[valid]
        nbr_cols_v = nbr_cols[valid]
        nbr_node = pixel_to_node[nbr_rows_v, nbr_cols_v]
        on_mask = nbr_node >= 0
        rows_list.append(valid_src[on_mask])
        cols_list.append(nbr_node[on_mask])
        weights_list.append(np.full(on_mask.sum(), w, dtype=np.float64))

    graph_rows = np.concatenate(rows_list)
    graph_cols = np.concatenate(cols_list)
    graph_weights = np.concatenate(weights_list)
    graph = scipy.sparse.csr_matrix((graph_weights, (graph_rows, graph_cols)),
                                    shape=(num_mask_pixels, num_mask_pixels))

    center_nodes = pixel_to_node[coords[:, 0].numpy(), coords[:, 1].numpy()]
    if (center_nodes < 0).any():
        raise ValueError('some sample centers are not on-object pixels in coarse_mask -- '
                         'coords/coarse_mask mismatch')

    # Compact [num_centers, num_mask_pixels] storage -- exactly what dijkstra already returns,
    # not expanded to dense [num_centers, H, W]. At the real scale (2048 centers, 205778 masked
    # pixels of 1080x810=874800 total) dense storage would be ~4.25x bigger for no reason (7.2GB
    # vs 1.7GB) -- every pixel training ever looks up is a masked one by construction (centers
    # and neighbours are both always sampled from the mask), so the background/off-mask cells a
    # dense array would carry are pure waste. `pixel_to_node` (the [H, W] -> compact-index map,
    # int64, ~7MB) is what a caller needs to convert a (row, col) into a column of this table --
    # saved alongside it.
    #
    # CHUNKED (2026-08-18, --center_mode grid): scipy's dijkstra always returns float64,
    # regardless of the graph's own dtype -- calling it with all num_centers indices at once
    # (fine at the original 2048-center scale, ~1.7GB float32 final size) needs BOTH the full
    # float64 result (2x the final float32 size) and the float32 copy in memory simultaneously
    # to downcast. At candidate-pool scale (tens of thousands of centers) that peak silently
    # exceeded this container's 58GB cgroup memory limit (not the host's, which is much larger
    # and easy to check instead by mistake) and got OOM-killed with no traceback, after the
    # expensive dijkstra pass had already finished (~35 min at 23521 centers, wasted). Fixed by
    # requesting only chunk_size centers' worth of float64 rows at a time, downcasting each
    # chunk into a preallocated float32 array immediately, and discarding the float64 chunk --
    # peak transient memory becomes ~chunk_size*num_mask_pixels*8 bytes instead of
    # num_centers*num_mask_pixels*8 bytes, independent of how large num_centers gets.
    num_centers = len(center_nodes)
    table = np.empty((num_centers, num_mask_pixels), dtype=np.float32)
    diagonal = 0.0
    t0 = time.time()
    for start in range(0, num_centers, chunk_size):
        end = min(start + chunk_size, num_centers)
        chunk_dist = scipy.sparse.csgraph.dijkstra(
            graph, indices=center_nodes[start:end], directed=False)

        chunk_finite = np.isfinite(chunk_dist)
        if not chunk_finite.all():
            n_inf = int((~chunk_finite).sum())
            raise ValueError(f'{n_inf}/{chunk_dist.size} (center, mask-pixel) pairs in centers '
                             f'[{start},{end}) are unreachable even after restricting to the '
                             f'largest connected component -- this should be impossible by '
                             f'construction (every pixel in that component is reachable from '
                             f'every other), so something other than the known speck issue is '
                             f'going on -- do not silently proceed with inf distances')

        diagonal = max(diagonal, float(chunk_dist.max()))
        table[start:end] = chunk_dist
        del chunk_dist
        print(f'  2D: dijkstra centers {start}-{end}/{num_centers} done '
              f'({time.time() - t0:.1f}s so far)')

    print(f'2D: dijkstra from {num_centers} centers done in {time.time() - t0:.1f}s '
          f'(chunk_size={chunk_size})')

    return torch.from_numpy(table), diagonal, geodesic_safe_mask, torch.from_numpy(pixel_to_node)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    if args.skip_3d:
        print('=== 3D geodesic table: SKIPPED (--skip_3d) -- unaffected by --center_mode, '
              'reusing whatever is already at the standard path ===')
    else:
        print('=== 3D geodesic table (heat method) ===')
        table_3d, diag_3d = build_3d_table(args.obj_path, args.name, args.t_coef, args.undershoot_tol)
        path_3d = os.path.join(args.out_dir, f'{args.name}_geo_table_3d.pth')
        torch.save({'table': table_3d, 'diagonal': diag_3d, 'obj_path': args.obj_path}, path_3d)
        print(f'saved 3D table {tuple(table_3d.shape)} (diagonal={diag_3d:.5f}) to {path_3d}')

    print('=== 2D geodesic table (dijkstra, whole mask) ===')
    coarse_mask = torch.load(args.object_mask_path, map_location='cpu').numpy()
    fine_segments = torch.load(args.fine_seg_path, map_location='cpu').numpy()

    if args.center_mode == 'grid':
        # CANDIDATE POOL, not a fixed center set -- built for --resample_centers
        # (phase2/train_relational_loss.py, 2026-08-18): every stride-spaced eligible
        # pixel becomes a table row, so training can draw a fresh random subset of
        # centers from this pool every step instead of using the same num_center_pixels
        # forever. See sample_grid_center_pixels's docstring.
        coords, _, segment_ids = sample_grid_center_pixels(
            coarse_mask, fine_segments, stride=args.grid_stride)
        print(f'  --center_mode grid: {coords.shape[0]} candidate centers at stride={args.grid_stride} '
              f'(vs. sample_center_pixels\'s usual {args.num_center_pixels} fixed ones)')
        suffix = f'_gridstride{args.grid_stride}'
    else:
        coords, _, segment_ids = sample_center_pixels(
            coarse_mask, fine_segments, args.num_center_pixels, seed=args.seed)
        suffix = f'_seed{args.seed}'

    table_2d, diag_2d, geodesic_safe_mask, pixel_to_node = build_2d_table(
        coords, coarse_mask, chunk_size=args.chunk_size)
    path_2d = os.path.join(
        args.out_dir,
        f'{os.path.splitext(os.path.basename(args.object_mask_path))[0]}_geo_table_2d{suffix}.pth')
    torch.save({
        # table: [num_centers, num_mask_pixels] float32, compact (dijkstra's own output shape,
        # not expanded to dense [num_centers, H, W] -- see build_2d_table's docstring for why).
        # A caller looks up (center_i, row, col) as table[center_i, pixel_to_node[row, col]].
        'table': table_2d, 'pixel_to_node': pixel_to_node, 'diagonal': diag_2d, 'coords': coords,
        # geodesic_safe_mask (largest connected component of object_mask_path, specks dropped --
        # see build_2d_table's docstring): geodesic training MUST use this, not the raw
        # coarse_mask, for sample_out_segment_neighbors's candidate pool, or a disconnected
        # speck pixel can get drawn as a "neighbour" and produce inf.
        'geodesic_safe_mask': torch.from_numpy(geodesic_safe_mask),
        'object_mask_path': args.object_mask_path,
        'center_mode': args.center_mode,
        # num_center_pixels/seed kept for the 'random' (existing, exact-match) mode's own
        # load_geodesic_table_2d validation; grid_stride for the 'grid' (pool) mode's.
        'num_center_pixels': args.num_center_pixels, 'seed': args.seed,
        'grid_stride': args.grid_stride if args.center_mode == 'grid' else None,
    }, path_2d)
    print(f'saved 2D table {tuple(table_2d.shape)} (diagonal={diag_2d:.5f}) to {path_2d}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')
    parser.add_argument('--t_coef', type=float, default=20.0,
                        help='heat method diffusion time coefficient -- see build_3d_table\'s '
                             'docstring for why this mesh needs a higher value than potpourri3d\'s '
                             'own default (1.0)')
    parser.add_argument('--undershoot_tol', type=float, default=0.15,
                        help='safety ceiling on (Euclidean - geodesic) before raising instead of '
                             'clamping -- measured worst case on this mesh at t_coef=20 is 0.066 '
                             '(mesh bbox diagonal ~1.89), so 0.15 is a generous margin above '
                             'ordinary heat-method approximation error, not a per-entry tolerance')
    parser.add_argument('--object_mask_path', type=str, required=True)
    parser.add_argument('--fine_seg_path', type=str,
                        default='./demo/guitar/phase2/guitar_01_fine_segments.pth')
    parser.add_argument('--center_mode', type=str, choices=['random', 'grid'], default='random',
                        help="random (default): sample_center_pixels's usual fixed "
                             '--num_center_pixels random centers, exact-match table -- byte-'
                             'identical to before this flag existed. grid: sample_grid_center_'
                             'pixels\'s deterministic --grid_stride-spaced CANDIDATE POOL '
                             '(typically far more rows), for --resample_centers training.')
    parser.add_argument('--grid_stride', type=int, default=3,
                        help='--center_mode grid only. Pixel spacing between candidate centers '
                             '(e.g. 3 -> ~1/9th of on-object pixels are eligible candidates). '
                             'Must match the training run\'s --center_grid_stride.')
    parser.add_argument('--skip_3d', action='store_true',
                        help='skip rebuilding the 3D (vertex-to-vertex) table -- it does not '
                             'depend on --center_mode/pixel centers at all, so re-running it is '
                             'wasted time once a valid one already exists at the standard path.')
    parser.add_argument('--num_center_pixels', type=int, default=2048,
                        help='--center_mode random only: must match --num_center_pixels of the '
                             'training run this table is for')
    parser.add_argument('--seed', type=int, default=0,
                        help='--center_mode random only: must match --seed of the training run '
                             'this table is for (sample_center_pixels\'s centers must be identical)')
    parser.add_argument('--chunk_size', type=int, default=2000,
                        help='dijkstra centers processed per batch before downcasting to float32 '
                             'and discarding the float64 result -- bounds peak memory to roughly '
                             'chunk_size*num_mask_pixels*8 bytes instead of needing the full '
                             'float64 AND float32 tables in memory at once (see build_2d_table\'s '
                             'docstring: this OOM-killed a --center_mode grid build at the '
                             'container\'s 58GB cgroup limit before this flag existed).')
    parser.add_argument('--out_dir', type=str, required=True)

    args = parser.parse_args()
    main(args)
