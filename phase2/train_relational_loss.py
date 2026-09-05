"""Phase-2 encoder training: unified relational (distance-proportional) loss.

`train_mapping_loss.py` trains on two separately-weighted signals -- an
attractive mapping_loss and a repulsive hinge repulsion_loss, combined as
`lambda_map * map_loss + lambda_repel * repel_loss` -- which needed a
hand-derived `repel_margin` to reconcile pixel-space and mesh-space distance
scales. This is a separate experiment, starting from the same phase-1
checkpoint, that replaces the split with one continuous loss instead: for a
sample of center pixels and randomly-drawn in-segment/out-of-segment
neighbours (resampled every step), penalize pixel-space distance and
mesh-space distance for not being proportional -- see
modules/relational_loss.py for the full derivation and the Relational
Knowledge Distillation (Park et al., CVPR 2019) formalization it's based on.

train_mapping_loss.py and modules/mapping_loss.py are not modified by this
script -- those runs (encoder_dino_mapping_adjacent, encoder_dino_mapping_repel,
encoder_dino_mapping_repel_wide300) stay exactly reproducible. This script
imports several of train_mapping_loss.py's helpers directly (safe: that file
guards its argument-parsing/training invocation behind `if __name__ ==
'__main__':`) rather than duplicating them.

FIRST RUN (300 steps, 2026-08-10) COLLAPSED -- vertex coverage fell to 0.27%,
worse than mapping_loss's own original collapse. Root cause: normalizing
d_3D by a live, every-step mean is invariant to every pairwise 3D distance
shrinking together in proportion, and nothing stopped the model drifting
there over 300 steps of resampled-every-step neighbours. Two independent
fixes were designed to isolate which change actually matters (2026-08-11,
A/B, see `--live_mu_3d`/`--fixed_neighbors` below and modules/relational_loss.py's
module docstring for the full argument):

  Version 1 (default, both flags off): FIXED mu_3D_reference, measured once
    from the phase-1 encoder (measure_mu_3D_reference below); neighbours
    still resampled every step. The 2026-08-10 end-of-day fix.
  Version 2 (--live_mu_3d --fixed_neighbors): mu_3D back to LIVE (the
    original behavior); neighbours sampled ONCE and reused every step
    instead. Tal's 2026-08-11 hypothesis.

Run (from /workspace/bsb, so `modules.*` and `phase2.*` both resolve):
cd /workspace/bsb && /workspace/miniconda3/envs/bsb/bin/python -m phase2.train_relational_loss \\
    --object_mask_path <coarse mask> --pix_dino_features_path <pixel features> \\
    [--live_mu_3d --fixed_neighbors]   # add for Version 2

See /workspace/PROJECT_NOTES.md for the full design discussion.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from modules.cycle_consistency_loss import cycle_consistency_loss
from modules.cycle_consistency_loss_soft import cycle_consistency_loss_soft
from modules.vertex_balance_loss import vertex_balance_loss
from modules.mapping_loss import (feature_drift, sample_center_pixels,
                                  sample_grid_center_pixels, soft_map_to_vertices)
from modules.feature_rkd_loss import (feature_rkd_angle_loss, feature_rkd_distance_loss,
                                      feature_rkd_pairwise_angle_loss, hard_feature,
                                      measure_feature_mu_references, soft_feature)
from modules.relational_loss import (build_segment_index, relational_loss, relational_loss_geodesic,
                                     sample_in_segment_neighbors, sample_out_segment_neighbors)
from modules.utils import device, loadmesh
from phase2.train_mapping_loss import (assert_resume_is_sane, get_fourier_basis, hash_input,
                                       load_phase1_encoder, set_fourier_basis)


def index_pixel_features(all_features, coords):
    """Per-step equivalent of train_mapping_loss.py's load_pixel_features,
    minus the disk read.

    That function re-reads the 3.6 GB [H, W, 1024] file from disk every call,
    which is fine there -- it's called once, since that script's neighbours
    are fixed for the whole run. Ours are resampled every step, so all_coords
    changes every step; re-reading the file from disk each time would mean
    a full 3.6 GB load + index + free per step, and was in fact the dominant
    cost the first time this script ran (~5.5s/step just in this call,
    measured). Fixed by reading the file once into `all_features` before the
    loop and only indexing it here -- same normalization, same detach, same
    reasoning (see load_pixel_features's docstring), just no repeated I/O.
    """
    features = all_features[coords[:, 0], coords[:, 1]].detach().to(device)
    return F.normalize(features.float(), dim=-1)


def measure_mu_3D_reference(encoder, vertices, coords, segment_ids, seg_idx, all_pixel_features,
                            k_vertices_per_pixel, num_neighbors_in_segment,
                            num_neighbors_out_segment, num_samples=5):
    """The FIXED mu_3D normalizer relational_loss() needs -- measured once,
    here, from the untouched phase-1 encoder, before any relational-loss
    training happens.

    The first version of relational_loss() recomputed mu_3D from the live,
    currently-training encoder's own output every step. That collapsed the
    first time it ran (postmortem in PROJECT_NOTES.md, 2026-08-10): a loss
    that only ever compares distances to its OWN batch's mean cannot see or
    resist every pairwise 3D distance shrinking together in proportion, so
    that direction was a free ride for 300 steps of gradient noise to drift
    down, and it did. A reference measured once, from a model this script
    has not yet touched, has nothing to do with wherever training later
    drifts -- restoring the role `repel_margin` played before normalization
    replaced it, just measured empirically instead of hand-picked.

    Averaged over `num_samples` independent random neighbour draws (the same
    sample_in_segment_neighbors/sample_out_segment_neighbors the training
    loop itself calls) rather than trusting a single draw, since any one
    draw of 2048*16 pairs is itself a sample with some variance.

    Returns a plain float.
    """
    num_centers = coords.shape[0]
    num_neighbors = num_neighbors_in_segment + num_neighbors_out_segment
    with torch.no_grad():
        vertex_features = encoder(vertices)
        draws = []
        for _ in range(num_samples):
            neighbor_coords_in = sample_in_segment_neighbors(
                coords, segment_ids, seg_idx, k=num_neighbors_in_segment)
            neighbor_coords_out = sample_out_segment_neighbors(
                coords, segment_ids, seg_idx, k=num_neighbors_out_segment)
            neighbor_coords = torch.cat([neighbor_coords_in, neighbor_coords_out], dim=1)

            all_coords = torch.cat([coords, neighbor_coords.reshape(-1, 2)], dim=0)
            pixel_features = index_pixel_features(all_pixel_features, all_coords)
            soft_points = soft_map_to_vertices(
                pixel_features, vertex_features, vertices, k_vertices_per_pixel)

            center_points = soft_points[:num_centers]
            neighbor_points = soft_points[num_centers:].view(num_centers, num_neighbors, 3)
            d_3D = (center_points.unsqueeze(1) - neighbor_points).pow(2).sum(-1).sqrt()
            draws.append(d_3D.mean().item())

    return sum(draws) / len(draws)


def measure_feature_mu(encoder, vertices, coords, segment_ids, seg_idx, all_pixel_features,
                      k_vertices_per_pixel, num_neighbors_in_segment, num_neighbors_out_segment,
                      temperature=1.0, num_samples=5, student_mode='soft', mu_stat='mean'):
    """The two FIXED feature-space normalizers feature_rkd_distance_loss() needs -- measured once, here,
    from the untouched phase-1 encoder, before any phase-2 training happens.

    Exactly the same construction, and the same reasoning, as measure_mu_3D_reference above: a
    normalizer taken from the live batch every step cannot see all distances shrinking together in
    proportion, which is how the original relational loss collapsed (PROJECT_NOTES.md 2026-08-10).
    Averaged over `num_samples` independent neighbour draws for the same reason it is there.

    The image side is frozen, so its mu is a genuine constant of the data; the mesh side is
    measured from phase-1 and then held while the encoder moves away from it -- which is the point.

    Returns (mu_img, mu_msh) as plain floats.
    """
    num_centers = coords.shape[0]
    num_neighbors = num_neighbors_in_segment + num_neighbors_out_segment
    with torch.no_grad():
        vertex_features = encoder(vertices)
        img_draws, msh_draws = [], []
        for _ in range(num_samples):
            neighbor_coords_in = sample_in_segment_neighbors(
                coords, segment_ids, seg_idx, k=num_neighbors_in_segment)
            neighbor_coords_out = sample_out_segment_neighbors(
                coords, segment_ids, seg_idx, k=num_neighbors_out_segment)
            neighbor_coords = torch.cat([neighbor_coords_in, neighbor_coords_out], dim=1)

            all_coords = torch.cat([coords, neighbor_coords.reshape(-1, 2)], dim=0)
            pixel_features = index_pixel_features(all_pixel_features, all_coords)
            _, topk_idx, weights = soft_map_to_vertices(
                pixel_features, vertex_features, vertices, k_vertices_per_pixel,
                return_topk=True, temperature=temperature)

            msh_all = (hard_feature if student_mode == 'argmax' else soft_feature)(
                vertex_features, topk_idx, weights)
            mu_img, mu_msh = measure_feature_mu_references(
                pixel_features[:num_centers],
                pixel_features[num_centers:].view(num_centers, num_neighbors, -1),
                msh_all[:num_centers],
                msh_all[num_centers:].view(num_centers, num_neighbors, -1),
                stat=mu_stat)
            img_draws.append(mu_img)
            msh_draws.append(mu_msh)

    return sum(img_draws) / len(img_draws), sum(msh_draws) / len(msh_draws)


def compute_bbox_diagonals(coarse_mask, vertices):
    """The two FIXED normalizers `--bbox_norm` needs -- each a plain
    geometric property of one domain's own data, computed once, before
    training, with no dependence on which pixels get sampled as centers/
    neighbours or on the encoder's live output at all.

    Unlike `measure_mu_3D_reference` (a MEAN over a sample of pairs, which
    inherits whatever imbalance that sample has -- on this mesh, centers are
    drawn uniform-over-pixels, so a segment's share of the 2048 centers is
    proportional to its pixel count: body 66.5%, neck 16.7%, headstock only
    4.5%), a bounding-box diagonal has nothing to do with sampling at all --
    it is a fixed fact about the whole object's extent, identical no matter
    which pixels or vertices happen to get drawn on a given step.

    Args:
        coarse_mask ndarray [H, W] bool  the object mask center pixels are
                                          drawn from
        vertices    tensor [V, 3]        mesh vertices (already loaded/
                                          normalized by loadmesh)

    Returns (bbox_diag_2D, bbox_diag_3D), both plain positive floats.
    """
    rows, cols = coarse_mask.nonzero()
    if rows.size == 0:
        raise ValueError('coarse_mask has zero True pixels -- cannot compute a bbox diagonal')
    height = float(rows.max() - rows.min() + 1)
    width = float(cols.max() - cols.min() + 1)
    bbox_diag_2D = (height ** 2 + width ** 2) ** 0.5

    mins = vertices.min(dim=0).values
    maxs = vertices.max(dim=0).values
    bbox_diag_3D = (maxs - mins).norm().item()

    for name, value in (('2D', bbox_diag_2D), ('3D', bbox_diag_3D)):
        if value is None or not (value == value) or value <= 0 or value == float('inf'):
            raise ValueError('computed bbox_diag_%s=%r is not a valid positive finite float -- '
                             'refusing to silently fall back to LIVE normalization under '
                             '--bbox_norm' % (name, value))

    return bbox_diag_2D, bbox_diag_3D


def default_geo_table_2d_path(geo_table_3d_path, object_mask_path, seed):
    """Shared by --geodesic and --cyc_weight>0 (cycle-consistency needs the 2D table even when
    --geodesic itself is off -- see modules/cycle_consistency_loss.py). Anchors to
    geo_table_3d_path's own directory (not a hardcoded one) so a custom --geo_table_3d_path
    still finds its matching 2D table alongside it, same as before this was factored out."""
    return os.path.join(os.path.dirname(geo_table_3d_path),
                        f'{os.path.splitext(os.path.basename(object_mask_path))[0]}'
                        f'_geo_table_2d_seed{seed}.pth')


def load_geodesic_table_2d(geo_table_2d_path, object_mask_path, coords, num_center_pixels, seed,
                           pool_mode=False):
    """Loads the 2D table (analysis/build_geodesic_tables.py) and verifies it was actually built
    against THIS run's centers/mask -- a stale or mismatched table would silently misattribute
    distances to the wrong pixels, exactly the class of bug check_mask_matches_knn already
    guards against elsewhere in this project's analysis scripts. Factored out of
    load_geodesic_tables so --cyc_weight>0 can load just this table without also requiring the
    (unrelated, --geodesic-only) 3D vertex-to-vertex table.

    pool_mode=False (default): existing exact-match behavior, byte-identical to before
    --resample_centers existed -- `coords` must be this run's fixed, already-sampled
    --num_center_pixels centers, and the table's every row is used every step.

    pool_mode=True (--resample_centers): `coords` is instead the FULL candidate pool from
    sample_grid_center_pixels (typically far more than num_center_pixels) -- the table's
    `num_center_pixels`/`seed` fields don't apply (that table was built with --center_mode
    grid, not random), so those checks are skipped in favor of checking `center_mode`/
    `grid_stride` instead. Returns the WHOLE pool table; the caller draws a fresh random
    row-subset from it every training step.

    Returns (geo_table_2d, geo_diag_2d float, pixel_to_node [H, W],
    geodesic_safe_mask bool ndarray [H, W]) -- all still on CPU. geo_table_2d is
    [num_center_pixels, M] in the default mode, [len(coords), M] (the whole pool) in pool_mode.
    """
    table_2d = torch.load(geo_table_2d_path, map_location='cpu')

    if pool_mode:
        if table_2d.get('center_mode') != 'grid':
            raise ValueError(f'{geo_table_2d_path} was not built with --center_mode grid '
                             f'(analysis/build_geodesic_tables.py) -- --resample_centers needs '
                             f'a candidate-pool table, not an exact-match one')
    else:
        if table_2d['num_center_pixels'] != num_center_pixels:
            raise ValueError(f'{geo_table_2d_path} was built with num_center_pixels='
                             f'{table_2d["num_center_pixels"]}, this run wants {num_center_pixels} '
                             f'-- rebuild the table or fix --num_center_pixels')
        if table_2d['seed'] != seed:
            raise ValueError(f'{geo_table_2d_path} was built with seed={table_2d["seed"]}, this run '
                             f'wants seed={seed} -- the sampled centers would not match')
    if os.path.basename(table_2d['object_mask_path']) != os.path.basename(object_mask_path):
        raise ValueError(f'{geo_table_2d_path} was built from mask '
                         f'{table_2d["object_mask_path"]!r}, this run\'s --object_mask_path is '
                         f'{object_mask_path!r} -- refusing to trust a table built from a '
                         f'different mask')
    if not torch.equal(table_2d['coords'], coords):
        raise ValueError(f'{geo_table_2d_path}\'s stored centers do not match the centers this '
                         f'run just sampled -- sample_center_pixels/sample_grid_center_pixels '
                         f'must have changed, or something else is inconsistent; table rows '
                         f'would silently line up with the wrong pixels otherwise')

    return (table_2d['table'], table_2d['diagonal'], table_2d['pixel_to_node'],
           table_2d['geodesic_safe_mask'].numpy())


def check_geo_table_3d_matches_mesh(table_3d, geo_table_3d_path, obj_path, num_vertices):
    """Guard against running with a 3D geodesic table built for a DIFFERENT mesh.

    Added 2026-09-02, when meshes/guitar_isotropic.obj entered the project. The 2D table has
    validated itself against its own mask/centers since it was written (load_geodesic_table_2d);
    the 3D table never did, because for a year there was only one mesh.

    The vertex-count check alone is not enough. Two meshes can share a vertex count and be
    completely different, and -- the reason this matters here -- modules.utils.loadmesh normalizes
    by `mean(vertices)`, the vertex CENTROID, so two triangulations of the SAME surface do not even
    live in the same coordinate frame. A mismatched table would not crash; it would silently train
    against another mesh's distances. So the table's recorded obj_path is compared too.
    """
    if table_3d.shape[0] != num_vertices:
        raise ValueError(
            f'3D geodesic table {geo_table_3d_path} is {tuple(table_3d.shape)} but the mesh at '
            f'{obj_path} has {num_vertices} vertices -- the table was built for a different mesh. '
            f'Rebuild it with scripts/build_geo_table_3d_only.py --obj_path {obj_path}')


def load_geodesic_tables(geo_table_3d_path, geo_table_2d_path, object_mask_path, coords,
                         num_center_pixels, seed, pool_mode=False):
    """Loads both precomputed tables (analysis/build_geodesic_tables.py) -- see
    load_geodesic_table_2d's docstring for the 2D-side validation this relies on, and for what
    pool_mode changes.

    Returns (geo_table_3d [V, V], geo_diag_3d float, geo_table_2d, geo_diag_2d,
    pixel_to_node [H, W], geodesic_safe_mask bool ndarray [H, W]) -- all
    tensors still on CPU, moved to `device` by the caller alongside everything else.
    """
    table_3d = torch.load(geo_table_3d_path, map_location='cpu')
    geo_table_2d, geo_diag_2d, pixel_to_node, geodesic_safe_mask = load_geodesic_table_2d(
        geo_table_2d_path, object_mask_path, coords, num_center_pixels, seed, pool_mode=pool_mode)

    return (table_3d['table'], table_3d['diagonal'], geo_table_2d, geo_diag_2d,
           pixel_to_node, geodesic_safe_mask)


def precompute_center_pixel_geodesic_distances(coords, geo_table_2d, pixel_to_node):
    """[num_centers, num_centers] pairwise GEODESIC pixel distance among this run's fixed
    center pixels, sliced directly out of the already-built 2D geodesic table -- no new
    geodesic computation. Same center_nodes lookup analysis/build_geodesic_tables.py's own
    build_2d_table uses for its dijkstra source nodes. Computed once (still CPU here -- caller
    moves the small result to `device`, same pattern as geodesic_distance_2d_lookup's own
    "only the small [n,k] RESULT crosses to the encoder's device" convention).
    """
    center_nodes = pixel_to_node[coords[:, 0], coords[:, 1]]
    if (center_nodes < 0).any():
        raise ValueError('a center pixel has no entry in the geodesic-safe mask (pixel_to_node '
                         '== -1) -- sample_center_pixels must be drawing against the same mask '
                         'build_geodesic_tables.py used')
    return geo_table_2d[:, center_nodes]


def save_loss_plot(losses, drifts, checkpoint_every, path):
    fig, (ax_loss, ax_drift) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_loss.plot(losses)
    ax_loss.set_ylabel('relational loss')
    ax_loss.set_title('phase-2 relational loss training')
    ax_loss.grid(alpha=0.3)

    if drifts:
        steps, values = zip(*drifts)
        ax_drift.plot(steps, values, marker='o')
        ax_drift.set_ylabel('mean cosine vs phase 1')
        ax_drift.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax_drift.set_xlabel('step')
    ax_drift.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def train(args):
    # Phase-2 was entirely unseeded until 2026-08-29: every torch.rand in the neighbour samplers
    # drew from the global RNG, so the same recipe never reproduced itself. Seeded in TWO places
    # -- here for the one-time setup draws (mu references, center sampling), and again immediately
    # before the training loop so the training stream does NOT depend on how much RNG the setup
    # happened to consume. That second reseed is what makes e.g. --feat_mu_mode fixed vs live a
    # clean A/B: 'live' skips measure_feature_mu's 5 draws, which would otherwise shift every
    # subsequent neighbour draw and confound the comparison.
    torch.manual_seed(args.train_seed)
    mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)
    vertices = mesh.vertices
    print('device=%s  vertices=%s' % (device, tuple(vertices.shape)))

    encoder = load_phase1_encoder(args)
    phase1_features = assert_resume_is_sane(
        encoder, vertices, os.path.join(args.init_encoder_dir, 'pred_f.pth'))

    # Everything here stays on CPU: these coordinates index the 3.6 GB pixel
    # feature file below, and torch rejects a CUDA index against a CPU source.
    coarse_mask = torch.load(args.object_mask_path, map_location='cpu').numpy()
    fine_segments = torch.load(args.fine_seg_path, map_location='cpu').numpy()
    if args.geodesic:
        # Restrict to the largest connected component BEFORE sampling centers, matching
        # analysis/build_geodesic_tables.py's own fix (2026-09-03, found on demo/shic's bear3
        # mask: 5 components, one center landed on a dropped speck and build_2d_table crashed
        # with "some sample centers are not on-object pixels"). MUST mirror that script's
        # filtering exactly, or the centers sampled here will not match the geo_table_2d file's
        # stored ones and load_geodesic_tables's own consistency check rejects the run. Gated on
        # --geodesic only: Euclidean training never touches the 2D table and a speck center is
        # harmless there (see build_2d_table's docstring) -- restricting unconditionally would
        # silently change which centers existing Euclidean (guitar) runs draw, for no benefit.
        from analysis.build_geodesic_tables import largest_connected_component
        coarse_mask, _num_components = largest_connected_component(coarse_mask)

    if args.resample_centers:
        if args.cyc_weight > 0.0:
            raise ValueError('--resample_centers is not implemented together with --cyc_weight '
                             '> 0 -- precompute_center_pixel_geodesic_distances is computed once '
                             'against a fixed center set, which --resample_centers breaks')
        if args.fixed_neighbors:
            raise ValueError('--resample_centers with --fixed_neighbors makes no sense: '
                             'fixed_neighbor_coords would be sampled once against ONE center '
                             'set and then reused against every step\'s different centers')
        if not args.geodesic:
            raise ValueError('--resample_centers currently requires --geodesic: in Euclidean '
                             'mode nothing stops a much larger, un-tabulated candidate pool, '
                             'but it has not been wired up (measure_mu_3D_reference etc. still '
                             'assume one fixed center set) -- ask if this is wanted')
        # CANDIDATE POOL (typically tens of thousands of pixels, not args.num_center_pixels) --
        # a fresh random subset of this becomes the actual centers every step, drawn below in
        # the training loop. Must be produced by the exact same call
        # (sample_grid_center_pixels(coarse_mask, fine_segments, stride)) the --center_mode grid
        # table build used, or the table's rows won't line up with these coordinates.
        candidate_coords, _, candidate_segment_ids = sample_grid_center_pixels(
            coarse_mask, fine_segments, stride=args.center_grid_stride)
        num_candidates = candidate_coords.shape[0]
        if args.num_center_pixels > num_candidates:
            raise ValueError(f'--num_center_pixels={args.num_center_pixels} exceeds the '
                             f'{num_candidates}-pixel candidate pool at --center_grid_stride='
                             f'{args.center_grid_stride} -- use a smaller stride or fewer '
                             f'centers per step')
        print('--resample_centers: %d candidate centers (grid stride %d), %d resampled fresh '
              'every step (was: %d centers sampled once, fixed for the whole run)'
              % (num_candidates, args.center_grid_stride, args.num_center_pixels,
                 args.num_center_pixels))
        # Placeholder for any one-time setup below that still wants SOME initial centers (e.g.
        # seg_idx doesn't need this, but keeping the variable defined keeps the rest of this
        # function's shape close to the non-resample path). Real per-step values are drawn
        # fresh in the training loop.
        step_center_idx = torch.randperm(num_candidates)[:args.num_center_pixels]
        coords = candidate_coords[step_center_idx]
        segment_ids = candidate_segment_ids[step_center_idx]
    else:
        # Centers: sampled once, fixed for the whole run (same convention as
        # train_mapping_loss.py). Only the 16 neighbours per center are resampled
        # every step -- see modules/relational_loss.py for why.
        coords, _, segment_ids = sample_center_pixels(
            coarse_mask, fine_segments, args.num_center_pixels, seed=args.seed)
    num_centers = coords.shape[0]
    print('sampled %d center pixels across %d segments'
          % (num_centers, len(torch.unique(segment_ids))))

    if args.geodesic:
        geo_table_2d_path = args.geo_table_2d_path or default_geo_table_2d_path(
            args.geo_table_3d_path, args.object_mask_path, args.seed)
        geo_table_3d, geo_diag_3d, geo_table_2d, geo_diag_2d, pixel_to_node, geodesic_safe_mask = \
            load_geodesic_tables(args.geo_table_3d_path, geo_table_2d_path,
                                 args.object_mask_path,
                                 candidate_coords if args.resample_centers else coords,
                                 args.num_center_pixels, args.seed,
                                 pool_mode=args.resample_centers)
        check_geo_table_3d_matches_mesh(geo_table_3d, args.geo_table_3d_path, args.obj_path,
                                        mesh.vertices.shape[0])
        table_3d_obj = torch.load(args.geo_table_3d_path, map_location='cpu').get('obj_path')
        if table_3d_obj is not None and os.path.realpath(table_3d_obj) != os.path.realpath(args.obj_path):
            raise ValueError(
                f'3D geodesic table {args.geo_table_3d_path} was built for {table_3d_obj} but this '
                f'run uses --obj_path {args.obj_path}. Same vertex count is NOT enough: loadmesh '
                f'normalizes by the vertex centroid, so two triangulations of the same surface sit '
                f'in different coordinate frames. Point --geo_table_3d_path at this mesh\'s table.')
        geo_table_3d = geo_table_3d.to(device)
        print('  geodesic tables loaded: 3D %s (diagonal=%.5f), 2D %s (diagonal=%.5f)'
              % (tuple(geo_table_3d.shape), geo_diag_3d, tuple(geo_table_2d.shape), geo_diag_2d))

        # seg_idx must never be able to hand out one of the disconnected speck pixels as a
        # neighbour -- caught by a real smoke-test failure, not anticipated: the specks turned
        # out to carry a genuine fine_segment id (16 here), not background, so
        # sample_in_segment_neighbors (which draws from build_segment_index's
        # segment_pixel_coords, built from `fine_segments == s` ALONE -- it does not consult its
        # own `coarse_mask` argument at all, only `mask_coords`/`mask_segids` do) could still
        # draw them even after restricting `coarse_mask` to geodesic_safe_mask, which only fixed
        # sample_out_segment_neighbors's path. Fixed at the source instead: zero out
        # fine_segments at every non-geodesic-safe pixel in a COPY used only for this seg_idx,
        # so no segment's pixel list can ever contain one, regardless of build_segment_index's
        # own (mask-blind) construction of that structure. build_segment_index itself is
        # untouched -- every existing Euclidean run keeps calling it exactly as before.
        fine_segments_geodesic_safe = fine_segments.copy()
        fine_segments_geodesic_safe[~geodesic_safe_mask] = 0
        n_dropped = int((fine_segments != fine_segments_geodesic_safe).sum())
        if n_dropped > 0:
            print(f'  {n_dropped} disconnected pixel(s) zeroed out of fine_segments for '
                  f'neighbour sampling (were labeled with a real segment id despite being '
                  f'geodesically unreachable from the main mask component)')
        seg_idx = build_segment_index(fine_segments_geodesic_safe, geodesic_safe_mask)
    else:
        seg_idx = build_segment_index(fine_segments, coarse_mask)
    num_neighbors = args.num_neighbors_in_segment + args.num_neighbors_out_segment

    # Cycle-consistency needs a 2D geodesic table too, but is otherwise independent of
    # --geodesic (which is about d_2D/d_3D in relational_loss itself) -- reuse the table already
    # loaded above if --geodesic is on, else load just the 2D table. See
    # modules/cycle_consistency_loss.py for the full design.
    if args.cyc_weight > 0.0:
        if args.geodesic:
            cyc_geo_table_2d, cyc_geo_diag_2d, cyc_pixel_to_node = (
                geo_table_2d, geo_diag_2d, pixel_to_node)
        else:
            cyc_geo_table_2d_path = args.geo_table_2d_path or default_geo_table_2d_path(
                args.geo_table_3d_path, args.object_mask_path, args.seed)
            cyc_geo_table_2d, cyc_geo_diag_2d, cyc_pixel_to_node, _ = load_geodesic_table_2d(
                cyc_geo_table_2d_path, args.object_mask_path, coords,
                args.num_center_pixels, args.seed)
        # Normalized ONCE here (center_geo_dist is fixed for the whole run, unlike
        # relational_loss_geodesic's own norm_d_2D which is recomputed every step against
        # resampled neighbours) -- same divide-by-geodesic-diameter convention as
        # relational_loss.py's norm_d_2D, so cyc_loss lands in roughly the same [0, ~1] range as
        # relational_loss instead of raw geodesic pixels (~0-1169.7px on this mesh/mask).
        # Tal's explicit call, following SHIC's own stated principle ("values that make the
        # losses roughly of similar scale") rather than reusing SHIC's literal beta -- our two
        # losses don't share SHIC's raw scale, but normalizing puts them on comparable footing so
        # --relational_weight/--cyc_weight sweep values are directly comparable, unlike raw pixels.
        center_geo_dist = precompute_center_pixel_geodesic_distances(
            coords, cyc_geo_table_2d, cyc_pixel_to_node).to(device) / max(cyc_geo_diag_2d, 1e-8)
        print('  cycle-consistency loss: ON (--cyc_weight %g, %s variant), round trip '
              'restricted to the %d fixed center pixels for both legs, geodesic-weighted, '
              'normalized by 2D geodesic diameter %.2fpx'
              % (args.cyc_weight, 'SOFT/no-snap' if args.cyc_soft else 'hard-snap',
                 num_centers, cyc_geo_diag_2d))
    else:
        center_geo_dist = None

    # 2026-08-11 A/B test: which 2026-08-10 collapse fix actually matters,
    # the FIXED mu_3D_reference or FIXED (not resampled) neighbours? Both
    # default to the already-verified-safe combination (fixed reference,
    # resampled neighbours -- Version 1 below); --fixed_neighbors/
    # --live_mu_3d switch each axis independently. See
    # modules/relational_loss.py's module docstring for the full argument.
    if args.fixed_neighbors:
        neighbor_coords_in = sample_in_segment_neighbors(
            coords, segment_ids, seg_idx, k=args.num_neighbors_in_segment)
        neighbor_coords_out = sample_out_segment_neighbors(
            coords, segment_ids, seg_idx, k=args.num_neighbors_out_segment,
            max_retries=args.out_segment_max_retries)
        fixed_neighbor_coords = torch.cat([neighbor_coords_in, neighbor_coords_out], dim=1)
        print('  %d in-segment + %d out-of-segment neighbours per center, sampled ONCE '
              '(--fixed_neighbors) and reused every step'
              % (args.num_neighbors_in_segment, args.num_neighbors_out_segment))
    else:
        print('  %d in-segment + %d out-of-segment neighbours per center, resampled every step'
              % (args.num_neighbors_in_segment, args.num_neighbors_out_segment))

    # Loaded once, kept resident for the whole run (~3.6 GB) and indexed
    # in-memory every step -- see index_pixel_features's docstring for why
    # this can't reuse load_pixel_features's per-call disk read here.
    all_pixel_features = torch.load(args.pix_dino_features_path, map_location='cpu')

    if sum([args.bbox_norm, args.live_mu_3d, args.geodesic]) > 1:
        raise ValueError('--bbox_norm, --live_mu_3d, and --geodesic are mutually exclusive '
                         'normalization/distance modes -- pick exactly one (or none, for the '
                         'FIXED-mean-from-phase-1 default).')

    if args.geodesic:
        # Geodesic diameters (max pairwise geodesic distance), not Euclidean bbox diagonals --
        # Tal's explicit call: since the distances themselves are now geodesic, the reference
        # they're compared against should be too, matching SHIC's own convention of normalizing
        # by the max geodesic distance on the mesh. See modules/relational_loss.py's geodesic-
        # variant docstring for the full design.
        mu_2D_reference, mu_3D_reference = geo_diag_2d, geo_diag_3d
        print('  normalization: GEODESIC (--geodesic) -- d_2D / %.2fpx (2D geodesic diameter), '
              'd_3D / %.5f (3D geodesic diameter), both fixed' % (mu_2D_reference, mu_3D_reference))
    elif args.bbox_norm:
        # FIXED bounding-box-diagonal normalizers for BOTH d_2D and d_3D --
        # see compute_bbox_diagonals's docstring for why this is immune to
        # the segment-sampling imbalance that a mean-of-pairs normalizer
        # (either mode below) inherits. mu_2D_reference is asserted non-None
        # here on purpose: if compute_bbox_diagonals ever returned something
        # that couldn't be trusted, we want a loud failure, not a silent
        # fall-through to relational_loss's None-means-LIVE default.
        mu_2D_reference, mu_3D_reference = compute_bbox_diagonals(coarse_mask, vertices)
        assert mu_2D_reference is not None and mu_3D_reference is not None
        print('  normalization: BBOX (--bbox_norm) -- d_2D / %.2fpx (coarse mask bbox '
              'diagonal), d_3D / %.5f (mesh vertex bbox diagonal), both fixed, neither a '
              'sampled-pair statistic' % (mu_2D_reference, mu_3D_reference))
    elif args.live_mu_3d:
        # LIVE mode: mu_3D recomputed from the live encoder every step, same
        # as the original (pre-2026-08-10-fix) relational_loss(). Known to
        # collapse when paired with resampled-every-step neighbours; this
        # session is testing it paired with --fixed_neighbors instead.
        mu_2D_reference = None
        mu_3D_reference = None
        print('  mu_3D: LIVE (--live_mu_3d) -- recomputed from the training encoder every step')
    else:
        # FIXED normalizer for relational_loss's d_3D, measured once from the
        # untouched phase-1 encoder -- see measure_mu_3D_reference's docstring
        # for why this must not be recomputed from the live encoder every step.
        mu_2D_reference = None
        mu_3D_reference = measure_mu_3D_reference(
            encoder, vertices, coords, segment_ids, seg_idx, all_pixel_features,
            args.k_vertices_per_pixel, args.num_neighbors_in_segment,
            args.num_neighbors_out_segment, num_samples=args.mu_3d_reference_samples)
        print('  mu_3D reference (fixed, phase-1 encoder, averaged over %d draws): %.5f'
              % (args.mu_3d_reference_samples, mu_3D_reference))

    # Feature-space RKD normalizers (added 2026-08-28). Measured from the same untouched phase-1
    # encoder as mu_3D above, for the same anti-collapse reason. Skipped entirely when neither
    # feature term is in use and nothing asked for their magnitudes.
    mu_img_reference = mu_msh_reference = None
    if (args.feat_dist_weight > 0.0 or args.feat_angle_weight > 0.0
            or args.feat_pair_angle_weight > 0.0 or args.log_feature_terms):
        if args.feat_mu_mode == 'fixed':
            mu_img_reference, mu_msh_reference = measure_feature_mu(
                encoder, vertices, coords, segment_ids, seg_idx, all_pixel_features,
                args.k_vertices_per_pixel, args.num_neighbors_in_segment,
                args.num_neighbors_out_segment, temperature=args.temperature,
                num_samples=args.mu_3d_reference_samples,
                student_mode=args.feat_student_mode, mu_stat=args.feat_mu_stat)
            print('  feature-RKD mu references (fixed, phase-1 encoder, %d draws, stat=%s): '
                  'mu_img=%.5f  mu_msh=%.5f'
                  % (args.mu_3d_reference_samples, args.feat_mu_stat,
                     mu_img_reference, mu_msh_reference))
        else:
            print('  feature-RKD mu: LIVE per batch (the RKD paper\'s own choice)')
        print('  feature-RKD student: %s'
              % ('ARGMAX (k=1, no mixing -- gradient lands on one vertex per pixel)'
                 if args.feat_student_mode == 'argmax' else 'soft (weighted average over k)'))
        print('  feature-RKD soft-map weights: %s'
              % ('DETACHED (circularity test -- gradient can only move features)'
                 if args.feat_detach_weights else 'attached (default)'))
        # 2026-09-04: the pair-angle weight MUST be printed here too. It was missing when the flag
        # was added, so a run with only the pair-angle term on printed "beta=0, gamma=0" and read
        # as though no feature term were active at all -- exactly backwards.
        print('  feature-RKD weights: beta (distance)=%g, gamma (3-point angle)=%g, '
              'pair-angle (2-point)=%g%s'
              % (args.feat_dist_weight, args.feat_angle_weight, args.feat_pair_angle_weight,
                 '  [logging only, no feature term is in the optimized loss]'
                 if (args.feat_dist_weight == 0.0 and args.feat_angle_weight == 0.0
                     and args.feat_pair_angle_weight == 0.0) else ''))
        print('  feature-RKD detach: features=%s, weights=%s  (gradFW = neither, '
              'gradF = weights detached, gradW = features detached)'
              % (args.feat_detach_features, args.feat_detach_weights))
        if args.relational_weight == 0.0:
            print('  *** relational_weight=0: the SPATIAL term is OFF -- nothing anchors the map '
                  'to geometry (2026-09-04: this diverged when tried) ***')

    print('  distance_weight: %s' % (args.distance_weight or 'none (every pair weighted equally)'))
    print('  mu_3d_scale: %g%s' % (args.mu_3d_scale, ' (no-op)' if args.mu_3d_scale == 1.0 else ''))
    if args.balance_weight > 0.0:
        print('  vertex-balance loss: ON (--balance_weight %g), Switch-Transformer-style '
              'load-balancing auxiliary loss over the %d fixed centers\' top-%d vertex '
              'candidates -- see modules/vertex_balance_loss.py'
              % (args.balance_weight, num_centers, args.k_vertices_per_pixel))

    optimizer = torch.optim.Adam(encoder.parameters(), args.lr, eps=args.adam_eps)
    os.makedirs(args.out_dir, exist_ok=True)

    losses = []
    drifts = []
    # See the note at the top of train(): decouples the per-step neighbour stream from the
    # amount of RNG the setup above consumed.
    torch.manual_seed(args.train_seed + 1)

    for step in range(1, args.num_iters + 1):
        optimizer.zero_grad()

        if args.resample_centers:
            # Fresh random subset of the candidate pool every step -- unlike the fixed-once
            # `coords` the non-resample branch below keeps for the whole run. Neighbours are
            # unaffected: sample_in_segment_neighbors/sample_out_segment_neighbors already draw
            # from the FULL on-object pixel set regardless of which centers are active this
            # step (see geodesic_distance_2d_lookup -- the table's columns already cover every
            # on-object pixel, only the ROWS are restricted to a candidate set).
            step_center_idx = torch.randperm(num_candidates)[:args.num_center_pixels]
            coords = candidate_coords[step_center_idx]
            segment_ids = candidate_segment_ids[step_center_idx]
            if args.geodesic:
                # Cheap fancy-indexing into the already-loaded pool table -- no new geodesic
                # computation, same as the non-resample path's one-time row selection.
                geo_table_2d_center_rows = geo_table_2d[step_center_idx]
        elif args.geodesic:
            geo_table_2d_center_rows = geo_table_2d

        if args.fixed_neighbors:
            neighbor_coords = fixed_neighbor_coords
        else:
            neighbor_coords_in = sample_in_segment_neighbors(
                coords, segment_ids, seg_idx, k=args.num_neighbors_in_segment)
            neighbor_coords_out = sample_out_segment_neighbors(
                coords, segment_ids, seg_idx, k=args.num_neighbors_out_segment,
                max_retries=args.out_segment_max_retries)
            neighbor_coords = torch.cat([neighbor_coords_in, neighbor_coords_out], dim=1)

        # Same flat-batch pattern as train_mapping_loss.py: centers are rows
        # [:n], neighbour (i, j) is row n + i*num_neighbors + j.
        all_coords = torch.cat([coords, neighbor_coords.reshape(-1, 2)], dim=0)
        pixel_features = index_pixel_features(all_pixel_features, all_coords)

        # Live encoder output, NOT the saved pred_f.pth -- see
        # train_mapping_loss.py's identical comment for why.
        vertex_features = encoder(vertices)

        if args.geodesic:
            # return_topk=True: geodesic distance needs each pixel's own top-K vertex
            # candidates/weights, not the already-collapsed (generally off-surface) point --
            # see modules/relational_loss.py's geodesic-variant docstring for why the collapsed
            # point can't be used here the way it is for Euclidean distance.
            _, topk_idx, weights = soft_map_to_vertices(
                pixel_features, vertex_features, vertices, args.k_vertices_per_pixel,
                return_topk=True, temperature=args.temperature)

            center_topk_idx = topk_idx[:num_centers]
            center_weights = weights[:num_centers]
            neighbor_topk_idx = topk_idx[num_centers:].view(
                num_centers, num_neighbors, args.k_vertices_per_pixel)
            neighbor_weights = weights[num_centers:].view(
                num_centers, num_neighbors, args.k_vertices_per_pixel)

            loss, mu_2D, mu_3D_live = relational_loss_geodesic(
                center_topk_idx, center_weights, neighbor_topk_idx, neighbor_weights,
                neighbor_coords, geo_table_3d, geo_table_2d_center_rows, pixel_to_node,
                mu_3D_reference=mu_3D_reference, mu_2D_reference=mu_2D_reference,
                mu_3d_scale=args.mu_3d_scale, huber_delta=args.huber_delta)
        else:
            # return_topk=True even here (unused for relational_loss itself, which only needs
            # `points`): --balance_weight needs the same per-pixel top-K weights the geodesic
            # branch above already gets for free, and requesting them costs nothing extra --
            # soft_map_to_vertices computes topk_idx/weights internally either way, return_topk
            # only controls whether they're handed back.
            soft_points, topk_idx, weights = soft_map_to_vertices(
                pixel_features, vertex_features, vertices, args.k_vertices_per_pixel,
                return_topk=True, temperature=args.temperature)

            center_points = soft_points[:num_centers]
            neighbor_points = soft_points[num_centers:].view(num_centers, num_neighbors, 3)
            center_topk_idx = topk_idx[:num_centers]
            center_weights = weights[:num_centers]

            loss, mu_2D, mu_3D_live = relational_loss(
                center_points, neighbor_points, coords, neighbor_coords,
                mu_3D_reference=mu_3D_reference, mu_2D_reference=mu_2D_reference,
                distance_weight=args.distance_weight, mu_3d_scale=args.mu_3d_scale,
                huber_delta=args.huber_delta)

        loss = args.relational_weight * loss

        # --- feature-space RKD (added 2026-08-28) -------------------------------------------
        # Two additive terms on the feature dimension, on top of the geometric relational loss
        # above: total = alpha*relational + beta*RKD-D(features) + gamma*RKD-A(features).
        # Both default to weight 0.0, so every pre-existing run is bit-identical.
        # Computed whenever either weight is on OR --log_feature_terms is set, so their raw
        # magnitudes can be read off a run that does not yet use them.
        feat_dist_raw = feat_angle_raw = cos_gap = float('nan')
        mu_img_live = mu_msh_live = float('nan')
        if (args.feat_dist_weight > 0.0 or args.feat_angle_weight > 0.0
            or args.feat_pair_angle_weight > 0.0 or args.log_feature_terms):
            img_center = pixel_features[:num_centers]
            img_neighbors = pixel_features[num_centers:].view(num_centers, num_neighbors, -1)
            # --feat_detach_weights (2026-08-29): THE CIRCULARITY TEST. `weights` come from a softmax
            # over image<->vertex feature similarity, i.e. from the very quantity this loss is trying
            # to fix. Left attached, the loss has two ways down: genuinely improve vertex_features
            # (what we want), or just reshuffle/sharpen the weights so the weighted averages land
            # where it wants (what we do not). Detaching blocks the second path, so the loss can only
            # fall by moving features. Only the FEATURE loss is affected -- the geometric relational
            # loss uses center_points/neighbor_points and is untouched. The other soft_feature() call
            # (measure_feature_mu) is under no_grad, so it is unaffected either way.
            if args.feat_student_mode == 'argmax':
                # k=1: no mixing at all, so --feat_detach_weights is irrelevant here (only the
                # argmax INDEX is taken from `weights`, and an index carries no gradient).
                msh_all = hard_feature(vertex_features, topk_idx, weights)
            else:
                # --feat_detach_features (2026-08-31, Tal): the MIRROR of --feat_detach_weights.
                # Detaching the F values leaves w as the only route to the encoder, which is exactly
                # the structure of the SPATIAL relational loss (its Sum w*X uses fixed vertex
                # POSITIONS, so it has no value path at all). The gradient still reaches the encoder,
                # because w is a softmax over f(i)*F computed from the live, undetached features.
                msh_all = soft_feature(
                    vertex_features.detach() if args.feat_detach_features else vertex_features,
                    topk_idx,
                    weights.detach() if args.feat_detach_weights else weights)
            msh_center = msh_all[:num_centers]
            msh_neighbors = msh_all[num_centers:].view(num_centers, num_neighbors, -1)

            fd_loss, mu_img_live, mu_msh_live = feature_rkd_distance_loss(
                img_center, img_neighbors, msh_center, msh_neighbors,
                mu_img_reference=mu_img_reference, mu_msh_reference=mu_msh_reference,
                huber_delta=args.huber_delta)
            fa_loss, cos_gap = feature_rkd_angle_loss(
                img_center, img_neighbors, msh_center, msh_neighbors,
                huber_delta=args.huber_delta)
            # PAIRWISE (2-point) angle, Tal's variant 2026-09-04 -- cos(center, neighbour), one
            # per pair, no mu anywhere. See feature_rkd_pairwise_angle_loss' docstring for how it
            # differs from the 3-point fa_loss above. Always COMPUTED when the block runs (it is a
            # cheap [n,k] cosine) so its raw magnitude is logged even at weight 0 -- that is how the
            # weight gets chosen, rather than guessed.
            #
            # --feat_pair_angle_neighbors in_segment (Tal, 2026-09-04): use only the CLOSE
            # neighbours. The neighbour axis is built as cat([in_segment, out_segment], dim=1) (see
            # the sampling above), so the first --num_neighbors_in_segment entries ARE the close
            # ones and a plain prefix slice is exactly right. THIS SLICE APPLIES TO THE PAIR-ANGLE
            # TERM ONLY -- the geometric relational loss above still sees all 16, so the matched
            # control (the spatial-only run) stays a ONE-variable comparison.
            if args.feat_pair_angle_neighbors == 'in_segment':
                n_close = args.num_neighbors_in_segment
                pa_img_neighbors = img_neighbors[:, :n_close]
                pa_msh_neighbors = msh_neighbors[:, :n_close]
            else:
                pa_img_neighbors = img_neighbors
                pa_msh_neighbors = msh_neighbors
            fpa_loss, pair_cos_gap = feature_rkd_pairwise_angle_loss(
                img_center, pa_img_neighbors, msh_center, pa_msh_neighbors,
                huber_delta=args.huber_delta)
            feat_dist_raw = fd_loss.item()
            feat_angle_raw = fa_loss.item()
            feat_pair_angle_raw = fpa_loss.item()

            if args.feat_dist_weight > 0.0:
                loss = loss + args.feat_dist_weight * fd_loss
            if args.feat_angle_weight > 0.0:
                loss = loss + args.feat_angle_weight * fa_loss
            if args.feat_pair_angle_weight > 0.0:
                loss = loss + args.feat_pair_angle_weight * fpa_loss

        if args.balance_weight > 0.0:
            bal_loss, bal_f, bal_P = vertex_balance_loss(
                center_topk_idx, center_weights, vertices.shape[0])
            loss = loss + args.balance_weight * bal_loss

        if args.cyc_weight > 0.0:
            cyc_fn = cycle_consistency_loss_soft if args.cyc_soft else cycle_consistency_loss
            cyc_loss, cyc_norm, cyc_num_verts = cyc_fn(
                pixel_features[:num_centers], vertex_features, vertices,
                args.k_vertices_per_pixel, center_geo_dist)
            loss = loss + args.cyc_weight * cyc_loss

        loss.backward()

        if step == 1:
            grad_norm = torch.cat([p.grad.reshape(-1) for p in encoder.parameters()
                                   if p.grad is not None]).norm().item()
            print('step    1  total grad norm %.4e' % grad_norm)

        optimizer.step()
        losses.append(loss.item())

        if step % args.checkpoint_every == 0 or step == args.num_iters:
            with torch.no_grad():
                drift = feature_drift(encoder(vertices), phase1_features)
            drifts.append((step, drift))
            ref_2D_str = 'live' if mu_2D_reference is None else '%.2f' % mu_2D_reference
            ref_3D_str = 'live' if mu_3D_reference is None else '%.5f' % mu_3D_reference
            cyc_str = ('' if args.cyc_weight == 0.0 else
                      '  cyc=%.4f refs=%d/%d' % (cyc_norm, cyc_num_verts, vertices.shape[0]))
            bal_str = ('' if args.balance_weight == 0.0 else
                      '  bal=%.4f cov=%d/%d' % (bal_loss.item(), int((bal_f > 0).sum().item()),
                                                 vertices.shape[0]))
            feat_str = ('' if (args.feat_dist_weight == 0.0 and args.feat_angle_weight == 0.0
                               and args.feat_pair_angle_weight == 0.0
                               and not args.log_feature_terms) else
                        '  featD=%.6f featA=%.6f featPA=%.6f (raw, unweighted; cos_gap=%.4f '
                        'pair_cos_gap=%.4f mu_img_live=%.5f mu_msh_live=%.5f)'
                        % (feat_dist_raw, feat_angle_raw, feat_pair_angle_raw, cos_gap,
                           pair_cos_gap, mu_img_live, mu_msh_live))
            print('step %4d/%d  loss %.6f (mu_2D=%.3f mu_3D_live=%.5f, ref_2D=%s ref_3D=%s)%s%s%s  '
                  'drift(cos vs phase 1) %.4f'
                  % (step, args.num_iters, loss.item(), mu_2D, mu_3D_live,
                     ref_2D_str, ref_3D_str, cyc_str, bal_str, feat_str, drift))

            torch.save({'encoder_state_dict': encoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'fourier_basis': get_fourier_basis(encoder),
                        'step': step, 'loss': loss.item(), 'drift': drift},
                       os.path.join(args.out_dir, 'encoder_checkpoint_step%d.pth' % step))

            if args.stop_at_drift is not None and drift < args.stop_at_drift:
                print('drift %.4f fell below --stop_at_drift %.4f; stopping.'
                      % (drift, args.stop_at_drift))
                break

    # Final artifacts, mirroring phase 1's / train_mapping_loss.py's layout so
    # the analysis scripts can consume this checkpoint exactly like theirs.
    torch.save({'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'fourier_basis': get_fourier_basis(encoder),
                'losses': losses, 'drifts': drifts},
               os.path.join(args.out_dir, 'encoder_checkpoint.pth'))
    with torch.no_grad():
        torch.save(encoder(vertices), os.path.join(args.out_dir, 'pred_f.pth'))

    save_loss_plot(losses, drifts, args.checkpoint_every,
                   os.path.join(args.out_dir, 'relational_loss.png'))
    config = {k: v for k, v in vars(args).items()}
    config['input_sha256'] = {
        path: hash_input(path) for path in (args.object_mask_path, args.fine_seg_path,
                                            args.pix_dino_features_path)}
    with open(os.path.join(args.out_dir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print('\nloss %.6f -> %.6f over %d steps' % (losses[0], losses[-1], len(losses)))
    print('final drift vs phase 1: %.4f' % drifts[-1][1])
    print('saved to %s' % args.out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Mesh / phase-1 checkpoint. The Encoder args MUST match the phase-1 run
    # (README step 2 used --model_type dino --width 1024 --n_classes 1024) or
    # the state dict will not load.
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')
    parser.add_argument('--init_encoder_dir', type=str,
                        default='./experiments/guitar/encoder_dino')
    parser.add_argument('--init_encoder_checkpoint', type=str, default=None,
                        help='defaults to <init_encoder_dir>/encoder_checkpoint.pth')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--n_classes', type=int, default=1024)
    parser.add_argument('--positional_encoding', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=5.0)

    # Data
    parser.add_argument('--object_mask_path', type=str, required=True,
                        help='coarse whole-object mask; center pixels are drawn from it')
    parser.add_argument('--fine_seg_path', type=str,
                        default='./demo/guitar/phase2/guitar_01_fine_segments.pth',
                        help='part-level label map from analysis/build_fine_segments.py; '
                             'defines the in-segment/out-of-segment split neighbours are '
                             'drawn from')
    parser.add_argument('--pix_dino_features_path', type=str, required=True)

    # Loss
    parser.add_argument('--num_center_pixels', type=int, default=2048,
                        help='pixels the loss is computed on per step. Sampled once, then '
                             'fixed for the whole run by default; with --resample_centers, a '
                             'fresh random subset of the candidate pool is drawn every step '
                             'instead, but the per-step COUNT is still this number.')
    parser.add_argument('--resample_centers', action='store_true',
                        help='2026-08-18: instead of sample_center_pixels\'s single fixed draw '
                             'of --num_center_pixels centers for the whole run, draw a fresh '
                             'random --num_center_pixels subset of a much larger candidate pool '
                             '(sample_grid_center_pixels, spacing --center_grid_stride) every '
                             'step -- the neighbours-side counterpart of this already existed '
                             '(resampled every step by default); this extends the same idea to '
                             'centers. Requires --geodesic and a --center_mode grid table '
                             '(analysis/build_geodesic_tables.py) at --geo_table_2d_path. Not '
                             'implemented with --cyc_weight>0 or --fixed_neighbors (see train()\'s '
                             'own checks for why). Off by default -- matches every existing '
                             'recipe/run byte-for-byte.')
    parser.add_argument('--center_grid_stride', type=int, default=3,
                        help='--resample_centers only: pixel spacing of the candidate-center '
                             'grid (sample_grid_center_pixels) -- must match the --center_mode '
                             'grid table\'s own --grid_stride it was built with '
                             '(analysis/build_geodesic_tables.py).')
    parser.add_argument('--feat_dist_weight', type=float, default=0.0,
                        help='beta -- weight of the feature-space RKD-D term (distance between two '
                             'IMAGE features vs distance between the two corresponding MESH '
                             'features). 0.0 = off, and every pre-existing run is bit-identical at '
                             'the default. See modules/feature_rkd_loss.py.')
    parser.add_argument('--feat_angle_weight', type=float, default=0.0,
                        help='gamma -- weight of the feature-space RKD-A term (angle at a center, '
                             'formed with two of its neighbours, in image-feature space vs in '
                             'mesh-feature space). 0.0 = off. Needs no normalizer: a cosine is '
                             'already scale-free.')
    parser.add_argument('--feat_pair_angle_weight', type=float, default=0.0,
                        help='weight of the PAIRWISE (2-point) feature angle term, Tal 2026-09-04: '
                             'cos(center, neighbour) in image-feature space vs in mesh-feature '
                             'space, ONE per (center, neighbour) pair -- the same 2-point structure '
                             '--feat_dist_weight uses, with a cosine instead of a distance, so no '
                             'mu is needed anywhere. NOT the same as --feat_angle_weight, which is '
                             "the paper's 3-point angle between two NEIGHBOURS at the center. "
                             'Intended as an ADDITIVE term alongside the geometric relational loss. '
                             '0.0 = off, and the raw value is logged as featPA even at 0 so the '
                             'weight can be chosen from a measurement. Start an order of magnitude '
                             'below the 3-point term (gamma=0.2 collapsed a run).')
    parser.add_argument('--feat_pair_angle_neighbors', type=str, default='all',
                        choices=['all', 'in_segment'],
                        help="which neighbours the PAIRWISE angle term uses. 'all' (default) = "
                             'every sampled neighbour, i.e. in-segment + out-of-segment, and is '
                             "bit-identical to every run before 2026-09-04. 'in_segment' = only the "
                             'CLOSE ones (the first --num_neighbors_in_segment along the neighbour '
                             'axis, which is built as cat([in, out]) so the prefix is exactly the '
                             'in-segment set). Tal 2026-09-04. AFFECTS THE PAIR-ANGLE TERM ONLY -- '
                             'the geometric relational loss still uses every neighbour, so a run '
                             'with this flag differs from the spatial-only control by the '
                             'pair-angle term alone.')
    parser.add_argument('--feat_detach_weights', action='store_true',
                        help='CIRCULARITY TEST: detach the soft-map weights inside soft_feature(), '
                             'so the feature-RKD terms can only be reduced by changing vertex '
                             'features, not by sharpening the mapping that produced them. Default '
                             'OFF, so every pre-2026-08-29 run is reproduced exactly.')
    parser.add_argument('--feat_student_mode', type=str, default='soft',
                        choices=['soft', 'argmax'],
                        help="how the feature-RKD terms reach the mesh. 'soft' (default) is the "
                             'softmax-weighted average over k vertices, as every run before '
                             "2026-09-01. 'argmax' takes the single best-matching vertex's own "
                             'feature instead: no mixing, so there is no sharpening knob for the '
                             "loss to exploit and a pixel's whole gradient lands on one vertex. "
                             'See hard_feature() in modules/feature_rkd_loss.py.')
    parser.add_argument('--feat_detach_features', action='store_true',
                        help='detach the vertex FEATURES inside the soft student, so the feature-RKD '
                             'gradient reaches the encoder only through the soft-map weights -- the '
                             'mirror of --feat_detach_weights, and structurally what the spatial '
                             'relational loss does. Soft student only (argmax takes an index, which '
                             'carries no gradient, so detaching features there kills the term).')
    parser.add_argument('--feat_mu_stat', type=str, default='mean',
                        help="which statistic of the pair-distance distribution the FIXED feature "
                             "normalizer uses: 'mean' (default, the paper's), 'max' (each space by "
                             "its own largest pair distance), or 'pNN' e.g. 'p95'. max/p95 are much "
                             "larger than the mean, so the loss shrinks and beta must be rescaled.")
    parser.add_argument('--feat_mu_mode', type=str, default='fixed', choices=['fixed', 'live'],
                        help="normalizer for the RKD-D feature term. 'fixed' (default) measures mu "
                             'once from the phase-1 encoder and holds it, matching what the '
                             'geometric relational loss already does and for the same '
                             'anti-collapse reason. \'live\' recomputes it from each batch, which '
                             'is what the RKD paper itself does -- kept as a flag because Tal '
                             'explicitly wants to try it later.')
    parser.add_argument('--log_feature_terms', action='store_true',
                        help='compute and log the two feature-RKD terms unweighted even when both '
                             'weights are 0, to read off their natural magnitudes before choosing '
                             'beta/gamma. Costs one extra soft_feature + two losses per step and '
                             'does NOT affect the optimized loss.')
    parser.add_argument('--temperature', type=float, default=1.0,
                         help='softmax temperature for soft_map_to_vertices (DPC Eq. 1-3). 1.0 = the '
                               'plain softmax every phase-2 run before 2026-08-25 trained with. NOTE: '
                               'the divide also scales the gradient reaching the encoder through the '
                               'weights by 1/T, so T=0.005 is a 200x gradient amplification on that '
                               'path, not just a sharper weighting -- treat lr accordingly.')
    parser.add_argument('--k_vertices_per_pixel', type=int, default=10,
                        help="mesh vertices each pixel is softly mapped onto, nearest in "
                             "FEATURE space (DPC's k_cc)")
    parser.add_argument('--num_neighbors_in_segment', type=int, default=8,
                        help='random pixels drawn from a center\'s own fine segment, '
                             'without replacement, resampled every step')
    parser.add_argument('--num_neighbors_out_segment', type=int, default=8,
                        help='random pixels drawn from outside a center\'s fine segment '
                             '(segment-exclusive), resampled every step')
    parser.add_argument('--out_segment_max_retries', type=int, default=10,
                        help='vectorized rejection-sampling rounds for out-of-segment '
                             'draws that land back in the center\'s own segment')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                        help='Huber transition point on the normalized-distance residual; '
                             "1.0 is RKD's own fixed choice")
    parser.add_argument('--mu_3d_reference_samples', type=int, default=5,
                        help='independent random neighbour draws averaged together to '
                             'measure the FIXED mu_3D normalizer from the phase-1 encoder, '
                             'before training starts -- see measure_mu_3D_reference. Ignored '
                             'when --live_mu_3d is set.')
    parser.add_argument('--live_mu_3d', action='store_true',
                        help='2026-08-11 A/B test, axis 1: normalize d_3D by THIS STEP\'s own '
                             'live mean (the original, pre-2026-08-10-fix behavior) instead of '
                             'a fixed reference measured once from the phase-1 encoder. Off by '
                             'default (matches the verified-safe 2026-08-10 fix).')
    parser.add_argument('--fixed_neighbors', action='store_true',
                        help='2026-08-11 A/B test, axis 2: sample each center\'s in-segment/'
                             'out-of-segment neighbours ONCE, before training, and reuse them '
                             'every step, instead of resampling every step. Off by default '
                             '(matches the verified-safe 2026-08-10 fix).')
    parser.add_argument('--bbox_norm', action='store_true',
                        help='Normalize BOTH d_2D and d_3D by a FIXED bounding-box diagonal '
                             '(d_2D by the coarse mask\'s own pixel-space bbox diagonal, d_3D '
                             'by the mesh\'s own vertex bbox diagonal), instead of any '
                             'mean-of-sampled-pairs statistic (live or fixed-reference). '
                             'Both computed once from the object\'s geometry alone, before '
                             'training -- unlike a sampled mean, immune to the imbalance in '
                             'which segments the 2048 centers happen to land in (measured: '
                             'body 66.5%%, neck 16.7%%, headstock only 4.5%% of centers). '
                             'Standard cross-domain-scale convention (e.g. NOCS\'s unit-cube '
                             'canonicalization). Mutually exclusive with --live_mu_3d.')
    parser.add_argument('--geodesic', action='store_true',
                        help='Both d_2D and d_3D become GEODESIC distances instead of Euclidean '
                             '-- 2D constrained to travel through the object mask (any segment, '
                             'not just the center\'s own), 3D constrained to the mesh surface. '
                             'Requires --geo_table_3d_path/--geo_table_2d_path from '
                             'analysis/build_geodesic_tables.py. Normalizes by GEODESIC diameters '
                             '(not bbox diagonals -- see modules/relational_loss.py\'s geodesic-'
                             'variant docstring for the SHIC-inspired design and why a naive '
                             'Euclidean->geodesic swap would silently kill the gradient). '
                             'Mutually exclusive with --bbox_norm/--live_mu_3d.')
    parser.add_argument('--geo_table_3d_path', type=str,
                        default='./experiments/guitar/geodesic_tables/guitar_geo_table_3d.pth',
                        help='--geodesic only: vertex-to-vertex geodesic table from '
                             'analysis/build_geodesic_tables.py')
    parser.add_argument('--geo_table_2d_path', type=str, default=None,
                        help='--geodesic and/or --cyc_weight>0: defaults to '
                             './experiments/guitar/geodesic_tables/<mask_basename>_geo_table_2d_seed<seed>.pth')
    parser.add_argument('--distance_weight', type=str, default=None, choices=['far', 'near'],
                        help='Reweight each sampled pair\'s Huber residual by its own '
                             'norm_d_2D before averaging, instead of every pair counting '
                             'equally (off by default). \'far\': pairs distant in the image '
                             'get more say in the loss. \'near\': the opposite (weight '
                             '1/norm_d_2D, floored). Composes with any normalization mode '
                             'above -- orthogonal concern, applied after normalization.')
    parser.add_argument('--mu_3d_scale', type=float, default=1.0,
                        help='Divide norm_d_3D by this AFTER normalization (1.0, the default, '
                             'is a no-op -- byte-identical to not passing this flag). '
                             'Mathematically identical to scaling mu_3D_reference/mu_3D_live by '
                             'the same value -- kept as a separate explicit knob so --bbox_norm\'s '
                             'reference stays a literal, honest bounding-box diagonal. A value > '
                             '1 raises the effective 3D-distance target the model has to satisfy, '
                             'pushing mapped points further apart in 3D for the same normalized '
                             'residual -- Tal\'s 2026-08-12 hypothesis for counteracting excessive '
                             'convergence (the headstock-drift family of problems). Composes with '
                             '--distance_weight and any normalization mode above.')
    parser.add_argument('--relational_weight', type=float, default=1.0,
                        help='Weight for the base relational term: loss = relational_weight * '
                             'relational_loss(...) + cyc_weight * cycle_consistency_loss(...) + '
                             'balance_weight * vertex_balance_loss(...). 1.0, the default, is a '
                             'no-op -- byte-identical to not passing this flag. Exists so '
                             '--relational_weight/--cyc_weight/--balance_weight can be swept '
                             'independently (alpha/beta/gamma) instead of the auxiliary terms '
                             'being the only tunable side of the combination.')
    parser.add_argument('--balance_weight', type=float, default=0.0,
                        help='Weight (beta) for an auxiliary vertex-usage load-balancing term: '
                             'loss = relational_weight * relational_loss(...) + balance_weight * '
                             'vertex_balance_loss(...) (+ cyc_weight * cycle_consistency_loss(...) '
                             'if that is also on). Off by default: 0.0 is an exact no-op. '
                             'Tal\'s 2026-08-15 request, after cycle_consistency_loss (both '
                             'variants) was parked without fixing the collapse problem: "start '
                             'from the simplest approach that could yield results." Adapted from '
                             'the Switch Transformer\'s additive MoE load-balancing auxiliary loss '
                             '(arXiv 2101.03961) -- a completely different, independent mechanism '
                             'from cyc_loss\'s reciprocity check, chosen because it is purely '
                             'additive (changes nothing about soft_map_to_vertices or any '
                             'existing loss, so it can be swept and abandoned exactly as cheaply '
                             'as cyc_weight was) rather than SwAV\'s stricter but much more '
                             'invasive Sinkhorn-Knopp equipartition, which would mean rewriting '
                             'the shared assignment mechanism itself. See '
                             'modules/vertex_balance_loss.py for the full design and the exact '
                             'formula as checked against the paper. '
                             'PARKED as of 2026-08-15, Tal\'s call: leave this at 0.0. Swept at '
                             '1e-4/3e-4/1e-3 (with --geodesic --mu_3d_scale 1) -- all increased '
                             'vertex coverage over baseline while keeping drift healthy (unlike '
                             'cyc_weight), but Tal\'s visual review found correspondence quality '
                             'WORSE despite the wider spread -- more vertices used, not more '
                             'correctly. See modules/vertex_balance_loss.py\'s CONCLUSION note.')
    parser.add_argument('--cyc_weight', type=float, default=0.0,
                        help='Weight for a cycle-consistency term: loss = relational_loss(...) + '
                             'cyc_weight * cycle_consistency_loss(...) -- same additive lambda-'
                             'weighted pattern train_mapping_loss.py uses for lambda_map/'
                             'lambda_repel. Off by default: 0.0 is an exact no-op. Motivated by '
                             'Tal\'s 2026-08-14 measurement (only 603/4036 vertices, 14.9%%, '
                             'receive any pixel mapping at all; top 1%% of covered vertices '
                             'absorb 10.4%% of all pixels) -- penalizes many pixels collapsing '
                             'onto the same vertex via a differentiable reciprocity check '
                             '(soft-to-hard forward pick, then a geodesic-distance-weighted '
                             'expected round trip back), restricted to this run\'s fixed centers '
                             'for both legs -- see modules/cycle_consistency_loss.py for the '
                             'full design (styled after both modules/best_segmentation_buddies.py'
                             '\'s find_bsb and SHIC\'s L_cyc, arXiv 2407.18907, with geodesic '
                             'distance substituted for Euclidean). Requires a 2D geodesic table '
                             '(analysis/build_geodesic_tables.py, see --geo_table_2d_path) even '
                             'when --geodesic itself is off. Its round-trip distance is '
                             'normalized by the same geodesic diameter relational_loss_geodesic '
                             'uses (see train()\'s own cyc_geo_diag_2d division and '
                             'modules/cycle_consistency_loss.py\'s docstring) so it lands in '
                             'roughly the same [0,~1] range as relational_loss -- start a sweep '
                             'at 1.0 (equal footing with --relational_weight\'s own default 1.0) '
                             'before trying more SHIC-style unequal weights, and watch '
                             'mu_3D_live/renders, not just vertex coverage, since this loss only '
                             'checks reciprocity, not geometric correctness. '
                             'PARKED as of 2026-08-15, Tal\'s call: leave this at 0.0 -- neither '
                             'this (hard-snap) nor --cyc_soft (fully-soft) ever beat the no-cyc '
                             'baseline on both drift and correspondence quality, and the root '
                             'cause traced back to this project having no SHIC-style absolute '
                             'per-pixel-to-vertex anchor at all, not to either implementation\'s '
                             'mechanics -- see modules/cycle_consistency_loss.py\'s CONCLUSION '
                             'note and PROJECT_NOTES.md (2026-08-15) before resuming this sweep.')
    parser.add_argument('--cyc_soft', action='store_true',
                        help='Use the fully-soft, SHIC-faithful cycle-consistency loss '
                             '(modules/cycle_consistency_loss_soft.py) instead of the default '
                             'hard-snap version (modules/cycle_consistency_loss.py). Checked '
                             'directly against SHIC (arXiv 2407.18907): the hard-snap version '
                             'collapses the forward pixel->vertex leg to a single argmin index '
                             'under no_grad, which SHIC never does (its p(xk|u)/p(v|xk) stay '
                             'full softmax distributions throughout L_cyc) -- and this project\'s '
                             'own 2026-08-14/15 training logs show the hard-snap forward leg '
                             'colliding constantly (refs=196-334/4036 out of 2048 centers), which '
                             'makes colliding pixels computationally indistinguishable to the '
                             'loss (identical vf_xk row -> identical gradient), a plausible reason '
                             'that sweep never produced a clean win. Only takes effect when '
                             '--cyc_weight > 0. See modules/cycle_consistency_loss_soft.py for '
                             'the full design. TESTED 2026-08-15 at cyc_weight 0.01/0.3 and found '
                             'WORSE than the hard-snap version on both coverage and drift, not '
                             'better -- the hard-snap bug was real but wasn\'t what stood between '
                             'this loss and being useful; see that module\'s CONCLUSION note. '
                             '--cyc_weight is parked at 0.0 as of this finding.')

    # Optimization -- same discipline as train_mapping_loss.py.
    parser.add_argument('--num_iters', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_eps', type=float, default=1e-12,
                        help="far below torch's 1e-8 default -- see "
                             'train_mapping_loss.py\'s identical flag for why')
    parser.add_argument('--checkpoint_every', type=int, default=25)
    parser.add_argument('--stop_at_drift', type=float, default=None,
                        help='stop early if mean cosine vs phase-1 features drops below '
                             'this (e.g. 0.9). Off by default.')

    parser.add_argument('--seed', type=int, default=0,
                        help='must match the phase-1 seed -- it is what restores the '
                             'Fourier basis (see load_phase1_encoder)')
    parser.add_argument('--train_seed', type=int, default=0,
                        help='seeds the TRAINING-time RNG: the per-step neighbour resampling in '
                             'sample_in_segment_neighbors/sample_out_segment_neighbors, which was '
                             'unseeded until 2026-08-29 and is a real source of run-to-run spread '
                             '(SEPARATE from --seed, which must keep matching phase-1). Change '
                             'this, not --seed, to repeat a recipe and measure its noise floor.')
    parser.add_argument('--out_dir', type=str,
                        default='./experiments/guitar/encoder_dino_relational')
    args = parser.parse_args()

    if args.init_encoder_checkpoint is None:
        args.init_encoder_checkpoint = os.path.join(args.init_encoder_dir,
                                                    'encoder_checkpoint.pth')
    train(args)
