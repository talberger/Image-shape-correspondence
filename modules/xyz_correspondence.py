"""
Free correspondence-pair extraction from the mesh's own known 3D geometry -- fallback
plan for the 2D feature post-processor (see PROJECT_NOTES.md's left/right symmetry
disambiguation thread; used only if the GeoAware-SC pretrained-network experiment
doesn't transfer to the guitar). Uses the existing xyz_bary_f cache (every foreground
pixel's exact barycentric-interpolated 3D surface point, index-aligned with dino_sd_f)
to derive dense correspondence with no manual annotation:
- cross-view pairs: two pixels (possibly different views) landing on the same real 3D
  surface point -- teaches general correspondence sharpness.
- mirror pairs: a pixel and its geometric mirror-symmetric counterpart (reflect
  mirror_axis -> -mirror_axis, nearest-neighbor match) -- the actual left/right
  disambiguation signal a downstream training loss would use.

New, additive module -- doesn't touch modules/barycentric_targets.py or any training
script. Standalone, no torch-training dependency: importable and testable on its own.
"""
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

from modules.barycentric_targets import BACKGROUND_SENTINEL


def load_all_fg_xyz(encoder_data_dir, render_res=224, n_views=1000):
    """Flattens every view's xyz_bary_f cache into one pool, with provenance so callers
    can index back into dino_sd_f (or any index-aligned feature cache) at the same
    (view, row, col).

    Returns: xyz [N,3] float32, view_id [N] int64, row [N] int64, col [N] int64
    (all numpy arrays).
    """
    xyz_dir = os.path.join(encoder_data_dir, 'xyz_bary_f', str(render_res))
    xyz_chunks, view_chunks, row_chunks, col_chunks = [], [], [], []
    for i in range(n_views):
        d = torch.load(os.path.join(xyz_dir, f'target_xyz_bary_f_{i}.pt'), map_location='cpu')
        xyz_f = d['model_f']  # [3, H, W]
        mask = (xyz_f != BACKGROUND_SENTINEL).any(dim=0)  # [H, W]
        rows, cols = torch.where(mask)
        xyz = xyz_f.permute(1, 2, 0)[mask]  # [n_fg, 3]

        xyz_chunks.append(xyz.numpy())
        view_chunks.append(np.full(len(rows), i, dtype=np.int64))
        row_chunks.append(rows.numpy().astype(np.int64))
        col_chunks.append(cols.numpy().astype(np.int64))

    return (np.concatenate(xyz_chunks).astype(np.float32), np.concatenate(view_chunks),
            np.concatenate(row_chunks), np.concatenate(col_chunks))


def build_cross_view_pairs(xyz, view_id, max_dist, n_pairs=200_000, k_query=8, seed=0):
    """For each foreground sample, its nearest sample from a DIFFERENT view within
    max_dist (same real 3D surface point, different render). Returns (idx_a, idx_b)
    index arrays into the flattened arrays, subsampled to n_pairs if more matches
    exist. One-off preprocessing step (not in any training loop), simplicity favored
    over vectorization speed."""
    tree = cKDTree(xyz)
    dists, idxs = tree.query(xyz, k=k_query)

    idx_a, idx_b = [], []
    for i in range(len(xyz)):
        for d, j in zip(dists[i], idxs[i]):
            if j == i or d > max_dist:
                continue
            if view_id[j] != view_id[i]:
                idx_a.append(i)
                idx_b.append(j)
                break
    idx_a, idx_b = np.array(idx_a, dtype=np.int64), np.array(idx_b, dtype=np.int64)
    return _subsample(idx_a, idx_b, n_pairs, seed)


def build_mirror_pairs(xyz, view_id, mirror_axis=0, max_dist=0.01, min_axis_offset=0.01,
                        same_view_only=True, n_pairs=200_000, seed=0):
    """Reflects xyz[:, mirror_axis], KD-tree-matches each reflected point against the
    real (unreflected) pool, keeps matches within max_dist. Drops points within
    min_axis_offset of the mirror plane (near-symmetry-axis points reflect close to
    themselves -- not a useful disambiguation pair). same_view_only=True (default)
    restricts matches to pairs visible together in one render, which a same-view
    softmax-argmax disambiguation loss needs (the mirror twin must be a candidate
    location in the same target map)."""
    mirrored = xyz.copy()
    mirrored[:, mirror_axis] *= -1
    keep = np.abs(xyz[:, mirror_axis]) > min_axis_offset

    if same_view_only:
        idx_a_parts, idx_b_parts = [], []
        for v in np.unique(view_id):
            sel = np.where((view_id == v) & keep)[0]
            if len(sel) < 2:
                continue
            tree = cKDTree(xyz[sel])
            d, j = tree.query(mirrored[sel], k=1)
            valid = d <= max_dist
            idx_a_parts.append(sel[valid])
            idx_b_parts.append(sel[j[valid]])
        idx_a = np.concatenate(idx_a_parts) if idx_a_parts else np.array([], dtype=np.int64)
        idx_b = np.concatenate(idx_b_parts) if idx_b_parts else np.array([], dtype=np.int64)
    else:
        tree = cKDTree(xyz)
        sel = np.where(keep)[0]
        d, j = tree.query(mirrored[sel], k=1)
        valid = d <= max_dist
        idx_a, idx_b = sel[valid], j[valid]

    ok = idx_a != idx_b
    idx_a, idx_b = idx_a[ok], idx_b[ok]
    return _subsample(idx_a, idx_b, n_pairs, seed)


def _subsample(idx_a, idx_b, n_pairs, seed):
    if len(idx_a) > n_pairs:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(idx_a), size=n_pairs, replace=False)
        idx_a, idx_b = idx_a[sel], idx_b[sel]
    elif len(idx_a) < n_pairs:
        print(f'xyz_correspondence: only {len(idx_a)} pairs available, requested {n_pairs}')
    return idx_a, idx_b
