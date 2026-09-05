import copy
from . import Normalizer


class MeshNormalizer:
    def __init__(self, mesh):
        self._mesh = mesh  # original copy of the mesh
        self.normalizer = Normalizer.get_bounding_sphere_normalizer(self._mesh.vertices)

    def __call__(self):
        # A mesh loaded through a `<mesh>.normalizer.json` sidecar is already in a DELIBERATE frame
        # (modules/utils.loadmesh stamps frame_locked on it). Re-normalizing here would silently
        # undo it: this normalizer re-centres on mean(vertices), which for a bounding-box-centred
        # mesh is not the origin. It IS a no-op for a centroid-normalized mesh -- mean is already 0
        # and max radius already 1 -- which is why color_mesh/setcolor_mesh/arbcolor_mesh have
        # always been able to call it on every colour update for free. It is not free for any other
        # frame. Measured 2026-09-02: one call took the bounding-box-framed guitar's Y-extent from
        # 1.9966 straight back to the centroid frame's 1.5919, and the 1000 phase-1 training
        # renders came out at 55.3% of frame height instead of the intended ~66%.
        if getattr(self._mesh, 'frame_locked', False):
            return self._mesh
        self._mesh.vertices = self.normalizer(self._mesh.vertices)
        return self._mesh

