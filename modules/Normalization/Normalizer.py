import torch


class Normalizer:
    @classmethod
    def get_bounding_box_normalizer(cls, x):
        shift = torch.mean(x, dim=0)
        scale = torch.max(torch.norm(x-shift, p=1, dim=1))
        return Normalizer(scale=scale, shift=shift)

    @classmethod
    def get_bounding_sphere_normalizer(cls, x):
        shift = torch.mean(x, dim=0)
        scale = torch.max(torch.norm(x-shift, p=2, dim=1))
        return Normalizer(scale=scale, shift=shift)

    @classmethod
    def get_bounding_sphere_normalizer_bbox(cls, x):
        """Bounding sphere centered on the BOUNDING-BOX centre instead of the vertex centroid.

        Added 2026-09-02. The centroid version above centres on `mean(vertices)`, which depends on
        how a surface is SAMPLED, not only on its shape -- so two triangulations of the same object
        land in different frames. Measured on the guitar: centroid-centred scale 1.00000 (original)
        vs 1.13202 (isotropic remesh), i.e. the remeshed guitar renders ~12% smaller in every view,
        purely because the headstock stopped holding 41% of the vertices. Bounding-box centring is
        sampling-independent: 0.90350 vs 0.90256 for the same pair, a 0.1% difference that is just
        the remesh's own sub-voxel geometry change.

        Not the default: every artifact in this project predating 2026-09-02 was built under the
        centroid normalizer, and silently reframing them would invalidate all of it. Selected per
        mesh via a `<mesh>.normalizer.json` sidecar -- see modules/utils.loadmesh.
        """
        shift = (torch.min(x, dim=0).values + torch.max(x, dim=0).values) / 2
        scale = torch.max(torch.norm(x-shift, p=2, dim=1))
        return Normalizer(scale=scale, shift=shift)

    def __init__(self, scale, shift):
        self._scale = scale
        self._shift = shift

    def __call__(self, x):
        return (x-self._shift) / self._scale

    def get_de_normalizer(self):
        inv_scale = 1 / self._scale
        inv_shift = -self._shift / self._scale
        return Normalizer(scale=inv_scale, shift=inv_shift)