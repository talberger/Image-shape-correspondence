"""
Model wrappers for feature extraction from large pre-trained vision models.
"""
from abc import abstractmethod, ABC

import numpy as np
import torch
from torch.nn import functional as F
import torchvision.transforms as T


class ModelWrapper(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, interpolation, images):
        pass

    @abstractmethod
    def patch_size(self):
        pass


class DINOWrapper(ModelWrapper):
    def __init__(self, device=None, small=False, target_size=224):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Tal edit start ---
        # Pinned to a local checkout of dinov2 @ 9c7e324 (2023-10-27, "Add new
        # backbones trained with registers") — the earliest commit with the
        # *_reg backbones, from well before upstream `main` introduced
        # `float | None` syntax (PEP 604, Python 3.10+ only) into
        # layers/attention.py and layers/block.py, which breaks under our
        # Python 3.9 env. Original code called
        # torch.hub.load('facebookresearch/dinov2', ...), which pulls
        # upstream `main` and would crash on import.
        dinov2_local_repo = '/workspace/.cache/dinov2_git'
        if not small:
            self.model = torch.hub.load(dinov2_local_repo, 'dinov2_vitg14_reg', source='local').to(device)
        else:
            self.model = torch.hub.load(dinov2_local_repo, 'dinov2_vitl14_reg', source='local').to(device)
        # --- Tal edit end ---
        self.model.eval()

        self.target_size = target_size
        
        self.image_transforms = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @staticmethod
    def pad(images):
        # image is a tensor of shape [B, C, H, W]
        # padding is a tuple of (left, right, top, bottom)

        target_size = max(*images.shape[2:])
        
        assert (
            len(images.shape) == 4
            and images.shape[1] == 3
            ), f"the images must be with shape BCHW."
        
        # Pad
        h, w = images.shape[-2:]
        padh = target_size - h
        padw = target_size - w
        images = F.pad(images, (0, padw, 0, padh))

        return images
    
    def postprocess_features(self, features, input_size, original_size,
                             fix_grid_alignment=False, patch_stride=None):
        """
        Remove padding and upscale features to the original image size.

        Arguments:
          features (torch.Tensor): Batched embedding features, in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        # --- Tal edit start (2026-08-27): optional patch-grid alignment fix ---
        # The default branch below resizes the n x n token grid straight to target_size, which
        # implicitly assumes cell i is centred at (i + 0.5) * target_size / n. The TRUE centre of a
        # token is patch_size/2 + i * stride. Those two agree exactly when stride == patch_size
        # (16 cells of 14 px -> 14i + 7 either way), which is why this has always been correct in
        # production -- but they diverge for any smaller stride (the overlapping-patch / "stride
        # trick" setting), producing a STRETCH about the image centre: measured on the 1080x810
        # photo at stride 7, up to 22 px (2.06%) at the edges, and 3.11% on the 64x64 render
        # targets. See demo/guitar/features_analyze/problem_a_alignment.png.
        #
        # fix_grid_alignment=True resizes instead to exactly the extent the cells cover --
        # (n-1)*stride + 1 samples spanning patch_size/2 .. patch_size/2 + (n-1)*stride -- and
        # replicate-pads the patch_size/2 margins. Correct for ANY stride. Left OFF by default so
        # every existing run stays bit-identical; pass it explicitly.
        if fix_grid_alignment:
            k = self.patch_size()
            s = k if patch_stride is None else patch_stride
            n_h, n_w = features.shape[-2:]
            span_h, span_w = (n_h - 1) * s + 1, (n_w - 1) * s + 1
            features = F.interpolate(features, (span_h, span_w), mode="bilinear", align_corners=True)
            before = k // 2
            features = F.pad(features, (before, self.target_size - span_w - before,
                                        before, self.target_size - span_h - before),
                             mode="replicate")
        else:
            features = F.interpolate(features, (self.target_size, self.target_size), mode="bilinear", align_corners=False)
        # --- Tal edit end ---
        features = features[..., :input_size[0], :input_size[1]]
        features = F.interpolate(features, original_size, mode="bilinear", align_corners=False)
        return features
    
    def forward(self, images):
        images = self.image_transforms(images)
        out = self.model.forward_features(images)
        features = out['x_norm_patchtokens']          
        N, num_patches, C = features.shape
        features = torch.permute(features, (0, 2, 1))
        features = features.view(N, C, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
        return features

    def patch_size(self):
        return 14
