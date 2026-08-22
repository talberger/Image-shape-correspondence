"""
DINO + Stable Diffusion fused features (Diff3F-style: no PCA, no learned
network -- L2-normalize each side, concat with fixed weights). New,
additive module -- does not modify modules/dino.py; imports DINOWrapper
directly rather than duplicating it.

Why not sd-dino's own PCA fusion: their PCA is fit per image PAIR (they
only ever compare 2 photos, fit a throwaway basis, use it once, discard
it) -- doesn't fit a pipeline needing ~1000 renders + 1 photo to share ONE
stable feature space. Diff3F (lukasuz/Diffusion-3D-Features, CVPR 2024) do
the same task as this project -- distilling DINO+SD features onto an
untextured mesh from many rendered views -- and sidestep the fitting
problem entirely: every image goes through the same frozen models
independently, so cross-image consistency is automatic. Their fusion
(diff3f.py::get_features_per_vertex): L2-normalize each side, concat with
fixed 0.5/0.5 weights. Adopted here, with sd-dino's simpler single-pass SD
extraction (not Diff3F's own heavier ControlNet-img2img pipeline, which
this project doesn't need -- we only want the intermediate feature, not a
generated image).

SD block choice: unet.up_blocks[1] -- matches Diff3F's own modified
unet_2d_condition.py exactly (`if extract_layer and i==1: f_map = sample[0]`),
confirmed via a Stage-0 smoke test: [1280, 32, 32] at 512px input, t=100.
Tal's explicit call to match this over the same-resolution-as-DINO
up_blocks[0] (also 1280ch, 16x16) shortcut, since up_blocks[1] is the one
actually validated by a real reference implementation.
"""
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

SD_MODEL_ID = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
SD_TIMESTEP = 100
SD_UP_BLOCK_INDEX = 1
SD_INPUT_RES = 512


class SDFeatureExtractor:
    """One SD UNet up_blocks[1] feature map (1280ch) per image, via a single
    forward pass at a fixed light-noise timestep (t=100), unconditional
    (empty prompt), no ControlNet.
    """

    def __init__(self, device, dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.unet = UNet2DConditionModel.from_pretrained(
            SD_MODEL_ID, subfolder='unet', torch_dtype=dtype).to(device).eval()
        self.vae = AutoencoderKL.from_pretrained(
            SD_MODEL_ID, subfolder='vae', torch_dtype=dtype).to(device).eval()
        text_encoder = CLIPTextModel.from_pretrained(
            SD_MODEL_ID, subfolder='text_encoder', torch_dtype=dtype).to(device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder='tokenizer')
        self.scheduler = DDPMScheduler.from_pretrained(SD_MODEL_ID, subfolder='scheduler')

        with torch.no_grad():
            empty_ids = tokenizer([''], padding='max_length', truncation=True,
                                  max_length=tokenizer.model_max_length,
                                  return_tensors='pt').input_ids.to(device)
            self._empty_text_emb = text_encoder(empty_ids)[0]
        del text_encoder, tokenizer

        self._captured = None
        self.unet.up_blocks[SD_UP_BLOCK_INDEX].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._captured = out[0] if isinstance(out, tuple) else out

    @torch.no_grad()
    def __call__(self, image_01):
        """image_01: [1, 3, H, W] float in [0, 1] (same convention
        encoder.py's preprocess_image/DINOWrapper callers already use), H
        and W already a multiple of 8 (the VAE's downsampling factor) --
        caller's job, same division of responsibility as DINOWrapper.pad.

        Returns the raw up_blocks[1] feature map, [1, 1280, H/16, W/16].
        """
        image = (image_01.to(self.device, self.dtype) * 2) - 1  # [0,1] -> [-1,1]
        latents = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor([SD_TIMESTEP], device=self.device, dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        self._captured = None
        _ = self.unet(noisy_latents, t, encoder_hidden_states=self._empty_text_emb).sample
        return self._captured


def fuse_dino_sd(sd_features, dino_features):
    """Diff3F-style fusion (diff3f.py::get_features_per_vertex): L2-normalize
    each side along the channel dim, resize SD onto DINO's spatial grid,
    concat with fixed 0.5/0.5 weights.

    sd_features:   [1, C_sd, h_sd, w_sd]       raw, e.g. 1280 x 32 x 32
    dino_features: [1, C_dino, h_dino, w_dino] raw patch grid, e.g. 1024 x 16 x 16
    Returns [1, C_sd + C_dino, h_dino, w_dino], float32.
    """
    sd_resized = F.interpolate(sd_features.float(), size=dino_features.shape[-2:],
                               mode='bilinear', align_corners=False)
    sd_norm = F.normalize(sd_resized, dim=1)
    dino_norm = F.normalize(dino_features.float(), dim=1)
    return torch.cat([sd_norm * 0.5, dino_norm * 0.5], dim=1)
