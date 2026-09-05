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


class _EarlyExit(Exception):
    """Raised out of the up_blocks[1] forward hook to stop the UNet as soon as the one feature map
    we want exists. See SDFeatureExtractor(early_exit=...)."""


class SDFeatureExtractor:
    """One SD UNet up_blocks[1] feature map (1280ch) per image, via a single
    forward pass at a fixed light-noise timestep (t=100), unconditional
    (empty prompt), no ControlNet.

    early_exit (default True, added 2026-08-28): stop the UNet the moment up_blocks[1] has produced
    the feature map, instead of running up_blocks[2], up_blocks[3] and conv_out and discarding all
    of them. The returned tensor is BIT-IDENTICAL either way -- verified at 0.000e+00 max abs
    difference with the noise seed held fixed -- because nothing downstream of the hook can affect
    what the hook already captured. Measured on an RTX 2000 Ada: 671 -> 470 ms per view, i.e. ~30%
    off the SD half, which is ~80% of the whole phase-1 precompute. Pass early_exit=False to get
    the old full forward back (only useful for debugging; the output is the same).

    noise_seed (default 0, added 2026-08-28): base seed for the latent noise, so extraction is
    REPRODUCIBLE. Before this the noise was unseeded and no two runs produced the same features;
    see _noise_like() for the measured cost of that. noise_seed=None restores the old behaviour.
    """

    def __init__(self, device, dtype=torch.float16, early_exit=True, noise_seed=0):
        self.device = device
        self.dtype = dtype
        self.noise_seed = noise_seed
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
        self.early_exit = early_exit
        self.unet.up_blocks[SD_UP_BLOCK_INDEX].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._captured = out[0] if isinstance(out, tuple) else out
        if self.early_exit:
            raise _EarlyExit

    def _generator(self, noise_index):
        """One generator per image, seeded from (noise_seed + noise_index), used for BOTH random
        draws in __call__. See the 2026-08-28 note in PROJECT_NOTES.md.

        There are TWO sources of randomness here, and the first one is easy to miss:
          1. `vae.encode(image).latent_dist.sample()` -- the VAE encoder returns a DIAGONAL GAUSSIAN
             and this samples from it. Small relative to the mean (~0.2% of the feature range) but
             genuinely random.
          2. `torch.randn_like(latents)` -- the diffusion noise added at t=100. The large one.
        Both were unseeded until 2026-08-28, which made feature extraction irreproducible.
        Measured consequence: re-running the identical recipe end to end moved the ground-truth
        eval by 0.22 / 0.19 / 0.02 percentage points -- as large as, or larger than, the differences
        between the recipes being compared. Seeding only (2) is NOT enough; that was tried first and
        the verification caught it still drifting by 3.1e-01.

        Seeded per IMAGE INDEX rather than by advancing one generator call-by-call, because the
        precompute loop skips views whose cache file already exists -- a resumed run would otherwise
        draw a different noise sequence than a fresh one.

        Returns None when noise_seed is None, which restores the old unseeded behaviour.
        """
        if self.noise_seed is None:
            return None
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.noise_seed) + int(noise_index))
        return g

    @torch.no_grad()
    def __call__(self, image_01, noise_index=0):
        """image_01: [1, 3, H, W] float in [0, 1] (same convention
        encoder.py's preprocess_image/DINOWrapper callers already use), H
        and W already a multiple of 8 (the VAE's downsampling factor) --
        caller's job, same division of responsibility as DINOWrapper.pad.

        noise_index: which view/image this is. The latent noise is drawn from a dedicated generator
        seeded with (noise_seed + noise_index), so view i gets the SAME noise no matter how many
        views ran before it -- which matters because the precompute loop skips views whose cache
        file already exists, so a resumed run would otherwise draw a different noise sequence than
        a fresh one. Single-image callers can leave it at 0.

        Returns the raw up_blocks[1] feature map, [1, 1280, H/16, W/16].
        """
        image = (image_01.to(self.device, self.dtype) * 2) - 1  # [0,1] -> [-1,1]
        g = self._generator(noise_index)
        latents = self.vae.encode(image).latent_dist.sample(generator=g) * self.vae.config.scaling_factor
        t = torch.tensor([SD_TIMESTEP], device=self.device, dtype=torch.long)
        noise = (torch.randn_like(latents) if g is None else
                 torch.randn(latents.shape, generator=g, device=latents.device, dtype=latents.dtype))
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        self._captured = None
        try:
            _ = self.unet(noisy_latents, t, encoder_hidden_states=self._empty_text_emb).sample
        except _EarlyExit:
            pass  # up_blocks[1] captured; everything after it is discarded anyway
        return self._captured


def sd_squash(image_t, res=SD_INPUT_RES):
    """The PRE-2026-08-27 framing: squash the raw image to res x res, ignoring aspect ratio.

    Restored as an explicit option 2026-08-31 at Tal's request. sd_square() (below) replaced this
    because it aligns SD's frame with DINO's, and it measured better on every eval metric at T=1
    -- but across the whole 2026-08-30 investigation the back-projection Tal judged best is still
    the one built on THIS framing, and neither of the two proposed explanations (edge padding,
    fusion resolution) survived testing. So the framing is now a knob rather than a fix, and the
    stride-7 + gridalign recipe gets tested on both.

    Note this is NOT the same as sd_frame_aspect_nopad: that one PRESERVES the aspect ratio and
    just drops the padding. This one distorts the image, which is the whole point -- it is what
    produced the features in demo/guitar/features_analyze/dino_sd_poc/.

    image_t: [B, 3, H, W] float in 0..1.
    """
    return F.interpolate(image_t, size=(res, res), mode='bilinear', align_corners=False)


def sd_square(image_t, res=SD_INPUT_RES):
    """Aspect-preserving resize + zero-pad to res x res, so SD sees the SAME frame DINO does.

    Added 2026-08-27. Before this, SD was fed `interpolate(image, (512, 512))` -- the raw image
    SQUASHED to a square -- while DINO gets encoder.preprocess_image (resize longest side, then
    zero-pad to a square). On a non-square photo the two feature grids therefore covered different
    extents, and fuse_dino_sd concatenated them as if they did not: measured on guitar_01.png
    (1080x810) the object's centroid sat 2.45 of 16 grid columns apart, ~165 px of the 810 px
    width. This also removes the aspect DISTORTION inside SD's own forward pass, not just the
    placement error.

    For a SQUARE input this is bit-identical to the old call (the resize is the same and the pad is
    zero-width), so every phase-1 render cache stays valid -- renders are square.

    image_t: [B, 3, H, W] float in 0..1.
    """
    h, w = image_t.shape[-2:]
    scale = res / max(h, w)
    img = F.interpolate(image_t, size=(round(h * scale), round(w * scale)),
                        mode='bilinear', align_corners=False)
    return F.pad(img, (0, res - img.shape[-1], 0, res - img.shape[-2]))


def sd_square_replicate(image_t, res=SD_INPUT_RES):
    """sd_square(), but the pad REPLICATES the edge pixel instead of writing zeros.

    Added 2026-09-03 at Tal's request. Motivation, measured that day on guitar_01: sd_square's ZERO
    padding contaminates the real columns next to it, because SD's UNet attention is global. SD's
    mean per-column feature norm runs ~210 across the object, then decays 204 -> 180 -> 169 -> 160
    -> 143 -> 142 -> 145 approaching the pad and SPIKES to 300/277 on the padded columns themselves.
    The squash framing shows no such profile (flat 188-217 across all 32 columns), which is a large
    part of why sd_square's clean pre-fuse geometry has never translated into better final results:
    it trades a geometric misalignment for a feature-quality defect over ~6 of 24 real columns.

    Replicating the edge pixel keeps sd_square's frame EXACTLY -- same scale, same content
    placement, so the DINO/SD grid alignment is unchanged and bit-identical on square inputs (the
    pad is zero-width, so every phase-1 render cache stays valid, same as sd_square) -- while
    removing the hard black boundary that produces the artifact.

    NOTE: DINO still ZERO-pads (encoder.preprocess_image). This changes the SD branch only, which is
    where the artifact was measured. On square renders both are no-ops.

    image_t: [B, 3, H, W] float in 0..1.
    """
    h, w = image_t.shape[-2:]
    scale = res / max(h, w)
    img = F.interpolate(image_t, size=(round(h * scale), round(w * scale)),
                        mode='bilinear', align_corners=False)
    # F.pad's 'replicate' needs the pad spec applied one axis at a time for 4D inputs in this
    # torch version; doing both at once is what raises "Padding size should be less than...".
    img = F.pad(img, (0, res - img.shape[-1], 0, 0), mode='replicate')
    return F.pad(img, (0, 0, 0, res - img.shape[-2]), mode='replicate')


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


def dino_only(dino_features):
    """The DINO half of fuse_dino_sd(), with the SD half simply absent.

    Added 2026-09-01 at Tal's request, for a DINO vs DINO+SD ablation ON THE CURRENT RECIPE.
    Every DINO-only run already in experiments/guitar/ predates DFD barycentric distillation, the
    L2-normalize fix and the current phase-2, so none of them isolates SD -- and the photo
    features every 2026-08-20..26 DINO+SD run trained on were deleted and cannot be regenerated
    (SD's noise was unseeded until 2026-08-28), so the historical comparison cannot be repaired.

    Deliberately keeps fuse_dino_sd's `F.normalize(...) * 0.5` on the DINO branch rather than
    returning raw or unit-norm features: this way the output is bit-exactly the fused tensor with
    the 1280 SD channels deleted, and DINO's own treatment is identical between the two arms. The
    0.5 is a no-op for every consumer that renormalizes (soft_map_to_vertices' cosine similarity
    does), so keeping it costs nothing and leaves the ablation with exactly ONE difference.

    dino_features: [1, C_dino, h, w] raw patch grid, e.g. 1024 x 16 x 16
    Returns [1, C_dino, h, w], float32.
    """
    return F.normalize(dino_features.float(), dim=1) * 0.5


# --- Tal edit, 2026-08-30: decoupled framing + common-grid fusion -------------------------------
# WHY. fuse_dino_sd() resizes SD onto DINO's RAW patch grid, which forces two things we do not
# actually want:
#   (a) SD is DOWNSAMPLED to DINO's resolution. At stride 14 that is 32x32 -> 16x16, i.e. half of
#       SD's spatial detail is discarded on every run. (At stride 7 DINO's grid is 31x23, so SD
#       maps ~1:1 and there is nothing to lose -- this only bites at stride 14.)
#   (b) both branches must share one padding convention, because the fuse aligns whole PADDED
#       squares rather than the valid image regions inside them. That is what forced sd_square()
#       to imitate DINO's frame, and it is why SD has to be zero-padded at all.
# The functions below let each branch be framed however suits it, crop each to its own valid
# region, and fuse on a common grid afterwards. Nothing here changes fuse_dino_sd() or any
# existing call site; both are additive.


def sd_frame_aspect_nopad(image_t, res=SD_INPUT_RES, multiple=8):
    """Aspect-preserving resize with NO padding, both sides rounded to a multiple of `multiple`.

    SD's VAE requires spatial dims divisible by 8; it does not require a square. Feeding the true
    aspect with no padding removes the artificial black border entirely -- and with it the boundary
    artifact measured on 2026-08-30 (SD's per-column feature norm decayed 212 -> 180 approaching the
    padding, and spiked to ~300 on the padded edge itself, contaminating the real columns because
    the UNet's attention is global).

    image_t: [B, 3, H, W] float in 0..1. Returns [B, 3, h', w'] with h', w' multiples of `multiple`.
    """
    h, w = image_t.shape[-2:]
    scale = res / max(h, w)
    nh = max(multiple, int(round(h * scale / multiple)) * multiple)
    nw = max(multiple, int(round(w * scale / multiple)) * multiple)
    return F.interpolate(image_t, size=(nh, nw), mode='bilinear', align_corners=False)


def crop_valid(grid, content_hw, framed_hw):
    """Crop a feature grid computed on a padded square down to the cells that are real image.

    grid:        [C, gh, gw] or [B, C, gh, gw]
    content_hw:  size of the real content inside the framed input
    framed_hw:   size of the framed (padded) input the grid was computed from
    Padding is right/bottom, matching both DINOWrapper.pad and sd_square.
    """
    gh, gw = grid.shape[-2:]
    vh = max(1, round(gh * content_hw[0] / framed_hw[0]))
    vw = max(1, round(gw * content_hw[1] / framed_hw[1]))
    return grid[..., :vh, :vw]


def fuse_dino_sd_common_grid(sd_valid, dino_valid, out_hw):
    """Diff3F-style fusion, but on a COMMON grid instead of on DINO's raw grid.

    Both inputs must already be cropped to their valid (real-image) region, so they describe the
    same extent and only their resolutions differ. Each is resized to out_hw, L2-normalized along
    channels, and concatenated with the same fixed 0.5/0.5 weights fuse_dino_sd uses.

    sd_valid:   [1, C_sd, h, w]    valid region of SD's grid
    dino_valid: [1, C_dino, h, w]  valid region of DINO's grid (or DINO features already at out_hw)
    out_hw:     (H, W) common grid, chosen large enough not to throw away either side
    """
    sd_r = F.interpolate(sd_valid.float(), size=out_hw, mode='bilinear', align_corners=False)
    dino_r = F.interpolate(dino_valid.float(), size=out_hw, mode='bilinear', align_corners=False)
    return torch.cat([F.normalize(sd_r, dim=1) * 0.5, F.normalize(dino_r, dim=1) * 0.5], dim=1)
