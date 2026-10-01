"""Leg Collision comparison, horse 164135 (Tal, 2026-09-28): ONE shared 3D render + the back-projection half per run, alpha 0.5."""
import os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
H = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = 'cycv2T0.05_c1.676e-05_retstride3_loraW_all_r4_lr1e-3_T0.01_seed0_mulr5e-2'
BP = 'backproj_lvis164135/lvis164135_OURS_soft_backproj_side_by_side.png'
runs = [('SHIC argmax', 'ZS_phase1/backproj_lvis164135/shic/lvis164135_SHIC_argmax_backproj_side_by_side.png'),
        ('ZS (phase-1)', 'ZS_phase1/' + BP),
        ('iso per-seg', 'mu_learn_perseg_%s/alpha0.5/%s' % (S, BP)),
        ('dir global 16in/0out (no cross-seg)', 'mu_globalaniso_perseg_%s/alpha0.5/%s' % (S, BP)),
        ('dir global 8in/8out (cross-seg)', 'mu_globalaniso_perseg_%s_in8out8/alpha0.5/%s' % (S, BP)),
        ('dir per-seg', 'mu_aniso_perseg_%s/alpha0.5/%s' % (S, BP)),
        ('hybrid 8in/8out', 'mu_anisoxglobal_perseg_%s_in8out8/alpha0.5/%s' % (S, BP)),
        ('two-stage (switch at step 150)', 'mu_globalthenaniso_perseg_%s_in8out8_sw150/alpha0.5/%s' % (S, BP))]
ims = [Image.open(os.path.join(H, p)) for _, p in runs]
w = ims[0].size[0] // 2
fig, axs = plt.subplots(3, 3, figsize=(18, 19))
axs[0, 0].imshow(Image.open(os.path.join(H, '../colored_mesh/horse_checker16_uv_azim0.png'))); axs[0, 0].set_title('3D mesh, azim 0 (the side seen in the photo)', loc='left', fontweight='bold')
for ax, (lab, _), im in zip(axs.flat[1:], runs, ims):
    ax.imshow(im.crop((w, 0, im.size[0], im.size[1]))); ax.set_title(lab, loc='left', fontweight='bold')
for ax in axs.flat: ax.axis('off')
fig.suptitle('Leg Collision -- horse LVIS 164135, soft back-projection, LORA-W alpha 0.5 + cycle', fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.96)); out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backproj_7runs_alpha0.5.png')
plt.savefig(out, dpi=90); print('saved', out)
