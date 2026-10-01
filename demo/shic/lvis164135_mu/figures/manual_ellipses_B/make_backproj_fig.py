"""Manual ellipses B vs learned, horse 164135 (Tal 2026-09-28): shared azim-0 render + back-projection halves."""
import os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
H = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = 'cycv2T0.05_c1.676e-05_retstride3_loraW_all_r4_lr1e-3_T0.01_seed0_mulr5e-2'
BP = 'backproj_lvis164135/lvis164135_OURS_soft_backproj_side_by_side.png'
runs = [('SHIC argmax', 'ZS_phase1/backproj_lvis164135/shic/lvis164135_SHIC_argmax_backproj_side_by_side.png'),
        ('ZS (phase-1)', 'ZS_phase1/' + BP),
        ('two-stage, LEARNED ellipses', 'mu_globalthenaniso_perseg_%s_in8out8_sw150/alpha0.5/%s' % (S, BP)),
        ('two-stage, MANUAL B ellipses (frozen)', 'mu_globalthenaniso_perseg_%s_in8out8_sw150_manualB/alpha0.5/%s' % (S, BP)),
        ('hybrid, LEARNED ellipses', 'mu_anisoxglobal_perseg_%s_in8out8/alpha0.5/%s' % (S, BP)),
        ('hybrid, MANUAL B + manual global', 'mu_anisoxglobal_perseg_%s_in8out8_manualB/alpha0.5/%s' % (S, BP)),
        ('two-stage + HYBRID stage B, MANUAL', 'mu_globalthenaniso_perseg_%s_in8out8_sw150_hybridB_manualB/alpha0.5/%s' % (S, BP))]
fig, axs = plt.subplots(2, 4, figsize=(24, 12.5)); axs = axs.ravel()
axs[0].imshow(Image.open(os.path.join(H, '../colored_mesh/horse_checker16_uv_azim0.png'))); axs[0].set_title('3D mesh, azim 0', loc='left', fontweight='bold', fontsize=14)
for ax, (lab, p) in zip(axs[1:], runs):
    im = Image.open(os.path.join(H, p)); w = im.size[0] // 2
    ax.imshow(im.crop((w, 0, im.size[0], im.size[1]))); ax.set_title(lab, loc='left', fontweight='bold', fontsize=14)
for ax in axs: ax.axis('off')
plt.tight_layout(); out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backproj_hybridB.png'); plt.savefig(out, dpi=80); print('saved', out)
