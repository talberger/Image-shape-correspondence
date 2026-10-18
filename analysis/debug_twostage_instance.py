"""Debug figures for ONE two-stage LVIS instance (Tal, 2026-10-06): back-projection ZS -> end of stage A -> final (+ SHIC),
and the fine-segment map with the learned ellipses (stage-A global ellipse, final per-segment ellipses).

Usage: python analysis/debug_twostage_instance.py --tag lvis184389 --run_dir <kept intermediates> --debug_dir <figures dir>
         --p2 <retrained phase-2 dir with mu_aniso_stageA_end.pth> --shic_backproj <plain SHIC back-projection png>
"""
import argparse, json, math, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_mu_ellipses import mu_dir

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True); ap.add_argument('--run_dir', required=True); ap.add_argument('--debug_dir', required=True)
ap.add_argument('--p2', required=True); ap.add_argument('--shic_backproj', required=True)
ap.add_argument('--shic_json', required=True, help='<tag>_vs_shic.json (SHIC per-point errors)')
ap.add_argument('--key_render', default='demo/shic/colored_mesh/horse_checker16_uv_azim135.png')
ap.add_argument('--arm_file', default='{O}/{arm}/{T}_{arm}_backproj.png')
ap.add_argument('--eval_json', default=None); ap.add_argument('--fine_seg_path', default=None); ap.add_argument('--suffix', default='')
ap.add_argument('--arm_dirs', nargs=3, default=['zs', 'stageA', 'final_rerun'], help='back-projection folders under --debug_dir: ZS, end of stage A, final')
ap.add_argument('--arm_keys', nargs=3, default=['zs', 'stageA', 'final_rerun'], help='the same three arms in the eval json')
ap.add_argument('--zs_eval_json', default=None)
ap.add_argument('--min_frac', type=float, default=0.01, help='segments below this share of the mask get a thin ellipse')
args = ap.parse_args()
T, D, O = args.tag, args.run_dir, args.debug_dir
img = np.asarray(Image.open(f'{D}/{T}.png').convert('RGB'))
gt = torch.load(f'{D}/{T}_gt.pth'); rows, cols = gt['rows'].tolist(), gt['cols'].tolist()
ev = json.load(open(args.eval_json or f'{O}/eval_arms.json'))['arms']
if args.zs_eval_json: ev.setdefault(args.arm_keys[0], json.load(open(args.zs_eval_json))['arms']['zs'])
sh = json.load(open(args.shic_json))

def pts(ax, colour='w'):
    for i, (r, c) in enumerate(zip(rows, cols)):
        ax.plot(c, r, 'o', ms=9, mfc='none', mec='k', mew=3); ax.plot(c, r, 'o', ms=9, mfc='none', mec=colour, mew=1.6)
        ax.annotate(str(i + 1), (c, r), xytext=(6, -6), textcoords='offset points', color=colour, fontsize=11, weight='bold',
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground='k')])
import matplotlib.patheffects  # noqa: E402

# ---- figure 1: back-projection through the stages ----
panels = [('crop + GT points', img, None), ('colour key', np.asarray(Image.open(args.key_render).convert('RGB')), None)]
for name, arm, key in zip(['ZS (phase 1)', 'after stage 1 (global ellipse)', 'final (per-segment ellipses)'], args.arm_dirs, args.arm_keys):
    e = ev[key]['soft_snap']['per_point']
    panels.append((name, np.asarray(Image.open(f'{O}/{arm}/{T}_{arm}_backproj.png').convert('RGB')), e))
panels.append(('SHIC (argmax)', np.asarray(Image.open(args.shic_backproj).convert('RGB')), sh['err_shic']))
fig, axs = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.3))
for ax, (name, im, e) in zip(axs, panels):
    ax.imshow(im); ax.axis('off'); ax.set_title(name, fontsize=12)
    if name != 'colour key': pts(ax)
    if e is not None:
        ax.text(0.5, -0.04, 'error  pt1 %.1f | pt2 %.1f | pt3 %.1f   mean %.1f' % (*e, np.mean(e)), transform=ax.transAxes,
                ha='center', va='top', fontsize=11)
fig.suptitle(f'{T}: back-projection (soft-snap, T=0.01), geodesic error 228-normalized', fontsize=13)
fig.tight_layout(); fig.savefig(f'{O}/{T}_backproj_stages{args.suffix}.png', dpi=130); plt.close(fig)

# ---- figure 2: fine segments + ellipses ----
seg = torch.load(args.fine_seg_path or f'{D}/{T}_fine_segments.pth').numpy(); mask = torch.load(f'{D}/{T}_mask.pth').numpy()
ids = [int(i) for i in np.unique(seg[mask]) if i > 0]; n = {i: int(((seg == i) & mask).sum()) for i in ids}; tot = int(mask.sum())
cmap = plt.get_cmap('tab10'); col = {i: cmap((k) % 10)[:3] for k, i in enumerate(ids)}
ov = img.astype(float) / 255 * 0.35
for i in ids: ov[(seg == i) & mask] = 0.35 * img[(seg == i) & mask] / 255 + 0.65 * np.array(col[i])
unl = mask & (seg == 0); ov[unl] = (1, 1, 1)
A = torch.load(f'{args.p2}/mu_aniso_stageA_end.pth'); B = torch.load(f'{args.p2}/mu_aniso.pth')
R = 0.11 * max(img.shape[:2])                       # pixels for mu = 1
th = np.arange(0, 361, 3)
def curve(p, cy, cx):
    m = np.array([mu_dir(p, t) for t in th]); return cx + R * m * np.sin(np.radians(th)), cy + R * m * np.cos(np.radians(th))
def ref(ax, cy, cx): ax.plot(cx + R * np.sin(np.radians(th)), cy + R * np.cos(np.radians(th)), color='w', lw=1, ls=':')
fig, axs = plt.subplots(1, 3, figsize=(21, 5.6))
ax = axs[0]; ax.imshow(ov); ax.axis('off'); ax.set_title('fine segments (white = mask pixels with no segment)', fontsize=12)
for i in ids:
    ys, xs = np.nonzero((seg == i) & mask)
    ax.annotate('%d' % i, (xs.mean(), ys.mean()), color='w', fontsize=12, weight='bold', ha='center', va='center',
                path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground='k')])
pts(ax, 'yellow')
ax.legend(handles=[matplotlib.patches.Patch(color=col[i], label='seg %d: %d px (%.1f%%)' % (i, n[i], 100 * n[i] / tot)) for i in ids]
          + [matplotlib.patches.Patch(fc='w', ec='k', label='no segment: %d px (%.1f%%)' % (unl.sum(), 100 * unl.sum() / tot))],
          loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False)
cov = img.astype(float) / 255; cov[~mask] *= 0.25                 # stage 1 has ONE ellipse for the whole object -> drawn on the COARSE mask
ax = axs[1]; ax.imshow(cov); ax.contour(mask, [0.5], colors='yellow', linewidths=1.2); ax.axis('off'); cy, cx = np.nonzero(mask)[0].mean(), np.nonzero(mask)[1].mean()
g = A['mu_aniso'][-1].tolist(); ref(ax, cy, cx); ax.plot(*curve(g, cy, cx), color='magenta', lw=2.5)
ax.set_title('end of stage 1, coarse mask: ONE global ellipse  (major %.2f / minor %.2f, major axis %.0f deg from vertical)'
             % (A['major'][-1], A['minor'][-1], A['major_angle_deg'][-1]), fontsize=11)
ax = axs[2]; ax.imshow(ov); ax.axis('off'); ax.set_title('final: per-segment ellipses (thin orange = segment < %.0f%% of the mask)' % (100 * args.min_frac), fontsize=11)
for i in ids:
    ys, xs = np.nonzero((seg == i) & mask); big = n[i] / tot >= args.min_frac
    if big: ref(ax, ys.mean(), xs.mean())
    ax.plot(*curve(B['mu_aniso'][i].tolist(), ys.mean(), xs.mean()), color='red' if big else 'orange', lw=2.5 if big else 0.9)
    ax.annotate('%d' % i, (xs.mean(), ys.mean()), color='w', fontsize=11, weight='bold', ha='center', va='center',
                path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground='k')])
for a in axs: a.set_xlim(-0.5, img.shape[1] - 0.5); a.set_ylim(img.shape[0] - 0.5, -0.5)
fig.tight_layout(); fig.savefig(f'{O}/{T}_segments_ellipses{args.suffix}.png', dpi=130); plt.close(fig)
# ---- figure 3: coarse mask ----
im01 = img.astype(float) / 255
fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
axs[0].imshow(im01); axs[0].set_title('crop'); pts(axs[0])
o = im01.copy(); o[~mask] *= 0.25; axs[1].imshow(o); axs[1].contour(mask, [0.5], colors='yellow', linewidths=1.2)
axs[1].set_title('coarse segment = object mask (%d px, %.1f%% of the crop)' % (tot, 100 * tot / mask.size))
axs[2].imshow(mask, cmap='gray'); axs[2].set_title('mask only')
for a_ in axs: a_.axis('off')
fig.tight_layout(); fig.savefig(f'{O}/{T}_coarse_mask{args.suffix}.png', dpi=110); plt.close(fig)
# ---- figure 4: every fine segment on its own panel ----
allids = [0] + ids; nc = 5; nr = int(np.ceil(len(allids) / nc)); grey = np.repeat(im01.mean(2, keepdims=True), 3, 2) * 0.55 + 0.25
fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr), squeeze=False)
for a_ in axs.ravel(): a_.axis('off')
for a_, i in zip(axs.ravel(), allids):
    s_ = (seg == i) & mask; o = grey.copy(); o[s_] = (1, 0, 0) if i else (0, .8, 1)
    a_.imshow(o, interpolation='nearest'); a_.contour(mask, [0.5], colors='k', linewidths=.5)
    a_.set_title(('seg %d' % i if i else 'mask pixels with NO segment') + ': %d px (%.1f%%)' % (s_.sum(), 100 * s_.sum() / tot), fontsize=12)
fig.tight_layout(); fig.savefig(f'{O}/{T}_segments_one_by_one{args.suffix}.png', dpi=100); plt.close(fig)
print('seg | px | share | final major / minor / angle')
for i in ids: print('%3d | %5d | %5.1f%% | %.2f / %.2f / %3.0f' % (i, n[i], 100 * n[i] / tot, B['major'][i], B['minor'][i], B['major_angle_deg'][i]))
print('global at end of stage 1: %.2f / %.2f / %.0f' % (A['major'][-1], A['minor'][-1], A['major_angle_deg'][-1]))
print('saved', f'{O}/{T}_backproj_stages{args.suffix}.png', f'{O}/{T}_segments_ellipses{args.suffix}.png')
