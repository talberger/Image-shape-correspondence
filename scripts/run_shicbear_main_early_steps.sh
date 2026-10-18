#!/bin/bash
# SHIC-bear, 2026-09-17 (Tal): the cells shrink within the first 25 steps -- look at steps 5/10/15/20/25.
# Re-runs MAIN's exact bear command for 25 steps with --checkpoint_every 5 (no LR schedule, same seed), checks the
# step-25 weights against the original run, then SMALL analysis per step + one comparison figure.
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT

OBJ=./demo/shic/shapes/bear.obj
NAME=shic_bear
MASK=./demo/shic/masks/bear3_click_350_300_sam_mask_1.pth
GEO3D=./experiments/shic_bear/geodesic_tables/shic_bear_geo_table_3d.pth
PIX=demo/shic/features_analyze/sdsquare_fuse16/bear3_pix_sdsquare_fuse16.pth
S=sdsquare_truediag2d_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0
ORIG=experiments/shic_bear/encoder_${S}_phase2
P2=experiments/shic_bear/encoder_${S}_phase2_early_every5
RUN=demo/shic/bear3_${S}_phase2/early_steps

if [ ! -f "$P2/encoder_checkpoint_step25.pth" ]; then
  python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" --init_encoder_dir ./experiments/shic_bear/encoder_dfd_crossview_phase1 \
    --object_mask_path "$MASK" --fine_seg_path ./demo/shic/bear3/bear3_fine_segments.pth --pix_dino_features_path "$PIX" \
    --width 2304 --n_classes 2304 --geodesic --temperature 0.01 --train_seed 0 --geo_diag_2d 967.21 \
    --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 \
    --geo_table_3d_path "$GEO3D" --geo_table_2d_path ./experiments/shic_bear/geodesic_tables/bear3_click_350_300_sam_mask_1_geo_table_2d_seed0.pth \
    --mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs \
    --cyc_weight 1.676e-05 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5 \
    --num_iters 25 --checkpoint_every 5 --out_dir "$P2"
fi

echo "########## reproducibility: step 25 vs the original run ##########"
python - "$ORIG/encoder_checkpoint_step25.pth" "$P2/encoder_checkpoint_step25.pth" <<'EOF'
import sys, torch
a, b = (torch.load(p, map_location='cpu') for p in sys.argv[1:])
a = a['encoder_state_dict']
b = b['encoder_state_dict']
m = max((a[k].float() - b[k].float()).abs().max().item() for k in a if torch.is_tensor(a[k]))
print(f'max |diff| over all weights, step 25 original vs rerun: {m:.3e}')
EOF

CKS=""; OUTS=""
for s in 5 10 15 20 25; do mkdir -p $P2/step$s; CKS="$CKS,$P2/encoder_checkpoint_step$s.pth"; OUTS="$OUTS,$P2/step$s/pred_f.pth"; done
python analysis/export_pred_f_l2norm.py --obj_path $OBJ --name $NAME --checkpoints "${CKS#,}" --outputs "${OUTS#,}" --width 2304 --n_classes 2304

for s in 5 10 15 20 25; do
  P2=$P2/step$s RUN=$RUN/step$s PIX=$PIX READ_T=0.01 LABEL="MAIN recipe, step $s" bash scripts/analyze_shicbear_arm.sh
done
python analysis/topmatch_geo_extent.py --geo_table_3d_path $GEO3D \
  --knn_paths demo/shic/bear3_ZS_sdsquare_fuse16_phase1old/analyze/ours/bear3_click_350_300_sam_mask_1_pixel_to_vertex_nn.pth \
    $(for s in 5 10 15 20 25; do echo $RUN/step$s/analyze/ours/bear3_click_350_300_sam_mask_1_pixel_to_vertex_nn.pth; done) \
  --labels step0_ZS step5 step10 step15 step20 step25 | tee $RUN/proportion_extent.txt

python - "$RUN" <<'EOF'
import sys
from PIL import Image, ImageDraw
run = sys.argv[1]; h, band = 300, 30
bp = 'analyze/back_projection/bear3_OURS_%s_backproj_checker16_uv_k1_withbg_side_by_side.png'
rows = [('SHIC', 'demo/shic/bear3/back_projection/bear3_%s_backproj_checker16_uv_k1_withbg_side_by_side.png'),
        ('step 0 (ZS)', 'demo/shic/bear3_ZS_sdsquare_fuse16_phase1old/' + bp)]
rows += [('step %d' % s, f'{run}/step{s}/' + bp) for s in (5, 10, 15, 20, 25)]
def photo(p):  # keep only the back-projected photo (right panel), drop the repeated mesh render
    im = Image.open(p).convert('RGB'); im = im.crop((im.width - 808, 0, im.width, im.height))
    return im.resize((int(im.width * h / im.height), h))
cols = [photo(p % 'top1') for _, p in rows], [photo(p % 'soft') for _, p in rows]
w = cols[0][0].width
c = Image.new('RGB', (w * len(rows), 2 * (h + band)), 'white'); d = ImageDraw.Draw(c)
for r, (kind, ims) in enumerate(zip(('top1', 'soft'), cols)):
    for i, ((t, _), im) in enumerate(zip(rows, ims)):
        d.text((i * w + 6, r * (h + band) + 8), f'{t} -- {kind}', fill=(0, 0, 0)); c.paste(im, (i * w, r * (h + band) + band))
c.save(run + '/early_steps_grid.png'); print('saved', run + '/early_steps_grid.png')
EOF
echo "EARLY STEPS DONE"
