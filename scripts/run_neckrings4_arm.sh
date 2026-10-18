#!/bin/bash
# NECK-RINGS arm (Tal, 2026-09-16): arm 3's exact recipe, on meshes/guitar_neckrings4.obj instead of
# meshes/guitar.obj. The ONLY difference between this arm and arm 3 is the 144 vertices added to the
# neck gap -- frame, phase-1 checkpoint, checker grid, photo side and every hyper-parameter are
# identical (see scripts/lib_neckrings4_override.sh for the verification numbers).
#
# The cycle weight c is deliberately NOT recalibrated: arm 3 did not recalibrate it when it switched
# meshes either, and matching arm 3 matters more here than being optimal -- otherwise the comparison
# stops isolating vertex density.
set -e
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"
cd $DISC_ROOT
source scripts/lib_isobbox_common.sh
# T is the knob being walked back toward MAIN (Tal, 2026-09-16): the arm that matched arm 3 used
# T=0.01; T=1 is the first step back. Train T and read T are kept equal, as in the whole squash
# -smoothness direction.
T=${T:-0.01}
# PIXARM: which photo-feature arm. NOT "PIXTAG" -- lib_isobbox_common.sh already owns that name
# for the tracked-pixel set (explicit9) and is sourced above, so it would clobber this. squash (dino_sd_stride14_nosdsquare) is what arm 3 and the
# PROTECTED smooth run use; sd_square (dino_sd_aligned) is MAIN's, and is the next step back toward
# MAIN after T (Tal, 2026-09-16).
PIXARM=${PIXARM:-sdsquash}
# TRUEDIAG2D=1: normalize d_2D by the TRUE all-pairs 2D geodesic diameter (1175.75, measured with
# analysis/measure_2d_geodesic_diameter.py) instead of the table's stored 1171.65, which is only a max
# over the sampled centre rows. The 3D side is a true all-pairs max, so without this the two sides of
# the comparison are measured differently. MAIN has used it since 2026-09-08; the whole squash line
# did not, to stay matched to the PROTECTED run. This is the LAST difference from MAIN besides T and
# the mesh (Tal, 2026-09-16).
TRUEDIAG2D=${TRUEDIAG2D:-0}
if [ "$TRUEDIAG2D" = "1" ]; then DIAG_FLAG="--geo_diag_2d 1175.75"; DIAG_TAG="_truediag2d"; else DIAG_FLAG=""; DIAG_TAG=""; fi
export NECKRINGS4=1 SMALL=1 READ_T=$T
source scripts/lib_neckrings4_override.sh

case "$PIXARM" in
  sdsquash) PIX=$PIX_sdsquash_fuse16 ;;
  sdsquare) PIX=$PIX_sdsquare_fuse16 ;;
  *) echo "unknown PIXARM '$PIXARM' (expected sdsquash or sdsquare)"; exit 1 ;;
esac
S=neckrings4_${PIXARM}${DIAG_TAG}_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T${T}_seed0
D=./experiments/guitar/encoder_${S}_phase2
R=bsb_${S}_phase2

if [ -f "$D/pred_f.pth" ]; then echo "already trained: $D"; else
  time python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" --init_encoder_dir "$PHASE1" \
    --object_mask_path "$MASK" --fine_seg_path "$FINE" --pix_dino_features_path "$PIX" \
    --width 2304 --n_classes 2304 --geodesic --temperature $T --train_seed 0 \
    --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 \
    --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" $DIAG_FLAG \
    --mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs \
    --cyc_weight 1.676e-05 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5 \
    --out_dir "$D"
fi
echo "===== NECKRINGS4 ARM TRAINED ====="
