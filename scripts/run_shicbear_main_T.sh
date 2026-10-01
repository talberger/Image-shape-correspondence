#!/bin/bash
# SHIC-bear on the NEW MAIN recipe (Tal, 2026-09-16). Flags copied from MAIN's run_config.json
# (encoder_isobbox_sdsquare_fuse16_truediag2d_mu_learn_perseg_cycv2T0.05_c1.676e-05_retstride5_T0.01_seed0).
# Kept as before on the bear, deliberately (Tal: as few changes as possible): raw bear.obj, its existing
# phase-1, 3D/2D geodesic tables, mask, fine segments. cyc_weight = the guitar's 1.676e-5, NOT recalibrated.
# New for the bear: sd_square fuse16 photo features (bear only had squash) + its true 2D diameter.
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb

T=${T:-0.01}
# MU=learn (MAIN) | closed (per-segment closed form, --mu_mode optimal)
MU=${MU:-learn}
# MU=zsfixed: per-segment scales fixed from the ZS mapping (analysis/perseg_mu_from_zs.py), idea 2, 2026-09-16
if [ "$MU" = closed ]; then MUFLAGS="--mu_mode optimal --mu_scope segment"
elif [ "$MU" = fixed1 ]; then MUFLAGS="--mu_mode learned --mu_scope segment --mu_lr 0"   # per-segment mu frozen at 1.0
elif [ "$MU" = aniso ]; then MUFLAGS="--mu_mode learned --mu_scope segment_aniso --mu_lr ${MU_LR:-1e-2}"   # DIRECTIONAL per-seg mu, 2026-09-21
elif [ "$MU" = grid ]; then MUFLAGS="--mu_mode learned --mu_scope grid_aniso --mu_grid ${MU_GRID:-6} --mu_lr 1e-2"   # DIRECTIONAL mu on a smooth grid, 2026-09-21
elif [ "$MU" = globalaniso ]; then MUFLAGS="--mu_mode learned --mu_scope global_aniso --mu_lr 1e-2"   # ONE ellipse for the whole bear, 2026-09-21
elif [ "$MU" = zsfixed ]; then MUFLAGS="--mu_mode learned --mu_scope segment --mu_lr 0 --mu_init_path ./experiments/shic_bear/mu_perseg/zs_p99_mu_scale.pth"
else MUFLAGS="--mu_mode learned --mu_scope segment --mu_lr 1e-2"; fi
# VISDIAM3D: normalize d_3D by the VISIBLE part's geodesic diameter (analysis/estimate_visible_geo_diameter.py), idea 1
VISDIAM3D=${VISDIAM3D:-}
VISFLAG=""; VISTAG=""
if [ -n "$VISDIAM3D" ]; then VISFLAG="--geo_diag_3d $VISDIAM3D"; VISTAG="_visdiam3d${VISDIAM3D}"; fi
OBJ=./demo/shic/shapes/bear.obj
NAME=shic_bear
P1=./experiments/shic_bear/encoder_dfd_crossview_phase1
MASK=./demo/shic/masks/bear3_click_350_300_sam_mask_1.pth
# FINE: env-overridable (2026-09-21) -- e.g. bear3_fine_segments_merge23.pth (segments 2+3 merged); tag it via SUFFIX
FINE=${FINE:-./demo/shic/bear3/bear3_fine_segments.pth}
GEO3D=./experiments/shic_bear/geodesic_tables/shic_bear_geo_table_3d.pth
GEO2D=./experiments/shic_bear/geodesic_tables/bear3_click_350_300_sam_mask_1_geo_table_2d_seed0.pth
FEATDIR=demo/shic/features_analyze/sdsquare_fuse16
PIX=$FEATDIR/bear3_pix_sdsquare_fuse16.pth
# SUFFIX: appended to the run tag, e.g. to re-run an existing arm into a fresh dir (default empty = old tags)
SUFFIX=${SUFFIX:-}
S=sdsquare_truediag2d${VISTAG}_mu_${MU}_perseg_cycv2T0.05_c1.676e-05_retstride5_T${T}_seed0${SUFFIX}
P2=./experiments/shic_bear/encoder_${S}_phase2
RUN=demo/shic/bear3_${S}_phase2

echo "########## photo features: sd_square, fuse16 ##########"
if [ -f "$PIX" ]; then echo "exists: $PIX"; else
  mkdir -p "$FEATDIR"
  python analysis/extract_dino_sd_features.py --img_path demo/shic/images/bear3.png \
    --save_dir "$FEATDIR" --output_path "$PIX" --sd_frame square_pad --cpu_postprocess
fi

echo "########## true 2D geodesic diameter ##########"
DIAG=$(python analysis/measure_2d_geodesic_diameter.py --mask_path "$MASK" --geo_table_2d_path "$GEO2D" \
  | tee /dev/stderr | awk '/TRUE 2D geodesic diameter/ {print $(NF-1)}')
echo "geo_diag_2d = $DIAG"

echo "########## phase-2, T=$T ##########"
if [ -f "$P2/pred_f.pth" ]; then echo "already trained: $P2"; else
  time python phase2/train_relational_loss_l2norm.py \
    --obj_path "$OBJ" --name "$NAME" --init_encoder_dir "$P1" \
    --object_mask_path "$MASK" --fine_seg_path "$FINE" --pix_dino_features_path "$PIX" \
    --width 2304 --n_classes 2304 --geodesic --temperature $T --train_seed 0 \
    --geo_diag_2d "$DIAG" $VISFLAG \
    --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 \
    --geo_table_3d_path "$GEO3D" --geo_table_2d_path "$GEO2D" \
    $MUFLAGS --mu_fit_space abs \
    --cyc_weight 1.676e-05 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 5 \
    --out_dir "$P2"
fi

if [ -n "${SKIP_ANALYSIS:-}" ]; then echo "SKIP_ANALYSIS set: training only"; exit 0; fi
READ_T=$T P2=$P2 RUN=$RUN PIX=$PIX LABEL="OURS (MAIN recipe, mu=$MU per-seg${VISTAG}, T=$T)" ./scripts/analyze_shicbear_arm.sh
echo "===== SHIC-BEAR MAIN mu=$MU${VISTAG} T=$T DONE ====="
