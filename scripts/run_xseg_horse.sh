#!/bin/bash
# XSEG experiment on horse 56069 (Tal, 2026-10-19; plan in workspace_meta/PROJECT_NOTES.md, entry of 2026-10-19).
# Problem: across segments the 2D geodesic is the wrong teacher when parts touch in the photo (nose and belly are close in
# the image, far on the mesh) -> the nose (GT pt 3) collapses in stage 1. Fix tried here: ONE stage, 300 steps, 8 in + 8 out
# neighbours; in-segment pairs -> rkd_spatial_dist with their segment ellipse, cross-segment pairs -> feature RKD instead.
#
#   ARM=perseg  exp 0: per-segment only (16 in + 0 out, segment ellipses, no stage 1) -- existing code, no cross pairs at all
#   ARM=balance calibration: 25 steps of in-segment spatial + cycle, then the grad norms of all three feature modes
#   ARM=table|wonly|Fonly  exp 1, cross-segment feature student (see xseg_feat_loss in modules/feature_rkd_loss.py);
#                          needs WEIGHT=<w> (from ARM=balance, equal grad norms = Tal's "similar proportion")
#   ARM=full   no detach at all (V and w both live; may form a cycle, Tal 2026-10-22)
#   ARM=nofeat   ablation (Tal 2026-10-22): the XSEG split with the feature term OFF (8 in-seg spatial + cycle; cross pairs dropped)
#   ARM=featonly ablation: Fonly feature term on the cross pairs + cycle, rkd_spatial_dist OFF; needs WEIGHT=<Fonly w>
#   ARM=balance_table  balance, then table at the equal-grad-norm weight (/RATIO)
#   ARM=all     the whole queue: perseg, balance, then the three arms at the equal-grad-norm weights (RATIO=1 default)
#
# Everything else = the current LVIS recipe of the two-stage run it is compared with (LoRA-W all r4 lr 1e-3, read at alpha 0.5,
# cycle v2 c=1.676e-5 T_cyc=0.05 return stride 3, T=0.01, mu lr 5e-2), on that run's own inputs (SAM 3 mask, merge 1.5%,
# fillseg; the horse mask is in one piece, so no amodal domain). GPU only (training + rendering-free eval).
# Output: runs data/densepose_lvis/runs_xseg/lvis56069/<arm>/, eval + back-projection demo/shic/xseg/horse_lvis56069/<arm>/,
# logs $DISC_WS/logs/xseg/<arm>.log.
# Other segment map (2026-10-21, VLPart): FINESEG=<label map .pth> SEGTAG=<name> -> runs_xseg_<name>/, xseg_<name>/, logs/xseg_<name>/. ~6-8 min per arm on an RTX 2000 Ada (estimate, not measured).
set -euo pipefail
source "${DISC_ROOT:-/workspace/disc}/env/activate.sh"; cd $DISC_ROOT
ARM=${ARM:?ARM=perseg|balance|table|wonly|Fonly|full|nofeat|featonly|all}
CLASS=${CLASS:-horse}; TAG=${TAG:-lvis56069}
# Bears (2026-10-24): CLASS=bear TAG=lvis<id> SRC=<run dir> LOGS=<per-instance dir> NO_FIG=1 -- driven by scripts/run_xseg_bears.sh
# Other horse instance (2026-10-22): TAG=<tag> SRC=<its inputs dir> TWOSTAGE=<pred_f.pth compared against> POSCOL=<position colours png>
SRC=${SRC:-data/densepose_lvis/runs_debug_merge0.015_amodal_fillseg_sam3mask/$TAG}  # inputs + the two-stage run compared against
TWOSTAGE=${TWOSTAGE:-$SRC/encoder_loraW_all_r4_phase2/alpha/alpha0.5/pred_f.pth}
OBJ=./demo/shic/shapes/$CLASS.obj; NAME=shic_$CLASS; P1=./experiments/shic_$CLASS/encoder_dfd_crossview_phase1
GEO3D=./experiments/shic_$CLASS/geodesic_tables/shic_${CLASS}_geo_table_3d.pth
GEO228=data/densepose_lvis/eval_shic_$CLASS/${CLASS}_heat_geodesics_228.npy
CHK=demo/shic/colored_mesh/${CLASS}_mesh_checker16_uv_color.pth
POSCOL=${POSCOL:-demo/shic/debug_twostage/${CLASS}_${TAG}_merge0.015_amodal_fillseg_sam3mask/summary/backproj_$TAG/${TAG}_position_color.png}
FINESEG=${FINESEG:-$SRC/${TAG}_fine_segments.pth}; SEGTAG=${SEGTAG:+_$SEGTAG}
# GEO2D: 2D geodesic table built WITH this FINESEG (centres are sampled from labelled pixels, so a map with other holes needs its own)
GEO2D=${GEO2D:-$SRC/${TAG}_mask_geo_table_2d_seed0.pth}
RUNS=${RUNS:-data/densepose_lvis/runs_xseg$SEGTAG}/$TAG; DEMO=${DEMO:-demo/shic/xseg$SEGTAG}/${CLASS}_$TAG; LOGS=${LOGS:-$DISC_WS/logs/xseg$SEGTAG}
# amodal geodesic domain (bears whose SAM 3 mask is split): the same flag the two-stage run (run_lvis_worker.py) trained with
AMODAL=(); [ -f $SRC/${TAG}_amodal_mask.pth ] && AMODAL=(--geodesic_domain_mask_path $SRC/${TAG}_amodal_mask.pth)
NO_FIG=${NO_FIG:-0}   # 1 = skip the back-projection figure (eval only)
mkdir -p $RUNS $DEMO $LOGS
for f in $SRC/${TAG}_mask.pth $FINESEG $SRC/${TAG}_pix_sdsquare_fuse16.pth $GEO2D \
         $SRC/${TAG}_gt.pth $TWOSTAGE $P1/encoder_checkpoint.pth $GEO3D $GEO228 $([ $NO_FIG = 1 ] || echo $CHK $POSCOL); do
  [ -e "$f" ] || { echo "MISSING INPUT $f"; exit 1; }
done
DIAG=$(python -c "import json; print(json.load(open('${GEO2D%.pth}_meta.json'))['true_diameter_2d'])")

phase2() {  # out_dir, extra flags...
  local out=$1; shift
  PYTHONUNBUFFERED=1 python phase2/train_relational_loss_l2norm.py \
    --obj_path $OBJ --name $NAME --init_encoder_dir $P1 \
    --object_mask_path $SRC/${TAG}_mask.pth --fine_seg_path $FINESEG \
    --pix_dino_features_path $SRC/${TAG}_pix_sdsquare_fuse16.pth \
    --width 2304 --n_classes 2304 --geodesic --temperature 0.01 --train_seed 0 --geo_diag_2d $DIAG \
    --geo_table_3d_path $GEO3D --geo_table_2d_path $GEO2D \
    --mu_mode learned --mu_scope segment_aniso --mu_lr 5e-2 --mu_fit_space abs \
    --cyc_weight 1.676e-05 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 3 \
    --lora_w_rank 4 --lora_w_lr 1e-3 --lora_w_layers all --no_step_checkpoints --num_iters 300 \
    ${AMODAL[@]+"${AMODAL[@]}"} --out_dir $out "$@"
}

analyse() {  # arm name, run dir: alpha export, per-point eval (zs / two-stage / this arm), soft-snap back-projection
  # ONE python process for the four scripts (2026-10-22): torch is imported once (~60 s each on a slow volume).
  local arm=$1 p2=$2 o=$DEMO/$1
  mkdir -p $o/soft_mapping
  local fig=()
  [ $NO_FIG = 1 ] || fig=(--- \
    analysis/color_mesh_from_soft_mapping.py --mask_path $SRC/${TAG}_mask.pth --img_path $SRC/$TAG.png --save_dir $o/soft_mapping \
      --features_path $SRC/${TAG}_pix_sdsquare_fuse16.pth --vertex_features_path $p2/alpha/alpha0.5/pred_f.pth --obj_path $OBJ --name $NAME \
      --color_mode position --color_path $POSCOL --k_vertices_per_pixel 10 --temperature 0.01 --- \
    analysis/color_image_from_knn.py --mask_path $SRC/${TAG}_mask.pth --knn_dir $o/soft_mapping \
      --knn_path $o/soft_mapping/${TAG}_mask_soft_snap_correspondence.pth --vertex_colors_path $CHK \
      --source_image_path $SRC/$TAG.png --k 1 --save_dir $o --output_path $o/${TAG}_${arm}_backproj.png --no_side_by_side)
  python analysis/run_in_one_process.py \
    analysis/export_pred_f_lora_alpha.py --obj_path $OBJ --name $NAME --phase1_checkpoint $P1/encoder_checkpoint.pth \
      --lora_checkpoint $p2/encoder_checkpoint.pth --alphas 0.5 --out_dir $p2/alpha --- \
    analysis/eval_lvis_ours.py --run_dir $SRC --tag $TAG --obj_path $OBJ --name $NAME --geodesics_path $GEO228 \
      --out_json $o/eval_arms.json --arm zs=$P1/pred_f.pth --arm twostage=$TWOSTAGE --arm $arm=$p2/alpha/alpha0.5/pred_f.pth \
    ${fig[@]+"${fig[@]}"}
  python - $o/eval_arms.json <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print('soft-snap per point (GT pt 1 / 2 / 3 = nose), mean:')
for a, v in r['arms'].items():
    s = v['soft_snap']
    print('  %-10s %s   mean %.1f' % (a, ' / '.join('%.1f' % x for x in s['per_point']), s['all']['mean']))
EOF
}

train_arm() {  # arm [weight]
  local arm=$1 p2=$RUNS/$1
  case $arm in
    perseg) phase2 $p2 --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 ;;
    table|wonly|Fonly|full) phase2 $p2 --num_neighbors_in_segment 8 --num_neighbors_out_segment 8 \
                          --xseg_feat_mode $arm --xseg_feat_weight ${2:?WEIGHT for $arm} ;;
    nofeat) phase2 $p2 --num_neighbors_in_segment 8 --num_neighbors_out_segment 8 --xseg_feat_mode Fonly --xseg_feat_weight 0 ;;
    featonly) phase2 $p2 --num_neighbors_in_segment 8 --num_neighbors_out_segment 8 \
                     --xseg_feat_mode Fonly --xseg_feat_weight ${2:?WEIGHT for $arm} --rkd_spatial_dist_weight 0 ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
}

analyse_bg() {  # arm: the analysis runs in the background, appending to the arm's log, while the next arm trains (2026-10-22)
  ( analyse $1 $RUNS/$1; echo "XSEG ARM DONE $1 -> $DEMO/$1" ) >> $LOGS/$1.log 2>&1 &
}

balance() {
  phase2 $RUNS/balance --num_neighbors_in_segment 8 --num_neighbors_out_segment 8 \
    --xseg_feat_mode table --xseg_feat_weight 0 --xseg_feat_balance --xseg_feat_balance_step 25
}

case $ARM in
  perseg|nofeat) train_arm $ARM 2>&1 | tee $LOGS/$ARM.log; analyse_bg $ARM ;;
  featonly) train_arm featonly ${WEIGHT:?WEIGHT=<Fonly w> from ARM=balance} 2>&1 | tee $LOGS/featonly.log; analyse_bg featonly ;;
  balance) balance 2>&1 | tee $LOGS/balance.log ;;
  table|wonly|Fonly|full) train_arm $ARM ${WEIGHT:?WEIGHT=<w> from ARM=balance} 2>&1 | tee $LOGS/$ARM.log; analyse_bg $ARM ;;
  all|balance_table)
    if [ $ARM = all ]; then train_arm perseg 2>&1 | tee $LOGS/perseg.log; analyse_bg perseg; fi
    balance 2>&1 | tee $LOGS/balance.log
    [ $ARM = all ] && MODES="table wonly Fonly full" || MODES=table
    for m in $MODES; do
      w=$(python -c "
import re, sys
for l in open('$LOGS/balance.log'):
    g = re.match(r'XSEG_WEIGHT mode=$m weight_equal=(\S+)', l)
    if g: print('%.4e' % (float(g.group(1)) / ${RATIO:-1})); sys.exit()
sys.exit('no XSEG_WEIGHT line for $m in $LOGS/balance.log')")
      echo "##### $m at weight $w (1/${RATIO:-1} of rkd_spatial_dist's grad norm) #####" | tee $LOGS/$m.log
      train_arm $m $w 2>&1 | tee -a $LOGS/$m.log
      analyse_bg $m
    done ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
wait   # the background analyses
echo "XSEG QUEUE DONE"
