#!/bin/bash
# One DensePose-LVIS bear instance through our whole per-image pipeline (2026-09-19, timing run first).
# Recipe = Tal's current choice: MAIN (rkd_spatial_dist, learned per-seg mu, 16 in-seg, cycle v2 T=0.01)
# + LORA-W ALL r4, read at alpha 0.5 -- the same flags as scripts/run_shicbear_loraW.sh.
# Usage: ANN_ID=117683 bash scripts/run_lvis_instance.sh
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb

ANN_ID=${ANN_ID:?}; ALPHA=${ALPHA:-0.5}
TAG=lvis$ANN_ID
# RUNS_DIR is overridable so a reference run can be pointed somewhere else and diffed against
# scripts/run_lvis_worker.py's output for the same instance (the merged-process equivalence check).
D=${RUNS_DIR:-data/densepose_lvis/runs}/$TAG
OBJ=./demo/shic/shapes/bear.obj; NAME=shic_bear
P1=./experiments/shic_bear/encoder_dfd_crossview_phase1
GEO3D=./experiments/shic_bear/geodesic_tables/shic_bear_geo_table_3d.pth
P2=$D/encoder_loraW_all_r4_phase2
# Results live OUTSIDE $D on purpose: $D is deleted at the end of every instance (7.4 GB each).
RESULTS=${RESULTS:-data/densepose_lvis/results}; mkdir -p $RESULTS
t() { local s=$(date +%s); "$@"; echo "[time] $(( $(date +%s) - s )) s : $1 $2" | tee -a $D/timing.txt; }
mkdir -p $D; : > $D/timing.txt

t python analysis/prep_lvis_instance.py --ann_id $ANN_ID --out_dir $D --long_side 400
t python analysis/build_fine_segments.py --img_path $D/$TAG.png --mask_path $D/${TAG}_mask.pth --save_dir $D --min_pixels_in_object 50
# --true_diameter folds in what used to be a separate ~90s measure_2d_geodesic_diameter.py process
# (it rebuilt this identical graph from scratch); --num_workers defaults to the container's CPU
# quota, not nproc. Both are bit-identical to the old two-step version -- see those flags' help.
t python analysis/build_geodesic_tables.py --obj_path $OBJ --name $NAME --skip_3d \
    --object_mask_path $D/${TAG}_mask.pth --fine_seg_path $D/${TAG}_fine_segments.pth \
    --num_center_pixels 2048 --seed 0 --true_diameter --out_dir $D 2>&1 | tee $D/geo_build.log
DIAG=$(awk '/TRUE 2D geodesic diameter/ {print $(NF-1)}' $D/geo_build.log)
[ -n "$DIAG" ] || { echo "FAILED to read geo_diag_2d from $D/geo_build.log" >&2; exit 1; }
echo "geo_diag_2d = $DIAG"
t python analysis/extract_dino_sd_features.py --img_path $D/$TAG.png --save_dir $D \
    --output_path $D/${TAG}_pix_sdsquare_fuse16.pth --sd_frame square_pad --cpu_postprocess
t python phase2/train_relational_loss_l2norm.py --obj_path $OBJ --name $NAME --init_encoder_dir $P1 \
    --object_mask_path $D/${TAG}_mask.pth --fine_seg_path $D/${TAG}_fine_segments.pth \
    --pix_dino_features_path $D/${TAG}_pix_sdsquare_fuse16.pth \
    --width 2304 --n_classes 2304 --geodesic --temperature 0.01 --train_seed 0 --geo_diag_2d $DIAG \
    --geo_table_3d_path $GEO3D --geo_table_2d_path $D/${TAG}_mask_geo_table_2d_seed0.pth \
    --mu_mode learned --mu_scope segment --mu_lr 1e-2 --mu_fit_space abs \
    --num_neighbors_in_segment 16 --num_neighbors_out_segment 0 \
    --cyc_weight 1.676e-05 --cyc_v2 --cyc_temperature 0.05 --cyc_return_all_pixels --cyc_return_stride 3 \
    --lora_w_rank 4 --lora_w_layers all --lora_w_lr 1e-3 --no_step_checkpoints --out_dir $P2
t python analysis/export_pred_f_lora_alpha.py --obj_path $OBJ --name $NAME \
    --phase1_checkpoint $P1/encoder_checkpoint.pth --lora_checkpoint $P2/encoder_checkpoint.pth \
    --alphas $ALPHA --out_dir $P2/alpha

# Scoring, in SHIC's Table 1 metric (analysis/eval_lvis_ours.py). Two arms per instance:
#   ours = MAIN + LORA-W ALL r4 read at alpha $ALPHA;  ZS = the phase-1 encoder, no phase-2 at all.
# ZS costs nothing extra -- phase-1's pred_f.pth is the same file for every instance -- and it is the
# baseline row the comparison needs (SHIC's own zero-shot SD-DINO bear row is 48.2).
GEO228=data/densepose_lvis/eval_shic_bear/bear_heat_geodesics_228.npy
t python analysis/eval_lvis_ours.py --run_dir $D --tag $TAG --obj_path $OBJ --name $NAME \
    --geodesics_path $GEO228 --out_json $RESULTS/${TAG}.json \
    --arm ours=$P2/alpha/alpha$ALPHA/pred_f.pth --arm zs=$P1/pred_f.pth
cp $D/timing.txt $RESULTS/${TAG}_timing.txt

echo "LVIS INSTANCE DONE: $D"; cat $D/timing.txt

# The intermediates are 7.4 GB/instance (geo table 2.6 + features 4.8); 132 of them would be ~1 TB,
# so they go as soon as they have been scored. The per-instance JSONs in $RESULTS are what survives.
if [ "${KEEP_INTERMEDIATES:-0}" != "1" ]; then
  rm -rf $D
  echo "deleted intermediates $D (KEEP_INTERMEDIATES=1 to keep them)"
fi
