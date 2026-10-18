#!/bin/bash
# Debug ONE LVIS instance of the two-stage method end to end (Tal, 2026-10-06; first done by hand on horse 184389):
# rerun with the eval recipe (keeping the end-of-stage-1 + final checkpoints), score ZS / end of stage 1 / final,
# back-project each, and build the figures: back-projection per stage, coarse mask, fine segments one by one, ellipses, summary vs SHIC.
# Usage: CLASS=horse ANN_ID=56069 [MERGE=0.015 (default; 0 = no merge)] [SEGONLY=1] [AMODAL=1] [SAM3=1 (mask from SAM 3, no polygon)] [FILLSEG=1 (default since 2026-10-07, Tal: unlabelled pixels join the nearest segment; 0 = off)] [ANALYSIS_ONLY=1] [OUT_DIR=..] [RUNS_DIR=..] bash scripts/debug_twostage_instance.sh
# ~7.7 min on an RTX 2000 Ada. Output: demo/shic/debug_twostage/<class>_lvis<id>[_merge..][_segonly]/, intermediates in data/densepose_lvis/runs_debug[...]/.
set -euo pipefail
source /workspace/activate_bsb.sh; cd /workspace/bsb
CLASS=${CLASS:?}; ANN_ID=${ANN_ID:?}; MERGE=${MERGE:-0.015}; TAG=lvis$ANN_ID   # default 1.5% since 2026-10-07 (Tal); MERGE=0 = the old eval recipe
SEGONLY=${SEGONLY:-0}   # 1 = --out_neighbors_segmented_only
AMODAL=${AMODAL:-0}     # 1 = occluded object: 2D geodesics through pix2gestalt's amodal mask (scripts/make_amodal_mask.sh, +~2 min)
SAM3=${SAM3:-0}         # 1 = NO LVIS polygon: the object mask is SAM 3's, box prompt = the whole crop (scripts/make_sam3_mask.sh; Tal, 2026-10-08)
FEATGEO=${FEATGEO:-}    # "BETA POWER" = 2D geodesic table with the feature step cost (crowded parts, 2026-10-13); empty = the original metric
SUF=$([ "$MERGE" = 0 ] && echo "" || echo "_merge$MERGE")$([ "$SEGONLY" = 1 ] && echo "_segonly" || echo "")$([ "$AMODAL" = 1 ] && echo "_amodal" || echo "")$([ "${FILLSEG:-1}" = 1 ] && echo "_fillseg" || echo "")$([ "$SAM3" = 1 ] && echo "_sam3mask" || echo "")$([ -n "$FEATGEO" ] && echo "_featgeo_b${FEATGEO% *}_p${FEATGEO#* }" || echo "")$([ "${FEATGEO_DIAM:-original}" = path ] && echo "_pathdiam" || echo "")
RUNS=${RUNS_DIR:-data/densepose_lvis/runs_debug$SUF}; D=$RUNS/$TAG; P2=$D/encoder_loraW_all_r4_phase2
P1=./experiments/shic_$CLASS/encoder_dfd_crossview_phase1; OBJ=./demo/shic/shapes/$CLASS.obj; NAME=shic_$CLASS
O=demo/shic/debug_twostage/${CLASS}_$TAG$SUF; mkdir -p $O
if [ "$CLASS" = bear ]; then CHK=demo/shic/colored_mesh/bear_shic_normalized_mesh_checker16_uv_color.pth; RENDER=demo/shic/colored_mesh/checker16_uv_azim135.png
else CHK=demo/shic/colored_mesh/${CLASS}_mesh_checker16_uv_color.pth; RENDER=demo/shic/colored_mesh/${CLASS}_checker16_uv_azim135.png; fi
O=${OUT_DIR:-$O}; mkdir -p $O   # OUT_DIR: write the analysis somewhere else (used to verify the speed-ups)
# 2026-10-07 speed-ups (same commands, same outputs). Each script pays ~15-25 s of imports alone, so:
#  - independent scripts run side by side;
#  - what does NOT depend on the training (position colours, ZS mapping + back-projection, SHIC inference + back-projection)
#    is built WHILE the training runs: the "early" chain starts as soon as the worker reports the features are written;
#  - the final / SHIC back-projections are computed once (KEEP_PLAIN=1 keeps the standalone PNG, moved below);
#  - only the end-of-stage-1 checkpoint is kept next to the final one (--stage1_checkpoint), not the 12 step dumps.
# ANALYSIS_ONLY=1 skips the worker and redoes the analysis on a kept run.
bp() {  # name, correspondence .pth, k-dir, extra flags
  python analysis/color_image_from_knn.py --mask_path $D/${TAG}_mask.pth --knn_dir $3 --knn_path $2 --vertex_colors_path $CHK \
    --source_image_path $D/$TAG.png --k 1 --save_dir $O/$1 --output_path $O/$1/${TAG}_$1_backproj.png --no_side_by_side
}
softmap() {  # name, pred_f
  mkdir -p $O/$1/soft_mapping
  python analysis/color_mesh_from_soft_mapping.py --mask_path $D/${TAG}_mask.pth --img_path $D/$TAG.png --save_dir $O/$1/soft_mapping \
    --features_path $D/${TAG}_pix_sdsquare_fuse16.pth --vertex_features_path $2 --obj_path $OBJ --name $NAME \
    --color_mode position --color_path $SB/${TAG}_position_color.png --k_vertices_per_pixel 10 --temperature 0.01
  bp $1 $O/$1/soft_mapping/${TAG}_mask_soft_snap_correspondence.pth $O/$1/soft_mapping
}
SB=$O/summary/backproj_$TAG; mkdir -p $SB/shic; WLOG=$O/worker.log; : > $WLOG
SLIM_FEAT=0
if [ "${ANALYSIS_ONLY:-0}" = 1 ] && [ ! -f $D/${TAG}_pix_sdsquare_fuse16.pth ]; then   # slim-kept run (2026-10-10): the photo features were dropped after scoring
  echo "##### photo features missing (slim-kept run) -> re-extracting (removed again at the end) #####"
  python analysis/extract_dino_sd_features.py --img_path $D/$TAG.png --save_dir $D --output_path $D/${TAG}_pix_sdsquare_fuse16.pth \
    --sd_frame square_pad --cpu_postprocess
  SLIM_FEAT=1
fi
(
set -e
if [ "${ANALYSIS_ONLY:-0}" != 1 ]; then   # wait for the worker's "features" stage; give up if the worker dies first
  until grep -q '^\[time\].* : features' $WLOG; do kill -0 $$ 2>/dev/null || exit 1; grep -q 'WORKER EXIT' $WLOG && exit 1; sleep 2; done
fi
echo "##### early chain (independent of the training): position colours, ZS, SHIC #####"
python analysis/pixel_coloring.py --img_path $D/$TAG.png --mask_path $D/${TAG}_mask.pth \
  --mode position --save_dir $SB --output_path $SB/${TAG}_position_color.png
( softmap zs $P1/pred_f.pth ) > $O/chain_zs.log 2>&1 &
PZ=$!
python analysis/shic_infer_correspondence.py --class_name $CLASS \
  --image_path $D/$TAG.png --mask_path $D/${TAG}_mask.pth --save_dir $SB/shic --top_k 10
python analysis/color_image_from_knn.py --mask_path $D/${TAG}_mask.pth \
  --knn_dir $SB/shic --knn_path $SB/shic/${TAG}_shic_pixel_to_vertex_nn.pth \
  --vertex_colors_path $CHK --source_image_path $D/$TAG.png --k 1 --save_dir $SB/shic \
  --output_path $SB/shic/${TAG}_SHIC_argmax_backproj.png --mesh_render_path $RENDER --keep_plain
python analysis/compare_ours_vs_shic_points.py --ann_id $ANN_ID --class_name $CLASS --run_dir $D --out_dir $O/summary --shic_only   # fills the SHIC point cache
wait $PZ; cat $O/chain_zs.log
) > $O/chain_early.log 2>&1 &
P_E=$!
if [ "${ANALYSIS_ONLY:-0}" != 1 ]; then
if [ "$AMODAL" = 1 ]; then
# 2026-10-08 (speed): pix2gestalt starts NOW, so its ~70 s of imports + model load overlap with SAM 3 / the fine segments instead of
# sitting on the critical path after the mask. It waits for the input (or for "mask in one piece, nothing to do") -> $O/chain_pix2gestalt.log
mkdir -p $D; DABS=$(realpath $D); rm -f $D/${TAG}_amodal.done $D/${TAG}_amodal.failed $D/${TAG}_amodal_input.ready $D/${TAG}_amodal_samples.done
( cd /workspace/pix2gestalt_repo/pix2gestalt && env -u PYTHONPATH -u CONDA_DEFAULT_ENV PATH=/workspace/pix2gestalt_repo/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONPATH=../stubs:taming-transformers:CLIP python ../run_amodal.py --run_dirs $DABS --tags $TAG --n_samples 3 --steps 50 --wait_for_input 900 ) > $O/chain_pix2gestalt.log 2>&1 &
fi
FEATW=""
if [ "$SAM3" = 1 ]; then
# 2026-10-08 (speed): SAM 3 is ~60 s, almost all of it process start-up; the pixel features need only the crop, so they are
# extracted in the background meanwhile (the worker waits for the marker, --features_wait, instead of running that stage).
echo "##### crop, then SAM 3 object mask (no polygon) + pixel features in the background -> $O/chain_features.log #####"
mkdir -p $D; rm -f $D/${TAG}_features.done $D/${TAG}_features.failed
python analysis/prep_lvis_instance.py --ann_id $ANN_ID --class_name $CLASS --out_dir $D --long_side 400
( python analysis/extract_dino_sd_features.py --img_path $D/$TAG.png --save_dir $D --output_path $D/${TAG}_pix_sdsquare_fuse16.pth \
    --sd_frame square_pad --cpu_postprocess && touch $D/${TAG}_features.done || touch $D/${TAG}_features.failed ) > $O/chain_features.log 2>&1 &
CLASS=$CLASS ANN_ID=$ANN_ID D=$D SKIP_PREP=1 bash scripts/make_sam3_mask.sh
FEATW="--features_wait 900"
fi
if [ "$AMODAL" = 1 ]; then
# 2026-10-08 (speed): built in the BACKGROUND while the worker makes the fine segments + features (the worker waits for it before the
# geodesic table, --amodal_wait), and only if the mask is split (ONLY_IF_SPLIT; AMODAL_FORCE=1 = always, as before).
echo "##### amodal mask (pix2gestalt), in the background -> $O/chain_amodal.log #####"
MSRC=$([ "$SAM3" = 1 ] && echo sam3 || echo polygon)
python analysis/prep_lvis_instance.py --ann_id $ANN_ID --class_name $CLASS --out_dir $D --long_side 400 --mask_source $MSRC
( CLASS=$CLASS ANN_ID=$ANN_ID D=$D MASK_SOURCE=$MSRC SKIP_PREP=1 PRELOADED=1 ONLY_IF_SPLIT=$([ "${AMODAL_FORCE:-0}" = 1 ] && echo 0 || echo 1) bash scripts/make_amodal_mask.sh \
    || touch $D/${TAG}_amodal.failed ) > $O/chain_amodal.log 2>&1 &
fi
echo "##### worker (eval recipe, stage-1 + final checkpoints, intermediates kept) #####"
PYTHONUNBUFFERED=1 python scripts/run_lvis_worker.py --ann_ids $ANN_ID --class_name $CLASS --results_dir $RUNS/results --runs_dir $RUNS \
  --mu_scope global_then_segment_aniso --mu_switch_step 150 --mu_lr 5e-2 --keep_intermediates --stage1_checkpoint $FEATW $([ "$SAM3" = 1 ] && echo "--mask_source sam3" || echo "") $([ "$AMODAL" = 1 ] && echo "--amodal_mask --amodal_wait 900" || echo "") $([ "${FILLSEG:-1}" = 1 ] && echo "--fill_unlabelled_segments" || echo "") \
  $([ -n "$FEATGEO" ] && echo "--feature_geodesic $FEATGEO --feature_geodesic_diameter ${FEATGEO_DIAM:-original}" || echo "") $([ "$MERGE" = 0 ] || echo "--merge_below_frac $MERGE") $([ "$SEGONLY" = 1 ] && echo "--out_neighbors_segmented_only" || echo "") 2>&1 | tee -a $WLOG \
  || { echo "WORKER EXIT" >> $WLOG; echo "DEBUG INSTANCE FAILED (worker)"; exit 1; }
fi
(
set -e
echo "##### end of stage 1 at alpha 0.5, per-point eval of the three stages, stage-1 back-projection #####"
[ ! -f $P2/encoder_checkpoint_stageA_end.pth ] && [ -f $P2/alpha_stageA_end/alpha0.5/pred_f.pth ] || \
python analysis/export_pred_f_lora_alpha.py --obj_path $OBJ --name $NAME --phase1_checkpoint $P1/encoder_checkpoint.pth \
  --lora_checkpoint $P2/encoder_checkpoint_stageA_end.pth --alphas 0.5 --out_dir $P2/alpha_stageA_end
( softmap stageA $P2/alpha_stageA_end/alpha0.5/pred_f.pth ) > $O/chain_stageA.log 2>&1 &
PS=$!
python analysis/eval_lvis_ours.py --run_dir $D --tag $TAG --obj_path $OBJ --name $NAME \
  --geodesics_path data/densepose_lvis/eval_shic_$CLASS/${CLASS}_heat_geodesics_228.npy --out_json $O/eval_arms.json \
  --arm zs=$P1/pred_f.pth --arm stageA=$P2/alpha_stageA_end/alpha0.5/pred_f.pth --arm final=$P2/alpha/alpha0.5/pred_f.pth
wait $PS; cat $O/chain_stageA.log
) > $O/chain_stage1.log 2>&1 &
P_A=$!
wait $P_E || { cat $O/chain_early.log; echo "DEBUG INSTANCE FAILED (early chain)"; exit 1; }
cat $O/chain_early.log
(
echo "##### summary vs SHIC (v2 layout) #####"
PRECOMPUTED=1 KEEP_PLAIN=1 CLASS=$CLASS ANN_ID=$ANN_ID PRED_F=$P2/alpha/alpha0.5/pred_f.pth RUNS=$RUNS BASE=$O/summary SHIC_BASE=$O/summary bash scripts/make_lvis_arm_summary.sh
) > $O/chain_summary.log 2>&1 &
P_B=$!
FAIL=0; for P in $P_A $P_B; do wait $P || FAIL=1; done
cat $O/chain_stage1.log $O/chain_summary.log
[ $FAIL = 0 ] || { echo "DEBUG INSTANCE FAILED (stage-1 chain or summary)"; exit 1; }
mkdir -p $O/final $O/shic
mv $O/summary/backproj_$TAG/${TAG}_OURS_soft_backproj.png $O/final/${TAG}_final_backproj.png
mv $O/summary/backproj_$TAG/shic/${TAG}_SHIC_argmax_backproj.png $O/shic/${TAG}_shic_backproj.png
echo "##### figures #####"
python analysis/debug_twostage_instance.py --tag $TAG --run_dir $D --debug_dir $O --p2 $P2 --key_render $RENDER \
  --shic_backproj $O/shic/${TAG}_shic_backproj.png --shic_json $O/summary/${TAG}_vs_shic.json --arm_dirs zs stageA final --arm_keys zs stageA final
[ $SLIM_FEAT = 1 ] && rm -f $D/${TAG}_pix_sdsquare_fuse16.pth
echo "DEBUG INSTANCE DONE $CLASS $ANN_ID -> $O"
