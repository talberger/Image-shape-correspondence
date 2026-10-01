#!/bin/bash
# Per-instance inputs for ANY SHIC class (2026-09-22): the prep half of scripts/run_lvis_instance.sh, kept in
# data/densepose_lvis/runs/lvis<ann_id> (never deleted). The class's 3D geodesic table is built on its first instance.
# Usage: CLASS=horse ANN_ID=164135 bash scripts/run_lvis_prep.sh   (then IMG_TAG=lvis<id> CLASS=.. run_shicbear_mu_lora.sh)
set -euo pipefail
source /workspace/activate_bsb.sh
cd /workspace/bsb
CLASS=${CLASS:?}; ANN_ID=${ANN_ID:?}; TAG=lvis$ANN_ID; D=data/densepose_lvis/runs/$TAG
OBJ=./demo/shic/shapes/$CLASS.obj; NAME=shic_$CLASS; GT=./experiments/shic_$CLASS/geodesic_tables
mkdir -p $D $GT
python analysis/prep_lvis_instance.py --ann_id $ANN_ID --class_name $CLASS --out_dir $D --long_side 400
python analysis/build_fine_segments.py --img_path $D/$TAG.png --mask_path $D/${TAG}_mask.pth --save_dir $D --min_pixels_in_object 50
SKIP3D=--skip_3d; [ -f $GT/${NAME}_geo_table_3d.pth ] || SKIP3D=""
python analysis/build_geodesic_tables.py --obj_path $OBJ --name $NAME $SKIP3D \
    --object_mask_path $D/${TAG}_mask.pth --fine_seg_path $D/${TAG}_fine_segments.pth \
    --num_center_pixels 2048 --seed 0 --true_diameter --out_dir $D 2>&1 | tee $D/geo_build.log
[ -f $D/${NAME}_geo_table_3d.pth ] && mv $D/${NAME}_geo_table_3d.pth $GT/
python analysis/extract_dino_sd_features.py --img_path $D/$TAG.png --save_dir $D \
    --output_path $D/${TAG}_pix_sdsquare_fuse16.pth --sd_frame square_pad --cpu_postprocess
echo "LVIS PREP DONE: $D"
