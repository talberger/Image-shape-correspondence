#!/bin/bash
# Numbers + the 5-figure one-row-per-arm set for a cycle-v2 sweep on top of the anchor (2026-09-10).
# Rows: ZS, strong, balanced, weak cycle weight, anchor-only control. Eval curves also carry the baseline.
# Usage: ./build_cycv2_sweep_figures.sh <CYC_T> <allpix 0|1> <w_strong> <w_balanced> <w_weak>
#   env: OUT_SUFFIX, LABEL_S/B/W (set to "" for no parenthetical), RW_S/B/W (per-row w_spatial), TITLE_NOTE
# Labels (2026-09-11, Tal): every row must show what differs between rows; "allpix" is never shown (it is the only
# variant in use), w_spatial is shown on all rows only when some row differs from 1.
set -e
source /workspace/activate_bsb.sh
cd /workspace/bsb
source scripts/lib_isobbox_common.sh
CYC_T=$1; ALLPIX=$2; WS=$3; WB=$4; WW=$5
[ -n "$WW" ] || { echo "usage: $0 <CYC_T> <allpix 0|1> <w_strong> <w_balanced> <w_weak>"; exit 1; }
# TAGX emptied 2026-09-14 (Tal): `allpix` is gone from encoder dirs and eval tags too, not just demo dirs
TAGX=""
LONG=sdsquare_fuse16_truediag2d_anchorsoftT0.01_w3.5e-7   # encoder dirs + eval tags keep the long name
SHORT=sdsquare_anchorsoftT0.01_w3.5e-7                    # demo dirs dropped fuse16/truediag2d
EG=experiments/guitar; E=$EVALDIR/eval_
OUT=$RUNS/comparison_cycv2_T${CYC_T}_sweep${OUT_SUFFIX}; mkdir -p "$OUT"
LABEL_S=${LABEL_S-strong}; LABEL_B=${LABEL_B-balanced, 1/50 of anchor}; LABEL_W=${LABEL_W-weak}
CHK=$(basename "${CHECKER%.pth}")
# RW_S/RW_B/RW_W (2026-09-11): per-row --relational_weight (w_spatial), matching run_isobbox_cycv2_QUEUE.sh's REL_W tag
rt() { [ "${1:-1}" = 1 ] || echo "_rkdw$1"; }
run() { echo "bsb_isobbox_${SHORT}_cycv2T${CYC_T}_c$1$(rt $2)_T1_seed0_phase2"; }
tag() { echo "isobbox_${LONG}_cycv2${TAGX}T${CYC_T}_c$1$(rt $2)_T1_seed0"; }
enc() { echo "$EG/encoder_isobbox_${LONG}_cycv2${TAGX}T${CYC_T}_c$1$(rt $2)_T1_seed0_phase2"; }
RW_S=${RW_S:-1}; RW_B=${RW_B:-1}; RW_W=${RW_W:-1}
SHOWW=""; [ "$RW_S$RW_B$RW_W" = 111 ] || SHOWW=1
lab() { local s="anchor + cycle T_cyc=$CYC_T c=$1"; [ -n "$SHOWW" ] && s="$s, w_spatial=$2"; [ -n "$3" ] && s="$s ($3)"; echo "$s"; }
ROWS=(
 "ZS (phase-1 only)|bsb_isobbox_ZS_sdsquare|isobbox_ZS_sdsquare_fuse16|$PHASE1/pred_f.pth"
 "$(lab $WS $RW_S "$LABEL_S")|$(run $WS $RW_S)|$(tag $WS $RW_S)|$(enc $WS $RW_S)/pred_f.pth"
 "$(lab $WB $RW_B "$LABEL_B")|$(run $WB $RW_B)|$(tag $WB $RW_B)|$(enc $WB $RW_B)/pred_f.pth"
 "$(lab $WW $RW_W "$LABEL_W")|$(run $WW $RW_W)|$(tag $WW $RW_W)|$(enc $WW $RW_W)/pred_f.pth"
 "anchor only (T_anchor=0.01, lambda=3.5e-7), no cycle|bsb_isobbox_${SHORT}_T1_seed0_phase2|isobbox_${LONG}_T1_seed0|$EG/encoder_isobbox_${LONG}_T1_seed0_phase2/pred_f.pth"
)
for r in "${ROWS[@]}"; do IFS='|' read -r l d t p <<< "$r"; echo "$l|$d|$t"; done > "$OUT/rows.txt"
python - "$OUT/rows.txt" "$E" "$RUNS" "$MB" "isobbox_sdsquare_fuse16_truediag2d_T1_seed0" <<'PY'
import sys, torch
rows, E, R, MB, base = sys.argv[1:6]
lines = [l.rstrip('\n').split('|') for l in open(rows)] + [['BASELINE (mu=1, no anchor)', 'bsb_isobbox_sdsquare_T1_seed0_phase2', base]]
print('| arm | AUC snap/argmax/soft | @3% | coverage | argmax cov | hub10 |\n|---|---|---|---|---|---|')
for lab, d, t in lines:
    e = torch.load(f'{E}{t}_v2.pth'); a = e['auc']; c = e['acc_curves']['geometric snap']; g = e['eps_grid']
    at3 = 100 * float(c[int(torch.argmin((g - 0.03).abs()))])
    k = torch.load(f'{R}/{d}/analyze/{MB}_pixel_to_vertex_nn.pth', map_location='cpu')['top1_vertex_idx']
    s = torch.load(f'{R}/{d}/analyze/color_mesh_from_soft_mapping/{MB}_soft_snap_correspondence.pth', map_location='cpu')['top1_vertex_idx']
    u, n = torch.unique(k, return_counts=True)
    print(f"| {lab} | {100*a['geometric snap']:.1f} / {100*a['feature argmax']:.1f} / {100*a['soft mean distance']:.1f} | {at3:.0f} | "
          f"{100*len(torch.unique(s))/3886:.1f}% | {100*len(u)/3886:.1f}% | {100*n.sort(descending=True).values[:10].sum()/len(k):.1f}% |")
PY
TITLE="cycle v2 T_cyc=$CYC_T on the anchor${TITLE_NOTE:+ -- $TITLE_NOTE}"
build() { args=(); for r in "${ROWS[@]}"; do IFS='|' read -r l d t p <<< "$r"; args+=(--panel "$RUNS/$d/analyze/$1" "$l"); done
  python analysis/stack_images_nway.py "${args[@]}" --title "$2" --output_path "$OUT/$3" | tail -1; }
build back_projection/${MB}_backproj_${CHK}_soft_snap_k1_side_by_side.png "back-projection soft-snap k=1, T_read=1 -- $TITLE" 1_back_projection_soft.png
build back_projection/${MB}_backproj_${CHK}_k1_side_by_side.png "back-projection top1k, T_read=1 -- $TITLE" 2_back_projection_top1k.png
build features_analyze/pca/guitar_isotropic_bboxnorm_mesh_pca_shared_color_side_by_side.png "PCA shared basis -- $TITLE" 3_pca.png
evals=(); labels=()
# full labels: stripping the parenthetical made rows that differ only in it (e.g. w_spatial) indistinguishable
for r in "${ROWS[@]}"; do IFS='|' read -r l d t p <<< "$r"; evals+=("${E}${t}_v2.pth"); labels+=("$l"); done
python analysis/compare_eval_runs.py --runs "${evals[@]}" "${E}isobbox_sdsquare_fuse16_truediag2d_T1_seed0_v2.pth" \
  --labels "${labels[@]}" "BASELINE" --save_path "$OUT/4_eval_curves.png" > /dev/null 2>&1 && echo "saved $OUT/4_eval_curves.png"
panels=()
for r in "${ROWS[@]}"; do
  IFS='|' read -r l d t p <<< "$r"; D=$RUNS/$d/analyze/features_analyze/sim_heatmap
  if [ ! -f "$D/simheatmap_row.png" ]; then
    python analysis/feature_similarity_heatmap.py --obj_path "$OBJ" --query_pixels $QUERY_PIXELS --query_vertices $QUERY_VERTICES \
      --save_dir "$D" --grid_rows 1 --grid_output_path "$D/simheatmap_row.png" \
      --mask_path "$MASK" --features_path "$PIX_sdsquare_fuse16" --vertex_features_path "$p" > /dev/null
  fi
  panels+=(--panel "$D/simheatmap_row.png" "$l")
done
python analysis/stack_images_nway.py "${panels[@]}" --title "similarity heatmaps, fixed queries -- $TITLE" --output_path "$OUT/5_sim_heatmap.png" | tail -1
echo "===== FIGURES DONE: $OUT ====="
