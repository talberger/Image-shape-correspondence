Fine-segment checks on horse lvis56069 (Mac, 2026-10-21). One-off diagnostics, not part of the pipeline. Usage: `python <script> <out_dir>`.
- `sam1_iou_sweep.py`   SAM 1 automatic masks at pred_iou_thresh 0.90/0.88/0.85/0.80, raw map vs the training recipe (merge 1.5% + fillseg). Env `disc`.
- `sam1_head_check.py`  SAM 1 single-point prompts (3 masks each) + box_nms x stability sweep. Env `disc`.
- `sam3_hf_auto.py`     SAM 3 automatic masks through the Hugging Face `mask-generation` pipeline (transformers 5.14.0), loose NMS/thresholds.
                        Separate env (Mac: `sam3hf`, python 3.12, torch 2.8); `[python] <script> <out_dir> [cpu|cuda]`. Gated weights: `hf auth login`.
All three turn the masks into a label map with build_fine_segments.py's own build_label_map (min_pixels_in_object 50, as the worker).
