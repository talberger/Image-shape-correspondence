"""One persistent process that runs the whole per-instance DensePose-LVIS pipeline for a LIST of
instances, instead of scripts/run_lvis_instance.sh's seven separate python processes per instance.

WHY (measured 2026-09-20, this pod): almost nothing in this pipeline is compute. Per instance, the
seven processes pay ~52-56s EACH just to `import torch` and friends before doing any work -- roughly
330s of the ~16min instance, i.e. a third of the whole 132-instance run spent re-importing the same
modules. Python caches imports in sys.modules, so running the identical stages inside ONE process
makes every import after the first free.

HOW, and why there is no refactor of the training code: every stage already has an
`if __name__ == '__main__':` argparse block, so `runpy.run_path(..., run_name='__main__')` with
sys.argv set runs it exactly as the shell does -- same code path, same arguments, no duplicated
logic and nothing reimplemented here. phase2/train_relational_loss_l2norm.py already uses this exact
pattern, so it is proven, not novel. This file is a driver, not a reimplementation.

CROSS-INSTANCE INDEPENDENCE (Tal's question, 2026-09-20, and the reason this needs care):
an instance = one image + the 3D shape, and instance i must not influence instance j in any way.
Sharing a process is only safe because:
  * The phase-2 encoder is rebuilt from the phase-1 checkpoint on every call -- load_phase1_encoder
    runs per instance, inside train(). No warm encoder is ever carried forward.
  * Both RNGs are re-seeded per instance by the existing code, not by anything here:
    load_phase1_encoder calls seed_everything(args.seed) (torch, cuda, random, numpy, deterministic
    cuDNN) and train() calls torch.manual_seed(args.train_seed) at its start. So instance i does not
    inherit instance i-1's RNG state, which is the one way ordering could have leaked in silently.
  * What IS shared is frozen and read-only: the phase-1 checkpoint, the 3D geodesic table, the
    228-normalized geodesic matrix, the mesh. None is ever written.
This is an argument, not a proof, so it is checked empirically -- see the ORDER-INDEPENDENCE test in
this module's __main__ help and PROJECT_NOTES: the same instances run through this worker and
through the shell script must produce bit-identical checkpoints, and running A,B must equal B,A.

Run:
cd $DISC_ROOT && python scripts/run_lvis_worker.py --ann_ids 117683 1234 --results_dir data/densepose_lvis/results
"""
import argparse
import gc
import glob
import json
import os
import runpy
import subprocess
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch  # noqa: E402  (after the sys.path fix, like every analysis/ script)

# Importing this applies its module-level patch of train_mapping_loss.load_phase1_encoder to the
# EncoderL2Norm-aware loader -- the same thing running that file as a script does. Phase-2 is then
# driven by runpy'ing train_relational_loss.py itself, so the patch is in place exactly as it would
# be under `python phase2/train_relational_loss_l2norm.py`.
import phase2.train_relational_loss_l2norm  # noqa: E402,F401

CLASS = 'bear'


def set_class(cls):
    """2026-10-04 (Tal: full horse eval): every per-class path, set once from --class_name. Assets come from
    scripts/run_shicclass_setup.sh (phase 1, heat geodesics) and scripts/run_lvis_prep.sh (3D geo table)."""
    global CLASS, OBJ, NAME, P1, GEO3D, GEO228
    CLASS = cls
    OBJ = f'./demo/shic/shapes/{cls}.obj'
    NAME = f'shic_{cls}'
    P1 = f'./experiments/shic_{cls}/encoder_dfd_crossview_phase1'
    GEO3D = f'./experiments/shic_{cls}/geodesic_tables/shic_{cls}_geo_table_3d.pth'
    GEO228 = f'data/densepose_lvis/eval_shic_{cls}/{cls}_heat_geodesics_228.npy'


set_class('bear')


# cuDNN's flags are PROCESS-GLOBAL and sticky, and this is the one real way instance i was found to
# leak into instance i+1 (2026-09-20, measured, not theorised). phase-2 calls seed_everything(),
# which sets cudnn.deterministic=True and never restores it; the NEXT instance's DINO/SD feature
# extraction then picks different conv algorithms than a fresh process would. Probe: the same image
# extracted twice in one process, differing only in that flag, gave maxdiff 2.362e-04 over 56% of
# the feature map -- exactly the discrepancy measured between this worker's second instance and a
# standalone run of it. Captured here at import, before anything has touched them, and restored
# before every stage so each stage starts from the state a fresh process would have.
_TORCH_DEFAULTS = {
    'cudnn_benchmark': torch.backends.cudnn.benchmark,
    'cudnn_deterministic': torch.backends.cudnn.deterministic,
    'cudnn_enabled': torch.backends.cudnn.enabled,
}


def reset_global_torch_state():
    torch.backends.cudnn.benchmark = _TORCH_DEFAULTS['cudnn_benchmark']
    torch.backends.cudnn.deterministic = _TORCH_DEFAULTS['cudnn_deterministic']
    torch.backends.cudnn.enabled = _TORCH_DEFAULTS['cudnn_enabled']


def stage(script, argv, timings, label=None):
    """Run one pipeline stage in-process, exactly as the shell would run it as a script.

    sys.path[0] is swapped to the script's OWN directory for the duration, because that is what
    `python analysis/foo.py` does and some stages depend on it: analysis/prep_lvis_instance.py does
    a sibling `from eval_densepose_lvis import ...`, which resolves only when analysis/ is on the
    path. runpy.run_path does not set that up, so without this the stage raises ModuleNotFoundError
    (found by running this worker, not by reading it). Restored afterwards so one stage's path
    cannot leak into the next.
    """
    t0 = time.time()
    script_dir = os.path.dirname(os.path.join(REPO, script))
    saved_path, saved_argv = sys.path[:], sys.argv[:]
    sys.path.insert(0, script_dir)
    sys.argv = [script] + [str(a) for a in argv]
    reset_global_torch_state()
    try:
        runpy.run_path(os.path.join(REPO, script), run_name='__main__')
    finally:
        sys.path[:], sys.argv[:] = saved_path, saved_argv
        # Between stages only -- never mid-stage, so this cannot perturb anything a stage computes.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    dt = time.time() - t0
    timings.append((label or script, round(dt, 1)))
    print(f'[time] {dt:.0f} s : {label or script}', flush=True)


def stage_subprocess(script, argv, timings, label=None):
    """Like stage(), but in a fresh `python <script>` process with OMP_NUM_THREADS=1 (see GEO_TABLE_SUBPROCESS)."""
    t0 = time.time()
    subprocess.run([sys.executable, os.path.join(REPO, script)] + [str(a) for a in argv], check=True, cwd=REPO,
                   env={**os.environ, 'OMP_NUM_THREADS': '1'})
    dt = time.time() - t0
    timings.append((label or script, round(dt, 1)))
    print(f'[time] {dt:.0f} s : {label or script} (subprocess)', flush=True)


def run_instance(ann_id, args):
    tag = f'lvis{ann_id}'
    d = f'{args.runs_dir}/{tag}'
    # --no_lora runs the MAIN recipe as it was BEFORE LORA-W was added (rkd_spatial_dist + learned
    # per-segment mu + 16 in-segment neighbours + cycle v2, T=0.01): the whole encoder trains, there
    # is no frozen-delta and therefore no read-time alpha, so pred_f comes straight out of phase-2.
    p2 = f'{d}/encoder_main_nolora_phase2' if args.no_lora else f'{d}/encoder_loraW_all_r4_phase2'
    # stage 1 = the global-ellipse stage; only two-stage LoRA runs have one (default on since 2026-10-08, --no_stage1 = off)
    stage1 = args.mu_scope == 'global_then_segment_aniso' and not args.no_lora and not args.no_stage1
    out_json = os.path.join(args.results_dir, f'{tag}.json')
    timings = []

    os.makedirs(d, exist_ok=True)
    stage('analysis/prep_lvis_instance.py', ['--ann_id', ann_id, '--class_name', CLASS, '--out_dir', d, '--long_side', 400,
                                             '--mask_source', args.mask_source], timings, 'prep')
    # --amodal_mask (Tal, 2026-10-07): {d}/{tag}_amodal_mask.pth must already exist (it is made in another env,
    # scripts/make_amodal_mask.sh); prep above only rewrites the crop / mask / gt next to it.
    # FINE-SEGMENT CACHE (Tal, 2026-10-11): the fine segments depend only on the crop, the object mask and these two options, so a
    # later run of the same instance copies them instead of running SAM again. The key carries the mask source and both options.
    fs_key = (f'{args.fine_seg_cache}/{tag}_{args.mask_source}_merge{args.merge_below_frac:g}'
              f'_fill{int(args.fill_unlabelled_segments)}') if args.fine_seg_cache else None
    if fs_key and os.path.exists(f'{fs_key}_fine_segments.pth'):
        for ext in ('pth', 'png'):
            if os.path.exists(f'{fs_key}_fine_segments.{ext}'):
                shutil.copyfile(f'{fs_key}_fine_segments.{ext}', f'{d}/{tag}_fine_segments.{ext}')
        timings.append(('fine_segments', 0.0))
        print(f'[time] 0 s : fine_segments (from the cache {fs_key})', flush=True)
    else:
        stage('analysis/build_fine_segments.py',
              ['--img_path', f'{d}/{tag}.png', '--mask_path', f'{d}/{tag}_mask.pth', '--save_dir', d,
               '--min_pixels_in_object', 50,
               *(['--merge_below_frac', args.merge_below_frac] if args.merge_below_frac > 0 else []),
               *(['--fill_unlabelled'] if args.fill_unlabelled_segments else [])],
              timings, 'fine_segments')
        if fs_key:
            os.makedirs(args.fine_seg_cache, exist_ok=True)
            for ext in ('pth', 'png'):
                if os.path.exists(f'{d}/{tag}_fine_segments.{ext}'):
                    shutil.copyfile(f'{d}/{tag}_fine_segments.{ext}', f'{fs_key}_fine_segments.{ext}')
    if args.features_wait > 0:
        # 2026-10-08 (speed): the features need only the crop, not the mask, so the caller (scripts/debug_twostage_instance.sh) extracts
        # them in another process while SAM 3 builds the mask -- same command as the stage below. Wait for its marker instead.
        t_w = time.time()
        while not os.path.exists(f'{d}/{tag}_features.done'):
            if os.path.exists(f'{d}/{tag}_features.failed'):
                raise RuntimeError(f'--features_wait: the feature extraction failed ({d}/{tag}_features.failed)')
            if time.time() - t_w > args.features_wait:
                raise TimeoutError(f'--features_wait: no {d}/{tag}_features.done after {args.features_wait} s')
            time.sleep(2)
        timings.append(('features', round(time.time() - t_w, 1)))
        print(f'[time] {timings[-1][1]:.0f} s : features (precomputed by the caller, waited)', flush=True)
    else:
        stage('analysis/extract_dino_sd_features.py',
              ['--img_path', f'{d}/{tag}.png', '--save_dir', d,
               '--output_path', f'{d}/{tag}_pix_sdsquare_fuse16.pth',
               '--sd_frame', 'square_pad', '--cpu_postprocess'],
              timings, 'features')
    # 2026-10-08 (speed): the features stage moved ABOVE the geodesic table (independent of it, same outputs) so that the amodal mask
    # can be built by another process meanwhile. --amodal_wait S: wait up to S seconds for {tag}_amodal.done (written by
    # scripts/make_amodal_mask.sh when it finishes OR decides the mask is in one piece and skips); then use the mask if it exists.
    amodal_path = f'{d}/{tag}_amodal_mask.pth'
    if args.amodal_mask and args.amodal_wait > 0:
        t_w = time.time()
        while not os.path.exists(f'{d}/{tag}_amodal.done'):
            if os.path.exists(f'{d}/{tag}_amodal.failed'):
                raise RuntimeError(f'--amodal_wait: the amodal mask step failed ({d}/{tag}_amodal.failed)')
            if time.time() - t_w > args.amodal_wait:
                raise TimeoutError(f'--amodal_wait: no {d}/{tag}_amodal.done after {args.amodal_wait} s')
            time.sleep(2)
        timings.append(('amodal_wait', round(time.time() - t_w, 1)))
        print(f'[time] {timings[-1][1]:.0f} s : amodal_wait ({"mask ready" if os.path.exists(amodal_path) else "not needed, mask in one piece"})', flush=True)
        amodal = ['--geodesic_domain_mask_path', amodal_path] if os.path.exists(amodal_path) else []
    else:
        amodal = ['--geodesic_domain_mask_path', amodal_path] if args.amodal_mask else []
        if args.amodal_mask and not os.path.exists(amodal_path):
            raise FileNotFoundError(f'--amodal_mask: {amodal_path} not found')
    # GEO_TABLE_SUBPROCESS=1 (2026-10-24, Run:ai): run this stage as a FRESH process with OMP_NUM_THREADS=1. In-process, its fork()ed
    # Dijkstra pool deadlocked on the Run:ai image (pip torch's libgomp: 102 threads in the parent after the features stage, every child
    # stuck in futex_wait with 0 s CPU). Same command, same output; the pods (conda torch) never hit it, so it stays opt-in.
    (stage_subprocess if os.environ.get('GEO_TABLE_SUBPROCESS') == '1' else stage)('analysis/build_geodesic_tables.py',
          ['--obj_path', OBJ, '--name', NAME, '--skip_3d',
           '--object_mask_path', f'{d}/{tag}_mask.pth',
           '--fine_seg_path', f'{d}/{tag}_fine_segments.pth',
           *amodal,
           *(['--feature_geodesic', *args.feature_geodesic, '--pix_features_path', f'{d}/{tag}_pix_sdsquare_fuse16.pth']
             if args.feature_geodesic else []),
           '--num_center_pixels', 2048, '--seed', 0, '--true_diameter', '--out_dir', d],
          timings, 'geo_table_2d')

    # geo_diag_2d comes from the sidecar the stage above writes, not from re-loading the 2.6GB table.
    meta_path = f'{d}/{tag}_mask_geo_table_2d_seed0_meta.json'
    with open(meta_path) as f:
        meta = json.load(f)
    # --feature_geodesic_diameter path (Tal, 2026-10-13): normalise the feature-geodesic table by ITS OWN longest path (pixels, max over
    # the table) instead of the original pixel diameter, so norm_d_2D is back in 0..1.
    diag = meta['diagonal'] if (args.feature_geodesic and args.feature_geodesic_diameter == 'path') else meta['true_diameter_2d']
    if diag is None:
        raise RuntimeError(f'{meta_path} has no true_diameter_2d -- --true_diameter did not run')
    print(f'geo_diag_2d = {diag}', flush=True)

    # ONE FINE SEGMENT = THE WHOLE MASK (Tal, 2026-10-09; bear 181519 after merge 1.5%): there is no "out of segment" pixel, so
    # stage 1 used to raise. Run ONLY stage 1 (the global ellipse, steps 1 .. mu_switch_step - 1) with all 16 neighbours drawn
    # inside the one segment; stage 2 is skipped, final = end of stage 1 (arm "stageA" = arm "ours").
    seg_labels = torch.load(f'{d}/{tag}_fine_segments.pth', map_location='cpu')
    single_seg = (args.mu_scope == 'global_then_segment_aniso'
                  and int((torch.unique(torch.as_tensor(seg_labels)) > 0).sum()) == 1)
    if single_seg:
        print(f'{tag}: ONE fine segment -> stage 1 only, 16 in-segment neighbours, {args.mu_switch_step - 1} steps', flush=True)
    stage('phase2/train_relational_loss.py',
          ['--obj_path', OBJ, '--name', NAME, '--init_encoder_dir', P1,
           '--object_mask_path', f'{d}/{tag}_mask.pth',
           '--fine_seg_path', f'{d}/{tag}_fine_segments.pth',
           '--pix_dino_features_path', f'{d}/{tag}_pix_sdsquare_fuse16.pth',
           '--width', 2304, '--n_classes', 2304, '--geodesic', '--temperature', 0.01,
           '--train_seed', 0, '--geo_diag_2d', diag,
           '--geo_table_3d_path', GEO3D,
           '--geo_table_2d_path', f'{d}/{tag}_mask_geo_table_2d_seed0.pth',
           '--mu_mode', 'learned', '--mu_scope', args.mu_scope, '--mu_lr', args.mu_lr,
           '--mu_fit_space', 'abs',
           *(['--num_neighbors_in_segment', 16, '--num_neighbors_out_segment', 0,
              '--mu_switch_step', args.mu_switch_step, '--switch_nin', 16, '--switch_nout', 0,
              '--num_iters', args.mu_switch_step - 1]
             if single_seg else
             ['--num_neighbors_in_segment', 8, '--num_neighbors_out_segment', 8,
              '--mu_switch_step', args.mu_switch_step, '--switch_nin', 16, '--switch_nout', 0]
             if args.mu_scope == 'global_then_segment_aniso' else
             ['--num_neighbors_in_segment', 16, '--num_neighbors_out_segment', 0]),
           '--cyc_weight', 1.676e-05, '--cyc_v2', '--cyc_temperature', 0.05,
           '--cyc_return_all_pixels', '--cyc_return_stride', 3,
           *([] if args.no_lora else
             ['--lora_w_rank', 4, '--lora_w_layers', 'all', '--lora_w_lr', 1e-3]),
           *(['--out_neighbors_segmented_only'] if args.out_neighbors_segmented_only else []),
           *amodal,
           *([] if args.step_checkpoints else ['--no_step_checkpoints']),
           *(['--save_stageA_end'] if stage1 else []), '--out_dir', p2],
          timings, 'phase2')
    if single_seg and stage1:   # the switch step is never reached, so the end-of-stage-1 state IS the final checkpoint
        shutil.copyfile(f'{p2}/encoder_checkpoint.pth', f'{p2}/encoder_checkpoint_stageA_end.pth')
        if os.path.exists(f'{p2}/mu_aniso.pth'):
            shutil.copyfile(f'{p2}/mu_aniso.pth', f'{p2}/mu_aniso_stageA_end.pth')
    if args.no_lora:
        ours_pred_f = f'{p2}/pred_f.pth'
    else:
        ours_pred_f = f'{p2}/alpha/alpha{args.alpha:g}/pred_f.pth'
        stage('analysis/export_pred_f_lora_alpha.py',
              ['--obj_path', OBJ, '--name', NAME,
               '--phase1_checkpoint', f'{P1}/encoder_checkpoint.pth',
               '--lora_checkpoint', f'{p2}/encoder_checkpoint.pth',
               '--alphas', args.alpha, '--out_dir', f'{p2}/alpha'],
              timings, 'alpha_export')
        if stage1:
            stage('analysis/export_pred_f_lora_alpha.py',
                  ['--obj_path', OBJ, '--name', NAME,
                   '--phase1_checkpoint', f'{P1}/encoder_checkpoint.pth',
                   '--lora_checkpoint', f'{p2}/encoder_checkpoint_stageA_end.pth',
                   '--alphas', args.alpha, '--out_dir', f'{p2}/alpha_stageA_end'],
                  timings, 'alpha_export_stageA')
    stage('analysis/eval_lvis_ours.py',
          ['--run_dir', d, '--tag', tag, '--obj_path', OBJ, '--name', NAME,
           '--geodesics_path', GEO228, '--out_json', out_json,
           '--arm', f'ours={ours_pred_f}',
           '--arm', f'zs={P1}/pred_f.pth',
           *(['--arm', f'stageA={p2}/alpha_stageA_end/alpha{args.alpha:g}/pred_f.pth'] if stage1 else [])],
          timings, 'eval')

    with open(out_json) as f:
        result = json.load(f)
    result['timings'] = timings
    result['single_segment_stage1_only'] = bool(single_seg)
    result['worker'] = 'run_lvis_worker.py'
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=1)

    if args.delete_run_dir:
        shutil.rmtree(d, ignore_errors=True)
        print(f'deleted intermediates {d}', flush=True)
    elif not args.keep_intermediates:
        # SLIM KEEP (Tal, 2026-10-10; default): figures later must not need a retrain. Keep the training RESULT (the alpha pred_f of
        # stage 1 + final, the ellipses) and the small inputs (crop, masks, segments, gt): ~95 MB. Drop what is recomputed without
        # training (photo features ~1 GB, ~1 min) or is only needed to train / to export another alpha (2D geodesic table, encoder
        # checkpoints, the raw pred_f). ANALYSIS_ONLY=1 in scripts/debug_twostage_instance.sh re-extracts the features.
        for f in (glob.glob(f'{d}/{tag}_pix_sdsquare_fuse16.pth') + glob.glob(f'{d}/{tag}_mask_geo_table_2d_seed0.pth')
                  + glob.glob(f'{p2}/encoder_checkpoint*.pth') + glob.glob(f'{p2}/pred_f.pth')):
            os.remove(f)
        print(f'slim-kept {d} (features, 2D geodesic table, encoder checkpoints removed)', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann_ids', nargs='+', required=True, type=int,
                    help='the instances this worker owns (split the 132 across GPUs by giving each '
                         'worker a disjoint list and its own CUDA_VISIBLE_DEVICES)')
    ap.add_argument('--results_dir', default='data/densepose_lvis/results')
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--keep_intermediates', action='store_true',
                    help='keep EVERYTHING (~2 GB/instance: features, 2D geodesic table, encoder checkpoints). '
                         'Default since 2026-10-10 is the slim keep (~95 MB: alpha pred_f of stage 1 + final, masks, segments).')
    ap.add_argument('--fine_seg_cache', default=None,
                    help='2026-10-11: directory of cached fine segments (keyed by instance, mask source, merge, fill); '
                         'read if present, written otherwise. Default: no cache.')
    ap.add_argument('--delete_run_dir', action='store_true',
                    help='delete the whole run dir after scoring (the default before 2026-10-10)')
    ap.add_argument('--no_lora', action='store_true',
                    help='train the MAIN recipe WITHOUT LORA-W (the pre-LORA arm: rkd_spatial_dist '
                         '+ learned per-segment mu + 16 in-segment + cycle v2 at T=0.01). The whole '
                         'encoder trains, so there is no read-time alpha and pred_f is read straight '
                         'from the phase-2 dir. Use a separate --results_dir: the per-instance JSON '
                         'is named by tag alone and would otherwise overwrite the LORA arm\'s.')
    ap.add_argument('--runs_dir', default='data/densepose_lvis/runs',
                    help='where per-instance intermediates live (and are DELETED after scoring). 2026-09-22: point a '
                         'new full run elsewhere so it cannot delete the kept runs/lvis{3044,7108,...} dirs')
    ap.add_argument('--out_neighbors_segmented_only', action='store_true',
                    help='train flag of the same name (Tal, 2026-10-06); off = every eval before')
    ap.add_argument('--fill_unlabelled_segments', action='store_true',
                    help='build_fine_segments.py --fill_unlabelled (Tal, 2026-10-07, trial); off = every eval before')
    ap.add_argument('--amodal_mask', action='store_true',
                    help='occluded objects (Tal, 2026-10-07): measure 2D geodesics through {runs_dir}/lvis<id>/lvis<id>_amodal_mask.pth '
                         '(must exist beforehand); centres/neighbours stay on the visible mask. Off = every eval before')
    ap.add_argument('--stage1_checkpoint', action='store_true',
                    help='(2026-10-07) kept for old command lines; since 2026-10-08 the end-of-stage-A save is the default, see --no_stage1')
    ap.add_argument('--features_wait', type=float, default=0,
                    help='(2026-10-08) the pixel features are being extracted by another process; wait up to this many seconds for '
                         'lvis<id>_features.done instead of running the features stage. 0 = run the stage here (every run before)')
    ap.add_argument('--amodal_wait', type=float, default=0,
                    help='with --amodal_mask (2026-10-08): the amodal mask is being built by another process; wait up to this many seconds '
                         'for lvis<id>_amodal.done before the geodesic table. 0 = the mask must already exist (every run before)')
    ap.add_argument('--mask_source', default='polygon', choices=['polygon', 'sam3'],
                    help='sam3 (Tal, 2026-10-08): the object mask is {runs_dir}/lvis<id>/lvis<id>_sam3_mask.pth instead of the LVIS polygon '
                         '(scripts/make_sam3_mask.sh, must exist beforehand). polygon = every eval before')
    ap.add_argument('--feature_geodesic', type=float, nargs=2, default=None, metavar=('BETA', 'POWER'),
                    help='2D geodesic table with the feature step cost (build_geodesic_tables.py --feature_geodesic); default = the original metric')
    ap.add_argument('--feature_geodesic_diameter', default='original', choices=['original', 'path'],
                    help='--feature_geodesic only: divide the table by the original pixel diameter, or by its own longest path (px)')
    ap.add_argument('--no_stage1', action='store_true',
                    help='Tal, 2026-10-08: every two-stage LoRA run saves the end-of-stage-A state and scores it as arm "stageA" in the '
                         'result json (so "stage 2 better than stage 1?" survives the cleanup). This flag turns that off = every eval before')
    ap.add_argument('--step_checkpoints', action='store_true',
                    help='debug (2026-10-06): keep the step checkpoints + the end-of-stage-A save (encoder_checkpoint_stageA_end.pth, mu_aniso_stageA_end.pth)')
    ap.add_argument('--merge_below_frac', type=float, default=0.0,
                    help='build_fine_segments.py --merge_below_frac (Tal, 2026-10-06: 0.007); 0 = off, as every eval before')
    ap.add_argument('--mu_switch_step', type=int, default=150,
                    help='global_then_segment_aniso (2026-09-29): stage A = learned global ellipse, 8 in + 8 out, until this step; '
                         'stage B = learned per-seg ellipses, 16 in + 0 out (Tal\'s best recipe on the horse)')
    ap.add_argument('--mu_scope', default='segment', choices=['segment', 'segment_aniso', 'global_then_segment_aniso'],
                    help='2026-09-22: segment_aniso = directional per-segment mu (Tal). Default = the '
                         'iso recipe of the first full run. Use a separate --results_dir.')
    ap.add_argument('--mu_lr', type=float, default=1e-2,
                    help='default 1e-2 = the first full run; Tal adopted 5e-2 for directional mu (2026-09-22)')
    ap.add_argument('--class_name', default='bear', choices=['bear', 'horse', 'elephant'],
                    help='2026-10-04: DensePose-LVIS class (default bear, unchanged). Use a separate --results_dir/--runs_dir.')
    ap.add_argument('--redo', action='store_true',
                    help='rescore instances that already have a results JSON (default: skip them, '
                         'so a killed queue can simply be relaunched with the same --ann_ids)')
    args = ap.parse_args()
    set_class(args.class_name)
    os.makedirs(args.results_dir, exist_ok=True)

    done, failed = [], []
    for i, ann_id in enumerate(args.ann_ids, 1):
        out_json = os.path.join(args.results_dir, f'lvis{ann_id}.json')
        if os.path.exists(out_json) and not args.redo:
            print(f'=== [{i}/{len(args.ann_ids)}] ann {ann_id}: already scored, skipping', flush=True)
            done.append(ann_id)
            continue
        print(f'=== [{i}/{len(args.ann_ids)}] ann {ann_id} ===', flush=True)
        t0 = time.time()
        try:
            run_instance(ann_id, args)
            done.append(ann_id)
            print(f'=== ann {ann_id} DONE in {time.time() - t0:.0f}s ===', flush=True)
        except Exception as e:                                   # noqa: BLE001
            # One bad instance (a degenerate mask, a missing image) must not take the queue with it.
            import traceback
            traceback.print_exc()
            failed.append((ann_id, repr(e)))
            print(f'=== ann {ann_id} FAILED: {e!r} ===', flush=True)

    print(f'WORKER DONE: {len(done)} scored, {len(failed)} failed', flush=True)
    for ann_id, err in failed:
        print(f'  FAILED {ann_id}: {err}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
