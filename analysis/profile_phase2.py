"""Where does a phase-2 training step spend its time? (Tal, 2026-10-08.) Replays a kept run's exact training command from its
run_config.json for a few steps under cProfile, with CUDA_LAUNCH_BLOCKING=1 so GPU time is charged to the Python line that asked for it.
Nothing in the kept run is touched (out_dir is redirected).
Usage: python analysis/profile_phase2.py --run_config <run>/encoder_loraW_all_r4_phase2/run_config.json --out_dir <scratch> [--num_iters 40]"""
import os; os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse, cProfile, io, json, pstats, runpy, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
ap = argparse.ArgumentParser(); ap.add_argument('--run_config', required=True); ap.add_argument('--out_dir', required=True)
ap.add_argument('--num_iters', type=int, default=40); ap.add_argument('--top', type=int, default=45)
a = ap.parse_args()
import phase2.train_relational_loss_l2norm  # noqa: F401  (the worker imports it before running the stage; same here)
c = json.load(open(a.run_config)); c['out_dir'] = a.out_dir; c['num_iters'] = a.num_iters; c['no_step_checkpoints'] = True; c['save_stageA_end'] = False
c['mu_switch_step'] = a.num_iters // 2   # both stages inside the profiled window
src = open(os.path.join(REPO, 'phase2/train_relational_loss.py')).read()
argv = []
for k, v in c.items():
    if f"'--{k}'" not in src or v is None or v is False: continue
    argv += [f'--{k}'] if v is True else [f'--{k}'] + ([str(x) for x in v] if isinstance(v, list) else [str(v)])
sys.argv = ['phase2/train_relational_loss.py'] + argv; os.chdir(REPO)
pr = cProfile.Profile(); t0 = time.time(); pr.enable()
try:
    runpy.run_path(os.path.join(REPO, 'phase2/train_relational_loss.py'), run_name='__main__')
finally:
    pr.disable(); print(f'\nTOTAL wall {time.time() - t0:.1f} s for {a.num_iters} steps (+ setup)')
    for key in ('cumulative', 'tottime'):
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats(key).print_stats(a.top); print(f'===== sorted by {key} =====\n' + s.getvalue())
    pr.dump_stats(os.path.join(a.out_dir, 'phase2.prof'))
