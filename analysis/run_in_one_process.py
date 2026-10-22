# Run several analysis scripts in ONE Python process, so torch / pytorch3d are imported once instead of once per script
# (Tal 2026-10-22: on a slow /workspace volume `import torch` alone takes ~60 s, and an XSEG arm ran 4 analysis scripts).
# Usage: python analysis/run_in_one_process.py <script.py> <args...> --- <script.py> <args...> --- ...
# Each script runs as __main__ with its own sys.argv, in order; the first failure (non-zero exit / exception) stops the chain.
import os, runpy, sys, time

cmds, cur = [], []
for tok in sys.argv[1:]:
    if tok == '---':
        cmds.append(cur); cur = []
    else:
        cur.append(tok)
cmds.append(cur)
cmds = [c for c in cmds if c]
if not cmds:
    sys.exit('usage: run_in_one_process.py <script.py> <args...> --- <script.py> <args...> ...')

for c in cmds:
    t0 = time.time()
    print('### run_in_one_process: %s' % ' '.join(c), flush=True)
    sys.argv = list(c)
    sys.path.insert(0, os.path.dirname(os.path.abspath(c[0])))   # what `python <script>` puts first on the path
    try:
        runpy.run_path(c[0], run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    finally:
        sys.path.pop(0)
    print('### done %s in %.0f s' % (os.path.basename(c[0]), time.time() - t0), flush=True)
