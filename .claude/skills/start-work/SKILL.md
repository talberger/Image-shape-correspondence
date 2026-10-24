---
name: start-work
description: Open a DISC work session on any machine (Tal's Mac or a RunPod / Run:ai GPU machine) — git pull, activate the env, pod setup checks, move pod Obsidian notes into the vault (Mac), read the top of PROJECT_NOTES.md and summarize where things stand. Use when Tal says /start-work, "start work", "פתיחת עבודה", or asks to pick up where the last session left off.
---

# Start a work session (DISC)

Reply to Tal in short Hebrew. Do the steps in order; stop and report if one fails.

## 1. Which machine
- **Mac:** the repo is at `~/Projects/disc` (no GPU; analysis, figures, Obsidian).
- **Pod / GPU machine:** the repo is at `/workspace/disc` (`/workspace` is the persistent volume).
Say which one in the first line of the summary.

## 2. Code and env
- Activate FIRST, in the same shell command as the pull: Mac `source env/activate.sh`; pod `source /workspace/activate_bsb.sh`
  (it installs the deploy key; without it pull fails with "Please make sure you have the correct access rights").
- `cd <repo> && <activate> && git pull --ff-only`. If it fails (local changes, diverged branch), STOP and show `git status` -- do not stash, reset or merge on your own.
- Shell state does not carry between tool calls: re-source the activate script in any later command that needs `$DISC_WS` or the env.
- `git status -s`: report anything uncommitted that was left over from before.

## 3. Pod only
- Skills link (one-time per volume): if `/workspace/.claude/skills` is missing, run
  `mkdir -p /workspace/.claude && ln -s /workspace/disc/.claude/skills /workspace/.claude/skills`
  (Tal starts Claude from `/workspace`, which would not see the repo's skills otherwise).
- Local env copy (fast imports): start `setsid nohup bash env/pod_local_env.sh > /workspace/logs/pod_local_env.log 2>&1 < /dev/null &`
  right away; it takes ~1-2 min, and `env/activate.sh` switches to `/root/disc_env` once `READY` exists. Report its last line at the end.
- `nvidia-smi` (GPU model + memory, or "no GPU") and `df -h /workspace`.
- Third-party clones: if `$DISC_WS/sam3_repo` / `$DISC_WS/pix2gestalt_repo` exist, make sure our scripts from `external/` are copied in
  (see `external/README.md`).
- Check nothing from an earlier session is still running (`ps aux | grep -E "python|train" | grep -v grep`) and say so.

## 3b. Run:ai (only if Tal wants a GPU there; driven FROM the Mac session)
TAU VPN on (RunPod SSH then times out). `export PATH=$HOME/.runai/bin:$PATH`; `runai workspace list -p avidan` (if "token has expired":
Tal runs `runai login`). Start with `runai workspace resume tberger3-disc-gpu -p avidan` (image pull ~2 min; the web UI can lag).
`runai workspace exec tberger3-disc-gpu -p avidan -- bash -c 'cd /workspace/disc && git pull --ff-only && nvidia-smi'`. Data comes from the
RunPod volume over S3 (see "THREE MACHINES" in the notes). Long jobs: `setsid nohup ... > /workspace/logs/<x>.log 2>&1 < /dev/null &`.

## 4. Mac only: data from the pod
`git pull` brings code and notes only; `data/` and `demo/` are gitignored. If the top notes entry (written by a pod's `/end-work`) has a
`SYNC TO MAC:` line, copy those paths now:
- Check the pod is up, with exactly `ssh -o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=accept-new runpod "echo CONNECTED && whoami"`.
- If it connects: read the address with `ssh -G runpod | awk '/^hostname |^port /'` and run
  `bash scripts/sync_from_pod.sh <hostname> <port> <the listed paths>` (pull-only, copies only what changed; long ones in the background).
  The Mac shell is zsh: it does NOT word-split `$P`, so pass the paths as separate words (`${=P}`, or paste them literally).
- If it does not connect: the pod is probably closed, or the TAU VPN is on (VPN on = RunPod SSH times out). Say which paths were NOT
  copied and go on; do not retry with other commands.

## 5. Mac only: Obsidian inbox
If `workspace_meta/obsidian_inbox/` has notes from a pod: move each into the vault `~/Documents/DISC` following the
"OBSIDIAN NOTE STYLE" rule in PROJECT_NOTES.md (Hebrew, short bullets, figures embedded; images into `Experiments/attachments/` with an
`lvis<id>_<topic>_` prefix; a link in `DISC.md`). If the note continues an existing experiment, update that note instead of making a new one.
Then `git rm` the inbox files, mark it done in the notes entry that created them, commit + push.

## 6. Where we stand
Read `workspace_meta/PROJECT_NOTES.md`: the short header and the TOP one or two dated entries (not the whole file, it is ~4000 lines),
plus the "How Tal wants the assistant to work" section if this session has not seen it. Then tell Tal, in 3-4 short lines:
- what the last session did and its result,
- what is running (if anything),
- the open NEXT items, and any "REMIND TAL" item that applies,
and ask what to continue with. Do not start work before Tal answers.
