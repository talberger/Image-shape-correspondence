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
- `cd <repo> && git pull --ff-only`. If it fails (local changes, diverged branch), STOP and show `git status` -- do not stash, reset or merge on your own.
- Activate: Mac `source env/activate.sh`; pod `source /workspace/activate_bsb.sh` (it also installs the deploy key that push needs).
- `git status -s`: report anything uncommitted that was left over from before.

## 3. Pod only
- Skills link (one-time per volume): if `/workspace/.claude/skills` is missing, run
  `mkdir -p /workspace/.claude && ln -s /workspace/disc/.claude/skills /workspace/.claude/skills`
  (Tal starts Claude from `/workspace`, which would not see the repo's skills otherwise).
- `nvidia-smi` (GPU model + memory, or "no GPU") and `df -h /workspace`.
- Third-party clones: if `$DISC_WS/sam3_repo` / `$DISC_WS/pix2gestalt_repo` exist, make sure our scripts from `external/` are copied in
  (see `external/README.md`).
- Check nothing from an earlier session is still running (`ps aux | grep -E "python|train" | grep -v grep`) and say so.

## 4. Mac only: Obsidian inbox
If `workspace_meta/obsidian_inbox/` has notes from a pod: move each into the vault `~/Documents/DISC` following the
"OBSIDIAN NOTE STYLE" rule in PROJECT_NOTES.md (Hebrew, short bullets, figures embedded; images into `Experiments/attachments/` with an
`lvis<id>_<topic>_` prefix; a link in `DISC.md`). If the note continues an existing experiment, update that note instead of making a new one.
Then `git rm` the inbox files, mark it done in the notes entry that created them, commit + push.

## 5. Where we stand
Read `workspace_meta/PROJECT_NOTES.md`: the short header and the TOP one or two dated entries (not the whole file, it is ~4000 lines),
plus the "How Tal wants the assistant to work" section if this session has not seen it. Then tell Tal, in 3-4 short lines:
- what the last session did and its result,
- what is running (if anything),
- the open NEXT items, and any "REMIND TAL" item that applies,
and ask what to continue with. Do not start work before Tal answers.
