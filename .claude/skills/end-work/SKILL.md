---
name: end-work
description: Close a DISC work session on any machine (Tal's Mac or a RunPod / Run:ai GPU machine) — check what is still running, write the PROJECT_NOTES.md entry, write the Obsidian summary (vault on the Mac, inbox on a pod), list data worth syncing to the Mac, git commit + push. Use when Tal says /end-work, "end work", "סגירת עבודה", or that he is closing the pod / finishing for now.
---

# End a work session (DISC)

Reply to Tal in short Hebrew. Machine: repo at `~/Projects/disc` = Mac, `/workspace/disc` = pod.

## 1. What is still running
Check this session's background jobs and, on a pod, `ps aux | grep -E "python|train" | grep -v grep` + `nvidia-smi`.
If something runs, tell Tal before going on (a pod that closes kills it); record it in the notes with its log path and how to check it.

## 2. PROJECT_NOTES.md
`git pull --ff-only` first (the other machine may have pushed). Add ONE new entry at the top, under the header, titled
`## STATE AT CLOSE — <date> (<machine: Mac / pod number + GPU>) — <topic>. **<what is running, or NOTHING IS RUNNING>**`. Content:
- what was done and decided (Tal's decisions marked as his), with the result numbers in a small table when there are any,
- code changed (files, flags), commit hash once known, scripts / commands to reproduce,
- where outputs live (figures, runs, logs); remember `data/` and `demo/` are not in git,
- **NEXT**: the open items in order, and any REMIND TAL item.
Keep it lean: state what is true now; if this session overturned an earlier claim, correct that claim where it is written (say what refuted it).
If the top of the file grew very long, tell Tal that a consolidation pass may be due -- do not trim on your own.

## 3. Obsidian (only if there are results, figures, a meeting or a decision worth keeping)
Style rule in the notes: Hebrew, SHORT bullets, figures embedded (`![[file.png]]`), sections Goal / per test / Conclusions / Next.
- **Mac:** write straight into `~/Documents/DISC` (experiments under `Experiments/`, images in `Experiments/attachments/` with an
  `lvis<id>_<topic>_` prefix, a link in `DISC.md`). If it continues an existing experiment note, update that note.
- **Pod:** no vault access. Write the note to `workspace_meta/obsidian_inbox/<date> <topic>.md` and COPY its figures next to it
  (a pod path alone is useless: the Mac cannot open `demo/` or `data/`). The Mac's `/start-work` moves it into the vault.
Show Tal the figures (SendUserFile, display render) rather than only naming paths.

## 4. Pod only: data for the Mac
List the outputs of this session that Tal may want on the Mac (figures, eval jsons, part maps, small run results -- not checkpoints
unless asked). Write them into the notes entry as ONE line, repo-relative paths separated by spaces:
`SYNC TO MAC: demo/shic/... data/densepose_lvis/...` -- the Mac's `/start-work` reads that line and copies them while the pod is still up.
Tell Tal to run `/start-work` on the Mac BEFORE closing the pod (TAU VPN off).

## 5. Git
`git add` only code, scripts, notes and the Obsidian inbox (never data, checkpoints or secrets); `git status -s` to check; commit with a
message that says what the session did, ending with the attribution line from the system reminder; `git push`. On a pod, push needs
`source /workspace/activate_bsb.sh` first. If push fails, say so and why -- do not force anything.

## 6. Hand-off
Tell Tal in a few lines: what was written where, the commit hash, what is running, and the line for the next session:
`/start-work` (on a NEW pod the skill exists only after the first pull: "cd /workspace/disc && git pull, then /start-work").
