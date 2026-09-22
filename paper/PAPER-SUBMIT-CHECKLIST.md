# Paper submit checklist

Paper: `paper/ARC-PAPER-WRITEUP-DRAFT.md` for the ARC Prize 2026 Paper Track. Deadline 9 Nov.
Submission it describes: ref 56470466, `oscarmorke/duck-control`.

Every box below is Oscar's click. No machine made anything public, pushed anything, or submitted anything.

## 1. Gate: the numbers

Run from the repo root on branch `paper/arc-writeup`:

```
DUCK_SRC=<unzipped taaf-kaggle-source-share>/src python paper/verify_numbers.py --tests --kaggle
python -m pytest paper/test_verify_numbers.py
```

- [ ] `verify_numbers.py` exits 0. It recomputes every number in the body from `runs/`, recounts the words, scans for placeholders, and checks that every cited repo path exists at the cited commit.
- [ ] Words: at most 1,450 by the stated method (last run: 1,446). Kaggle's limit is 1,500.
- [ ] Placeholders: none (TODO, TBD, XXX, FIXME, ??, `<FILL>`, `[TK]`, `{{x}}`). The script checks this. Also read the page once by eye.
- [ ] If Kaggle shows a public score, update section 1 ("PENDING, with no public score"). The `--kaggle` check prints WARN when the status changes.
- [ ] After any change to `runs/`, rerun `python paper/ARC-PAPER-FIG1-AB.py` to redraw Figure 1.

## 2. Private objects the paper cites

A reader must be able to open each one. Today none of them is public.

| | Object | Where it is now | What the paper needs | Oscar's click |
|---|---|---|---|---|
| a | Repo `arc-agi-3-probe`, commits `9f8b1de` and `0a1bf51` | Local only. `git remote -v` is empty. `gh repo view Morkeeth/arc-agi-3-probe` finds no repo. | A public GitHub repo with branch `paper/arc-writeup` pushed, so both commit hashes resolve. Do not rebase or squash: the hashes are cited. | [ ] create repo, push `main` and `paper/arc-writeup`, set public |
| b | Kaggle notebook `oscarmorke/duck-control` v1 (the submission) | `is_private: true` in `kaggle/duck-control/kernel-metadata.json`. Version 1 is not verified by any check. | Public, so a reader can see the submitted notebook. | [ ] confirm the version, then set public on Kaggle |
| c | Kaggle notebook `oscarmorke/duck-deadprobe` v2 (the guard arm) | `is_private: true` in `kaggle/duck-deadprobe/kernel-metadata.json` | Version 2 is not verified by any check. Public, as the guard arm's run. The paper does not name it yet; add it to Sources when public. | [ ] set public, add to Sources |
| d | Kaggle dataset `oscarmorke/arc-agi-3-probe` (the guard code the notebook loads) | Private (`kaggle datasets list --mine` lists it, last updated 2026-09-22 08:49) | Public, or the guard notebook cannot run for anyone else. | [ ] set public |
| e | `ARC-PREP-RETURN.md` (2026-09-22) and `ARC-SCOPE-RETURN.md` (2026-09-21) | `~/.local/state/day-run/`. They hold scratch paths and fleet notes. | **Dropped from the paper on this branch.** The 7,920 s cap now cites `log.txt`. The decision rule is stated in section 5. To cite them again, publish a cleaned copy in the repo after review. | [ ] keep dropped, or review and publish a cleaned copy |
| f | `runs/duck-*/summary.txt` and `runs/duck-*/taaf_setup_env.json` | Gitignored under `runs/*`. Force-added on this branch in commit `0a1bf51`. | Nothing more, once (a) is done. Checked: no keys or local paths inside. | [ ] covered by (a) |
| g | Figure 1 PNG and its script | `paper/ARC-PAPER-FIG1-AB.png`, `paper/ARC-PAPER-FIG1-AB.py`. The script now reads the run files. | Upload the PNG as the writeup's image. The script is public with (a). | [ ] upload PNG in section 5, step 4 |

## 3. Claims in the draft that rest on something private

- [ ] **"Fixed before either GPU run finished".** The only timed record of the rule is a private Claude Code session log: the text "Decision rule, fixed now before the numbers exist" first appears at 2026-09-22T14:46:40Z. The runs ended 16:50:28 (guard) and 17:13:08 (control) by `summary.txt`, on the Kaggle clock, which we take to be UTC. `ARC-PREP-RETURN.md` cannot prove the order: its last write is 19:14 local, after both runs. The draft now says "not registered publicly" and no longer says "pre-registered". Oscar: accept that wording, or change it.
- [ ] **"34 pass".** Confirmed 2026-09-22 on a fresh `kaggle datasets download jeroencottaar/taaf-kaggle-source-share` (zip sha256 `e9d0b722...a94c`): `34 passed in 3.64s` with `DUCK_SRC`, `27 passed, 7 skipped` without it.
- [ ] **"17:22 UTC".** Confirmed with the read-only `kaggle competitions submissions arc-prize-2026-arc-agi-3 -v`: `56470466, 2026-09-22 17:22:56.260000, SubmissionStatus.PENDING`. The CLI prints no zone. The Kaggle API reports UTC.
- [ ] **"two games flip a counter cell on every click".** From the `deadprobe/memo.py` docstring (vc33, su15). Not recomputed by the script.

## 4. What changed from the 19:31 draft

The diff from commit `26d9796` shows every edit. In short:

1. The GPU citation pointed at `taaf_setup_env.json`, which has no GPU name. It now points at the GPU check line in `log.txt`.
2. Section 7 said each refusal triggers a new model call. Duck's model acts through a Python script, and a refused action returns an error to that script while the rest of the script runs on. A new model call follows only when nothing else ran. The logs do not count those cases, so the draft now says so.
3. "Pre-registered" became "fixed in advance", see section 3 above.
4. New clause: 10 known repeats still ran (memo `repeat_noop`: tu93 8, cd82 2).
5. Tokens per second now match `summary.txt` exactly (101.53 and 109.45, the draft had 101.5 and 109.5).
6. Sources no longer claim the run files sit in `9f8b1de`.
7. Word count now has one stated method, enforced by the script. The 19:31 draft body was 1,500 by raw `wc -w` (tables and pipes count as words) and 1,423 by the stated method. The fixes above added words, then cuts brought the body to 1,446.

## 5. Manual steps, in order

1. [ ] Do section 2 (a) to (d). Push first, then flip visibility.
2. [ ] Add `oscarmorke/duck-deadprobe` and the public repo URL to Sources.
3. [ ] Rerun the section 1 gate. It must exit 0.
4. [ ] Open the Kaggle writeup for the Paper Track. Paste from the title line to the end of Sources. Upload `paper/ARC-PAPER-FIG1-AB.png`.
5. [ ] Read the rendered page once. Check the table renders.
6. [ ] Submit. That is Oscar's click, not a machine's.
