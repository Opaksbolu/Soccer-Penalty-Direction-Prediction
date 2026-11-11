# Final submission: safe and reproducible workflow

This is an **additive completion kit**, based on the uploaded project ZIP. It does **not** replace your newer local repository, delete data, or rerun the expensive 30 neural-model fold/horizon training jobs.

## 1. Preserve your current local project

```bash
cd ~/Desktop/Soccer-Penalty-Direction-Prediction
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate soccer-penalty
git status -sb
git log -1 --oneline
ls -lh docs/final_results/summary.json
ls -lh results/metrics/professor_comparison_v2/fold_results.json
```

Expected latest commit from prior verification: `e13c942`. Your untracked `results/demo/` must be preserved.

## 2. Install only the additive kit files

Extract `Soccer_Penalty_Completion_Kit.zip` into Downloads. **Do not replace your repo with the ZIP.** Copy only the new tools and guide:

```bash
cd ~/Desktop/Soccer-Penalty-Direction-Prediction
mkdir -p tools
cp ~/Downloads/Soccer_Penalty_Completion_Kit/tools/final_submission_audit.py tools/
cp ~/Downloads/Soccer_Penalty_Completion_Kit/tools/safe_cleanup.py tools/
cp ~/Downloads/Soccer_Penalty_Completion_Kit/FINAL_SUBMISSION_STEPS.md .
```

If your extracted folder is different, locate it with `find ~/Downloads -name final_submission_audit.py` and adjust the copy path.

## 3. Validate, without retraining

```bash
python -m compileall -q src tools
python -m pytest -q
python tools/final_submission_audit.py
cd docs/final_results && shasum -a 256 -c SHA256SUMS.txt && cd ../..
```

`final_submission_audit.py` verifies the twelve exact published metric rows, complete five-fold flags, 119 predictions per row, confusion-matrix totals, and published file checksums. A missing demo in the GitHub ZIP is only a warning. It does not independently validate the original videos or prove real-world generalization.

## 4. Optional safe cleanup

```bash
python tools/safe_cleanup.py
```

Only if the preview identifies `src/training/train.py.backup` and you want to archive it:

```bash
python tools/safe_cleanup.py --apply
```

This moves the redundant backup into `results/backups/cleanup_<timestamp>/`; it does **not** delete the file. No other files are cleanup candidates. Never run `git clean -fdx`, `rm -rf results`, or `git add -A` for this project.

## 5. Review demo and submission materials

- Open the local MP4 in `results/demo/`; ensure approximately 1 minute and legible predictions.
- Show the **binary left/right** output and disclose that center classification was not validated.
- Include the demo in the presentation or provide it separately, as required by the instructor.
- Use the final Word/PDF report, verified comparison plot, and code repository.
- Cite the original dataset source; document the 119 eligible kicks, 66 left and 53 right.
- The 0.2-second pose-logistic result (63.87% accuracy) is best overall; the GRU is not consistently better than single-frame.

## 6. Commit only the new audit utilities after review

```bash
git status -sb
git add tools/final_submission_audit.py tools/safe_cleanup.py FINAL_SUBMISSION_STEPS.md
git diff --cached --check
git diff --cached --stat
git commit -m "Add final submission audit and safe cleanup utilities"
git push origin main
```

Do **not** commit the large checkpoints, raw videos, private backups, or untracked demo unless your instructor requests them and you review their size/licensing.

## Reproducing experiments (optional; costly)

```bash
python -m src.compare_single_frame --device mps --epochs 15 --batch-size 4 --seed 42 --architecture both --output results/metrics/a_new_reproduction_run
```

This requires the original downloaded dataset and prepared clips. It is **not needed** to finalize the existing validated experiment. Do not use the already populated `professor_comparison_v2` directory for a fresh run.
