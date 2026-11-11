#!/usr/bin/env python3
"""Non-destructive final publication of completed penalty prediction experiments."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path('results/metrics/professor_comparison_v2')
FILES = ('summary.json', 'comparison.csv', 'folds.json', 'run_config.json', 'macro_f1_comparison.png')
MODELS = ('majority', 'pose_logistic', 'gru', 'single_frame')
HORIZONS = ('1.0', '0.5', '0.2')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git(root, *args):
    p = subprocess.run(['git', *args], cwd=root, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout.strip()


def validate(root):
    src = root / SOURCE
    missing = [str(src / f) for f in FILES + ('fold_results.json',) if not (src / f).is_file()]
    if missing:
        raise RuntimeError('Missing completed experiment files:\n' + '\n'.join(missing))
    summary = json.loads((src / 'summary.json').read_text())
    if summary.get('complete_requested_run') is not True:
        raise RuntimeError('Experiment not marked complete; refusing publication')
    if summary.get('scientific_pretrained_weights') is not True:
        raise RuntimeError('Pretrained weights not verified; refusing publication')
    for horizon in HORIZONS:
        for model in MODELS:
            key = f'{horizon}_{model}'
            result = summary['summary'].get(key)
            if not result or result.get('folds') != 5 or result.get('complete_five_folds') is not True:
                raise RuntimeError(f'Missing/incomplete five-fold result: {key}')
            if result.get('n_test_predictions') != 119:
                raise RuntimeError(f'Unexpected prediction count in {key}; inspect dataset changes')
            for metric in ('accuracy', 'macro_f1'):
                if not 0 <= result[metric] <= 1:
                    raise RuntimeError(f'Invalid {metric} in {key}')
    with (src / 'comparison.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError('Comparison CSV is empty')
    return summary, rows


def report(summary, commit):
    lines = [
        '# Final experimental results — soccer penalty direction prediction', '',
        '## Scope and provenance', '',
        'Exploratory **binary left/right** prediction from pre-contact video, '
        'with labels defined from the goalkeeper’s perspective. The available dataset '
        'contained only one center kick, so these results **do not validate three-class prediction**.', '',
        '- Dataset: 119 eligible kicks (66 left, 53 right).',
        '- Evaluation: five grouped cross-validation folds, pooled out-of-fold predictions.',
        '- Pre-contact horizons: 1.0, 0.5 and 0.2 seconds.',
        '- Compared: majority, pose logistic regression, MobileNetV3-Small + GRU, '
        'and single-frame MobileNetV3-Small.',
        '- ImageNet-pretrained frozen visual encoder; 15 training epochs per fold for the paired neural comparison.',
        f'- Source-code commit at publication: `{commit}`.',
        '- Original raw videos, processed clips and large model checkpoints are not published in this lightweight results folder.', '',
        '## Pooled five-fold results', '',
        '| Horizon | Model | Accuracy | Macro-F1 |',
        '|---|---|---:|---:|',
    ]
    for horizon in HORIZONS:
        for model in MODELS:
            r = summary['summary'][f'{horizon}_{model}']
            lines.append(f"| {horizon}s | {model.replace('_', ' ')} | {r['accuracy']:.2%} | {r['macro_f1']:.2%} |")
    lines += ['', '## Paired GRU minus single-frame differences', '',
              '| Horizon | Accuracy difference (percentage points) | Macro-F1 difference (percentage points) |',
              '|---|---:|---:|']
    for horizon in HORIZONS:
        d = summary['paired_differences'][horizon]
        lines.append(f"| {horizon}s | {100*d['accuracy_gru_minus_single_frame']:+.2f} | {100*d['macro_f1_gru_minus_single_frame']:+.2f} |")
    lines += ['', '## Interpretation and limitations', '',
              'The pose-logistic baseline performed best at 0.2s (63.87% accuracy, 63.36% macro-F1). '
              'The GRU exceeded the single-frame model only at 0.2s, by approximately 2.52 accuracy '
              'percentage points and 1.48 macro-F1 percentage points; this is not evidence of statistical significance.', '',
              'The GRU and single-frame heads have different capacities, so this is a comparison of complete '
              'model configurations rather than a controlled isolation of temporal information. '
              'The small dataset, uncertain generalization to new players/matches, and lack of a viable '
              'center class limit conclusions. Model timing in the paired experiment includes decoding '
              '16 frames and must not be described as network-only inference latency.', '',
              '## Files', '',
              '`summary.json`, `comparison.csv`, `folds.json`, `run_config.json`, '
              '`macro_f1_comparison.png`, and `SHA256SUMS.txt` are published alongside this report. '
              'The full 283 MB experiment folder, including fold-level predictions and checkpoints, remains local.', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, default=Path.cwd())
    ap.add_argument('--apply', action='store_true', help='Create lightweight tracked results; default is preview only')
    ap.add_argument('--verify-tests', action='store_true', help='Run pytest before any publication')
    args = ap.parse_args()
    root = args.project.expanduser().resolve()
    if not (root / '.git').exists():
        ap.error('Not a Git repository; use --project PATH')
    if args.verify_tests:
        subprocess.run(['python', '-m', 'compileall', '-q', 'src'], cwd=root, check=True)
        subprocess.run(['python', '-m', 'pytest', '-q'], cwd=root, check=True)
    summary, rows = validate(root)
    commit = git(root, 'rev-parse', '--short', 'HEAD')
    dest = root / 'docs/final_results'
    print(f'Validated {len(rows)} comparison CSV rows; 12 complete five-fold summaries.')
    print(f'Source: {root / SOURCE}')
    print(f'Destination: {dest}')
    print(f'Git HEAD: {commit}')
    print('Files:', ', '.join(FILES), 'FINAL_RESULTS.md, SHA256SUMS.txt')
    print('Will NOT modify data/, results/, .git/, checkpoints or demos.')
    if not args.apply:
        print('DRY RUN: no changes made. Use --apply to create final publication files.')
        return
    dest.mkdir(parents=True, exist_ok=True)
    # Never delete or mutate the original experiment. Copy only lightweight summaries.
    for filename in FILES:
        shutil.copy2(root / SOURCE / filename, dest / filename)
    (dest / 'FINAL_RESULTS.md').write_text(report(summary, commit) + '\n')
    checksums = [f'{digest(dest / f)}  {f}' for f in FILES + ('FINAL_RESULTS.md',)]
    (dest / 'SHA256SUMS.txt').write_text('\n'.join(checksums) + '\n')
    print('\nCreated:', dest)
    print('Review with: git status -sb && git diff --no-index /dev/null docs/final_results/FINAL_RESULTS.md')
    print('Then commit explicitly if satisfied. No git commit or push was performed.')
    print('Optional cleanup review: find . -name "*.backup" -o -name "*.bak"')


if __name__ == '__main__':
    main()
