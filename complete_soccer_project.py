#!/usr/bin/env python3
"""Read-only final audit and optional submission ZIP builder for the existing project.

This does NOT retrain, delete, reset Git, commit, push, or overwrite source/results.
Run from the root of Soccer-Penalty-Direction-Prediction.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'docs' / 'final_results'
SOURCE = ROOT / 'results' / 'metrics' / 'professor_comparison_v2'
EXPECTED = ('majority', 'pose_logistic', 'gru', 'single_frame')
HORIZONS = ('1.0', '0.5', '0.2')


def command(*args):
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_results(errors, warnings):
    summary_path = RESULTS / 'summary.json'
    csv_path = RESULTS / 'comparison.csv'
    if not summary_path.is_file() or not csv_path.is_file():
        errors.append('Published summary.json or comparison.csv is missing.')
        return None
    try:
        data = json.loads(summary_path.read_text(encoding='utf-8'))
        summaries = data['summary']
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(f'Cannot parse summary.json: {exc}')
        return None
    if data.get('complete_requested_run') is not True:
        errors.append('Experiment does not report complete_requested_run=true.')
    if data.get('scientific_pretrained_weights') is not True:
        errors.append('Pretrained weights are not confirmed by the summary.')
    if len(summaries) != 12:
        errors.append(f'Expected 12 model/horizon summaries, found {len(summaries)}.')
    rows = []
    for horizon in HORIZONS:
        for model in EXPECTED:
            key = f'{horizon}_{model}'
            item = summaries.get(key)
            if not isinstance(item, dict):
                errors.append(f'Missing result: {key}')
                continue
            if item.get('folds') != 5 or item.get('n_test_predictions') != 119 or item.get('complete_five_folds') is not True:
                errors.append(f'{key}: incomplete five-fold evaluation.')
            matrix = item.get('confusion_matrix')
            if not isinstance(matrix, list) or len(matrix) != 2 or any(not isinstance(r, list) or len(r) != 2 for r in matrix):
                errors.append(f'{key}: invalid confusion matrix.')
                continue
            try:
                a, b = matrix[0]
                c, d = matrix[1]
                if not all(type(v) is int and v >= 0 for v in (a,b,c,d)):
                    raise ValueError('counts must be nonnegative integers')
                if a+b != 66 or c+d != 53:
                    errors.append(f'{key}: class support does not match 66 left / 53 right.')
                acc = (a+d)/119
                if not math.isclose(acc, float(item['accuracy']), abs_tol=1e-10):
                    errors.append(f'{key}: accuracy disagrees with confusion matrix.')
                def f1(tp, fp, fn):
                    denominator = 2*tp + fp + fn
                    return 2*tp/denominator if denominator else 0.0
                macro = (f1(a,c,b) + f1(d,b,c))/2
                if not math.isclose(macro, float(item['macro_f1']), abs_tol=1e-10):
                    errors.append(f'{key}: macro-F1 disagrees with confusion matrix.')
                rows.append((horizon, model, acc, macro, item.get('mean_seconds_per_kick_including_decode')))
            except (TypeError, ValueError, KeyError) as exc:
                errors.append(f'{key}: invalid numerical fields: {exc}')
    try:
        with csv_path.open(newline='', encoding='utf-8') as f:
            csv_rows = list(csv.DictReader(f))
        if len(csv_rows) != 12:
            errors.append(f'comparison.csv should have 12 rows, found {len(csv_rows)}.')
        for row in csv_rows:
            key = f"{float(row['horizon_seconds']):.1f}_{row['model']}"
            item = summaries.get(key)
            if item is None or not math.isclose(float(row['accuracy']), float(item['accuracy']), abs_tol=1e-10) or not math.isclose(float(row['macro_f1']), float(item['macro_f1']), abs_tol=1e-10):
                errors.append(f'comparison.csv mismatch for {key}.')
    except (OSError, KeyError, ValueError) as exc:
        errors.append(f'Cannot validate comparison.csv: {exc}')
    manifest = RESULTS / 'SHA256SUMS.txt'
    if manifest.is_file():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                errors.append(f'Invalid SHA256 manifest line: {line}')
                continue
            expected_hash, filename = parts
            filename = filename.lstrip('*').strip()
            target = RESULTS / filename
            if not target.is_file() or sha256(target) != expected_hash:
                errors.append(f'SHA256 mismatch/missing: {filename}')
    else:
        warnings.append('SHA256SUMS.txt not found.')
    return rows


def git_check(warnings):
    r = command('git', 'status', '--short')
    if r.returncode:
        warnings.append('Git status unavailable; check repository manually.')
        return
    if r.stdout.strip():
        warnings.append('Git working tree has changes/untracked files (may be intentional).')
        print('Git status:\n' + r.stdout.rstrip())
    else:
        print('Git tracked/untracked status: clean.')
    log = command('git', 'log', '-1', '--format=%h %s')
    if log.returncode == 0:
        print('Latest commit:', log.stdout.strip())


def run_tests(errors):
    print('\nRunning existing project tests ...')
    r = command(sys.executable, '-m', 'pytest', '-q')
    print((r.stdout + r.stderr).strip()[-3500:])
    if r.returncode:
        errors.append('pytest did not pass.')


def cleanup_preview():
    print('\nCLEANUP PREVIEW (NO DELETION):')
    for relative in ('src/training/train.py.backup', 'results/backups', 'results/demo'):
        p = ROOT / relative
        print(f'  {relative}: {"present" if p.exists() else "absent"}')
    print('  Preserve results/demo and all trained checkpoints / datasets.')
    print('  A tracked backup file should not be removed automatically.')
    print('  For actual cleanup, inspect each file and make a separate manual Git commit.')


def create_zip(output, include_demo, warnings):
    # Zip only committed files to avoid accidentally packaging raw datasets,
    # credentials, temporary files, or large checkpoints.
    tracked = command('git', 'ls-files', '-z')
    if tracked.returncode:
        raise RuntimeError('git ls-files failed; cannot safely build ZIP.')
    paths = [Path(p.decode('utf-8')) for p in tracked.stdout.encode('utf-8').split(b'\0') if p]
    paths = [p for p in paths if (ROOT / p).is_file()]
    excluded = ('.env', '.env.', 'data/raw/', 'data/processed/', 'data/annotations/', 'results/models/')
    paths = [p for p in paths if not (p.as_posix() == '.env' or p.as_posix().startswith(('.env.', 'data/raw/', 'data/processed/', 'data/annotations/', 'results/models/')))]
    if include_demo:
        demo_dir = ROOT / 'results' / 'demo'
        if demo_dir.is_dir():
            for p in demo_dir.rglob('*'):
                if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.m4v'):
                    paths.append(p.relative_to(ROOT))
        else:
            warnings.append('Demo folder missing; ZIP will not contain demo.')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing ZIP: {output}')
    if output.resolve().is_relative_to(ROOT) and output.name in {p.name for p in paths}:
        raise ValueError('Output ZIP conflicts with a source file.')
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in sorted(set(paths)):
            z.write(ROOT / rel, arcname=f'Soccer-Penalty-Direction-Prediction/{rel.as_posix()}')
    print(f'Created: {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB, {len(set(paths))} files)')
    if not include_demo:
        print('Demo excluded. Submit the ~1-minute demo separately or rerun with --include-demo.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true', help='Run existing pytest suite')
    parser.add_argument('--package', action='store_true', help='Create ZIP of tracked repository files')
    parser.add_argument('--include-demo', action='store_true', help='Add local MP4/MOV demo files to ZIP')
    parser.add_argument('--output', type=Path, help='ZIP output path (default: Desktop with timestamp-free filename)')
    args = parser.parse_args()
    if not (ROOT / '.git').exists():
        parser.error('Place this script at the root of the existing Git repository.')
    errors, warnings = [], []
    print('Project:', ROOT)
    rows = verify_results(errors, warnings)
    if rows:
        print(f'Validated {len(rows)} model/horizon results against confusion matrices and CSV.')
        print('\nHorizon  Model          Accuracy  Macro-F1')
        for horizon, model, accuracy, macro, _ in rows:
            print(f'{horizon:>5}s   {model:<14} {accuracy:>7.2%}   {macro:>7.2%}')
    git_check(warnings)
    cleanup_preview()
    if args.test:
        run_tests(errors)
    for warning in warnings:
        print('WARNING:', warning)
    for error in errors:
        print('ERROR:', error)
    if errors:
        print(f'FINAL AUDIT: FAIL ({len(errors)} errors); no ZIP created.')
        return 1
    print(f'FINAL AUDIT: PASS (0 errors, {len(warnings)} warnings)')
    if args.package:
        output = args.output or (ROOT.parent / 'Soccer_Penalty_Final_Submission.zip')
        try:
            create_zip(output.expanduser(), args.include_demo, warnings)
        except (OSError, RuntimeError, ValueError) as exc:
            print('PACKAGE FAILED:', exc)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
