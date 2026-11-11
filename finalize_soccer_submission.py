#!/usr/bin/env python3
"""Safe final verification, packaging and optional archival for the soccer project.

Default: read-only validation followed by creation of a NEW ZIP outside the repo.
No retraining, no Git writes, no deletions, no overwrites.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

REQUIRED = (
    'docs/final_report/Soccer_Penalty_Final_Report.docx',
    'docs/final_report/Soccer_Penalty_Final_Report.pdf',
    'docs/presentation/Soccer_Penalty_Final_Presentation.pptx',
    'results/demo/soccer_penalty_final_demo.mp4',
    'docs/final_results/comparison.csv',
    'docs/final_results/summary.json',
)
# Supplementary artifacts to include when present. Never include raw datasets or model weights.
OPTIONAL = (
    'results/figures/final_comparison/accuracy_comparison.png',
    'results/figures/final_comparison/macro_f1_comparison.png',
    'results/figures/final_comparison/confusion_matrices_0.2s.png',
    'results/metrics/final_comparison/model_comparison.csv',
    'results/metrics/final_comparison/model_comparison.json',
)
# These are safe *candidates* for archival only if they are UNTRACKED.
ARCHIVE_CANDIDATES = (
    'results/demo/audio',
    'results/demo/create_design_preview.py',
    'results/demo/create_final_demo.py',
    'results/demo/create_final_demo_original.py',
    'results/demo/first_prediction_step65.json',
    'results/demo/penalty_16-5_precontact_0.2s.mp4',
    'results/demo/penalty_2-2_unseen_0.2s.mp4',
    'results/demo/professional_design_preview.png',
    'results/demo/unseen_prediction_step69.json',
    'results/backups/cleanup_20261111_151835',
)
SKIP_PARTS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache', '.DS_Store'}
SKIP_SUFFIXES = ('.pyc', '.pyo', '.backup', '.ckpt', '.pth', '.pt', '.h5', '.aiff')


def run(cmd: list[str], root: Path, *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=root, text=True, capture_output=True, timeout=timeout, check=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_file(root: Path, relative: str) -> bool:
    p = root / relative
    return p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve())


def archive_clutter(root: Path, destination: Path, tracked: set[str]) -> None:
    if destination.exists():
        raise RuntimeError(f'Archive destination already exists: {destination}')
    if destination.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Archive destination must be outside the repository')
    selected: list[tuple[Path, str]] = []
    for relative in ARCHIVE_CANDIDATES:
        target = root / relative
        if not target.exists():
            continue
        if target.is_symlink():
            print(f'SKIP symlink: {relative}')
            continue
        candidates = sorted(target.rglob('*')) if target.is_dir() else [target]
        for p in candidates:
            if not p.is_file() or p.is_symlink():
                continue
            rel = p.relative_to(root).as_posix()
            if rel in tracked:
                print(f'SKIP tracked file: {rel}')
                continue
            selected.append((p, rel))
    if not selected:
        print('Nothing eligible to archive; no files changed.')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Copy first, verify byte-for-byte, then move ORIGINAL files to a separate
    # archive folder. Never delete them; never touch tracked files.
    copied = []
    try:
        for source, relative in selected:
            dest = destination / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            if sha256(source) != sha256(dest):
                raise RuntimeError(f'Checksum mismatch while archiving {relative}')
            copied.append((source, dest))
    except Exception:
        print('Archive copy failed. Originals were not changed.')
        raise
    print(f'Verified archival copies of {len(copied)} untracked files.')
    for source, _ in copied:
        source.unlink()
    # Only remove empty directories in the candidate paths.
    for relative in ARCHIVE_CANDIDATES:
        target = root / relative
        if target.is_dir():
            for folder in sorted((p for p in target.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                try:
                    folder.rmdir()
                except OSError:
                    pass
            try:
                target.rmdir()
            except OSError:
                pass
    print(f'Archived untracked extras to: {destination}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, default=Path.cwd())
    ap.add_argument('--output', type=Path, help='New ZIP path; defaults to Desktop with a timestamp')
    ap.add_argument('--archive-clutter', action='store_true', help='Opt-in: move selected untracked extras to a verified archive OUTSIDE the repo')
    ap.add_argument('--archive-to', type=Path, help='Destination for opt-in archival')
    ap.add_argument('--skip-tests', action='store_true')
    args = ap.parse_args()
    root = args.project.expanduser().resolve()
    if not (root / '.git').exists() or not (root / 'src').is_dir():
        print(f'ERROR: Not the expected project root: {root}', file=sys.stderr)
        return 2
    print(f'PROJECT: {root}')
    status = run(['git', 'status', '--short'], root)
    if status.returncode:
        print(f'ERROR: git status failed: {status.stderr}', file=sys.stderr)
        return 2
    print('GIT STATUS:\n' + (status.stdout.strip() or '(clean)'))
    head = run(['git', 'rev-parse', 'HEAD'], root)
    remote = run(['git', 'rev-parse', 'origin/main'], root)
    if head.returncode or remote.returncode or head.stdout.strip() != remote.stdout.strip():
        print('WARNING: Local HEAD and origin/main do not match, or origin/main is unavailable.')
    else:
        print(f'GIT: local HEAD matches origin/main ({head.stdout.strip()[:12]})')
    tracked_result = run(['git', 'ls-files', '-z'], root)
    if tracked_result.returncode:
        print('ERROR: cannot list tracked files', file=sys.stderr)
        return 2
    tracked = {p for p in tracked_result.stdout.split('\0') if p}

    missing = [rel for rel in REQUIRED if not safe_file(root, rel) or (root / rel).stat().st_size == 0]
    if missing:
        print('ERROR: Required files missing/empty/unsafe:')
        for rel in missing:
            print('  -', rel)
        print('No ZIP created. Restore the required files first.')
        return 2
    print(f'REQUIRED FILES: PASS ({len(REQUIRED)} found)')

    # Verify the DOCX and PPTX are valid Office Open XML archives.
    for rel in REQUIRED:
        if rel.endswith(('.docx', '.pptx')):
            try:
                with zipfile.ZipFile(root / rel) as zf:
                    if zf.testzip() is not None:
                        raise ValueError('CRC validation failed')
                    if '[Content_Types].xml' not in zf.namelist():
                        raise ValueError('Missing OOXML content types')
            except (zipfile.BadZipFile, ValueError) as exc:
                print(f'ERROR: Office document invalid: {rel}: {exc}')
                return 2
    pdf = root / REQUIRED[1]
    if not pdf.open('rb').read(5) == b'%PDF-':
        print('ERROR: Final report PDF header invalid')
        return 2
    print('DOCUMENT CONTAINER CHECKS: PASS (DOCX, PPTX, PDF headers)')

    # Run the project's own checks, without modifying source or results.
    checks = []
    if not args.skip_tests:
        checks.append(('pytest', [sys.executable, '-m', 'pytest', '-q']))
    if (root / 'tools/final_submission_audit.py').is_file():
        checks.append(('final audit', [sys.executable, 'tools/final_submission_audit.py']))
    for label, command in checks:
        print(f'RUNNING: {label}')
        try:
            result = run(command, root, timeout=600)
        except subprocess.TimeoutExpired:
            print(f'ERROR: {label} timed out')
            return 2
        print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        if result.returncode:
            print(f'ERROR: {label} failed; no ZIP created')
            return 2
    print('PROJECT CHECKS: PASS')

    if args.archive_clutter:
        destination = (args.archive_to or root.parent / ('Soccer_Penalty_Local_Extras_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).expanduser().resolve()
        archive_clutter(root, destination, tracked)
    elif args.archive_to:
        print('ERROR: --archive-to requires --archive-clutter')
        return 2

    # Preserve tracked project code/config/docs/results, excluding temporary and
    # backup material. Explicitly include final deliverables even if Git-ignored.
    candidates = tracked | set(REQUIRED) | {rel for rel in OPTIONAL if safe_file(root, rel)}
    included = []
    for rel in sorted(candidates):
        if any(part in SKIP_PARTS for part in Path(rel).parts):
            continue
        if rel.endswith(SKIP_SUFFIXES) or rel.startswith(('results/backups/', 'data/raw/', 'data/processed/', 'results/models/', 'results/checkpoints/')):
            continue
        if safe_file(root, rel):
            included.append(rel)
    # Additional tracked demo assets are omitted except the final video.
    included = [r for r in included if not r.startswith('results/demo/') or r == REQUIRED[3]]
    if not set(REQUIRED).issubset(included):
        print('ERROR: Final deliverables would be excluded from ZIP')
        return 2

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = (args.output or (root.parent / f'Soccer_Penalty_FINAL_VERIFIED_{timestamp}.zip')).expanduser().resolve()
    if output.exists():
        print(f'ERROR: ZIP already exists; will not overwrite: {output}')
        return 2
    if output.is_relative_to(root):
        print('ERROR: Output ZIP must be outside repository')
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental overwrite.
    try:
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
            for rel in included:
                zf.write(root / rel, arcname=f'{root.name}/{rel}')
            manifest = {
                'created_at_local': datetime.now().isoformat(timespec='seconds'),
                'git_head': head.stdout.strip(),
                'file_count': len(included),
                'required_files': list(REQUIRED),
                'files_sha256': {rel: sha256(root / rel) for rel in included},
            }
            zf.writestr(f'{root.name}/SUBMISSION_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        with zipfile.ZipFile(output) as zf:
            bad = zf.testzip()
            names = set(zf.namelist())
            absent = [rel for rel in REQUIRED if f'{root.name}/{rel}' not in names]
            if bad or absent:
                raise RuntimeError(f'ZIP verification failed: bad={bad}, absent={absent}')
    except Exception as exc:
        print(f'ERROR: Packaging failed: {exc}')
        print(f'Inspect incomplete output if present: {output}')
        return 2
    print(f'ZIP: PASS ({len(included)} project files + manifest)')
    print(f'ZIP PATH: {output}')
    print(f'ZIP SHA256: {sha256(output)}')
    print('FINISHED: No training, tracked-file edits, commits, pushes, or deletions performed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
