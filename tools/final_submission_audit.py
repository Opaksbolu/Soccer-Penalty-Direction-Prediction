"""Read-only final project audit. No training, deletion, or Git modifications."""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess
from pathlib import Path

EXPECTED = {"1.0_majority": (.5546218487394958,.3567567567567568),
"1.0_pose_logistic": (.3697478991596639,.36688657161098104),
"1.0_gru": (.453781512605042,.44611528822055135),
"1.0_single_frame": (.4789915966386555,.46224489795918366),
"0.5_majority": (.5546218487394958,.3567567567567568),
"0.5_pose_logistic": (.5714285714285714,.5713074803983895),
"0.5_gru": (.5042016806722689,.5029380530973452),
"0.5_single_frame": (.5378151260504201,.53306699008347),
"0.2_majority": (.5546218487394958,.3567567567567568),
"0.2_pose_logistic": (.6386554621848739,.6335839598997494),
"0.2_gru": (.5378151260504201,.5146459028550241),
"0.2_single_frame": (.5126050420168067,.49985507246376815)}

def audit(root: Path):
    problems=[]; warnings=[]
    pub=root/'docs/final_results'; src=root/'results/metrics/professor_comparison_v2'
    path=pub/'summary.json'
    if not path.is_file(): problems.append(f'Missing published summary: {path}')
    else:
        report=json.loads(path.read_text())
        if report.get('complete_requested_run') is not True: problems.append('Experiment completion flag not true')
        if report.get('scientific_pretrained_weights') is not True: problems.append('Pretrained weights flag not true')
        summaries=report.get('summary',{})
        for name,(acc,f1) in EXPECTED.items():
            row=summaries.get(name)
            if row is None: problems.append(f'Missing result {name}');continue
            if row.get('folds')!=5 or row.get('n_test_predictions')!=119 or row.get('complete_five_folds') is not True:
                problems.append(f'Incomplete five-fold evaluation: {name}')
            if abs(row['accuracy']-acc)>1e-10 or abs(row['macro_f1']-f1)>1e-10:
                problems.append(f'Metric mismatch: {name}')
            cm=row.get('confusion_matrix',[])
            if len(cm)!=2 or any(len(r)!=2 for r in cm) or sum(map(sum,cm))!=119:
                problems.append(f'Confusion matrix mismatch: {name}')
        print('Published result rows checked:',len(EXPECTED))
    for name in ('comparison.csv','folds.json','run_config.json','macro_f1_comparison.png','FINAL_RESULTS.md','SHA256SUMS.txt'):
        if not (pub/name).is_file(): problems.append(f'Missing publication file: {name}')
    checks=pub/'SHA256SUMS.txt'
    if checks.exists():
        for line in checks.read_text().splitlines():
            parts=line.split(maxsplit=1)
            if len(parts)!=2: problems.append(f'Invalid checksum line: {line}');continue
            expected, filename=parts; file=pub/filename.lstrip('*')
            if not file.is_file(): problems.append(f'Checksum file missing: {file.name}');continue
            actual=hashlib.sha256(file.read_bytes()).hexdigest()
            if actual!=expected: problems.append(f'Checksum mismatch: {file.name}')
    if src.exists():
        for name in ('summary.json','fold_results.json','folds.json','run_config.json'):
            if not (src/name).is_file(): warnings.append(f'Local source missing {name}')
    else: warnings.append('Local 283 MB experiment directory absent (normal for a GitHub ZIP)')
    demo=root/'results/demo'
    if not demo.exists() or not list(demo.rglob('*.mp4')): warnings.append('One-minute MP4 demo not found in this working tree')
    if (root/'src/training/train.py.backup').exists(): warnings.append('Redundant train.py.backup exists; consider archiving it')
    try:
        status=subprocess.run(['git','status','--short'],cwd=root,capture_output=True,text=True,check=True)
        if status.stdout.strip(): warnings.append('Git working tree has uncommitted/untracked files (inspect before committing)')
    except (FileNotFoundError,subprocess.CalledProcessError): warnings.append('Git metadata unavailable (normal for ZIP downloads)')
    for item in warnings: print('WARNING:',item)
    for item in problems: print('FAIL:',item)
    print('AUDIT:', 'PASS' if not problems else 'FAIL',f'({len(problems)} errors, {len(warnings)} warnings)')
    return 1 if problems else 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args()
    raise SystemExit(audit(args.root.resolve()))
