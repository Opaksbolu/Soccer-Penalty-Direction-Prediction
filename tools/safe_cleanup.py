"""Preview or archive known redundant backups; never delete any file."""
from __future__ import annotations
import argparse,shutil,datetime
from pathlib import Path

CANDIDATES=('src/training/train.py.backup',)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='Move known backup to a timestamped archive, never delete')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    root=args.root.resolve()
    for rel in CANDIDATES:
        src=root/rel
        if not src.is_file(): print('SKIP absent:',rel);continue
        print('FOUND redundant backup:',rel, f'({src.stat().st_size} bytes)')
        if not args.apply: print('PREVIEW ONLY: would archive; no changes made');continue
        archive=root/'results/backups'/('cleanup_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))/rel
        archive.parent.mkdir(parents=True,exist_ok=True)
        if archive.exists(): raise FileExistsError(archive)
        shutil.move(str(src),str(archive));print('ARCHIVED TO:',archive)
    print('No dataset, checkpoint, experiment result, demo, or tracked source was deleted.')

if __name__=='__main__': main()
