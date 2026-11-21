import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lab4.dictionaries import rebuild_all
if __name__=='__main__':
    w,e=rebuild_all(ROOT)
    print(f'word forms: {len(w)}; entities: {len(e)}')
