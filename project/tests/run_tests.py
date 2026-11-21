import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for name in ['test_levenshtein.py','test_data.py']:
    subprocess.check_call([sys.executable,str(ROOT/'tests'/name)])
print('ALL TESTS: OK')
