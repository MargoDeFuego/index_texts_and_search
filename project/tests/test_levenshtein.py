import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lab4.levenshtein import levenshtein

def run():
    cases=[('', '',0),('', 'а',1),('кот','кот',0),('кот','кит',1),('кот','коты',1),('коты','кот',1),('дорго','дорог',2),('Яднекс'.lower(),'Яндекс'.lower(),2)]
    for a,b,expected in cases:
        got=levenshtein(a,b); assert got==expected,(a,b,got,expected)
    print('test_levenshtein: OK')
if __name__=='__main__': run()
