import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run():
    dev=json.loads((ROOT/'data'/'queries_dev.json').read_text(encoding='utf-8'))
    test=json.loads((ROOT/'data'/'queries_test.json').read_text(encoding='utf-8'))
    assert len(dev)==20 and len(test)==20
    assert not ({x['id'] for x in dev}&{x['id'] for x in test})
    for data in (dev,test):
        counts={c:sum(x['category']==c for x in data) for c in ['clean','common_typo','entity_typo','rare_correct']}
        assert counts=={'clean':5,'common_typo':5,'entity_typo':5,'rare_correct':5}
        for r in data:
            for e in r['original_entities']:
                assert r['original_text'][e['start']:e['end']]==e['text']
            for e in r['correct_entities']:
                assert r['correct_text'][e['start']:e['end']]==e['text']
            for t in r['typos']:
                assert r['original_text'][t['start']:t['end']]==t['wrong']
    print('test_data: OK')
if __name__=='__main__': run()
