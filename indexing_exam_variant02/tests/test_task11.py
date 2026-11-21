from indexing_exam.task11 import levenshtein, gazetteer_ner

GAZ = [
    {"id": "P", "name": "Пушкин", "type": "PERSON", "forms": ["Пушкин"]},
    {"id": "L", "name": "Пушкин", "type": "LOC", "forms": ["Пушкин"]},
]


def test_levenshtein():
    assert levenshtein("музй", "музей") == 1
    assert levenshtein("кот", "скот") == 1
    assert levenshtein("кот", "кит") == 1


def test_homonym_context():
    assert gazetteer_ner("Город Пушкин известен дворцами", GAZ)[0]["type"] == "LOC"
    assert gazetteer_ner("Поэт Пушкин известен стихами", GAZ)[0]["type"] == "PERSON"
