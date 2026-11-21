import pytest
from indexing_exam.task02 import MorphIndex


def has_ru_model():
    try:
        import spacy
        spacy.load("ru_core_news_sm")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not has_ru_model(), reason="ru_core_news_sm is not installed in the build environment")
def test_lemma_matches_forms():
    corpus = [
        {"doc_id": "D1", "title": "", "text": "музей"},
        {"doc_id": "D2", "title": "", "text": "музея"},
    ]
    surface = MorphIndex.build(corpus, "surface", False)
    lemma = MorphIndex.build(corpus, "lemma", False)
    assert surface.search("музей")[0] == ["D1"]
    assert lemma.search("музей")[0] == ["D1", "D2"]
