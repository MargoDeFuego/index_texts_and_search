import pytest
from indexing_exam.task03 import PositionalIndex, QuerySyntaxError, parse_query

DOCS = [
    {"doc_id": "D1", "title": "", "text": "музей анализ 1917"},
    {"doc_id": "D2", "title": "", "text": "музей обзор"},
    {"doc_id": "D3", "title": "", "text": "археология музей анализ"},
]


def test_boolean_phrase_near():
    idx = PositionalIndex.build(DOCS)
    assert idx.search("музей AND NOT обзор") == ["D1", "D3"]
    assert idx.search('"музей анализ"') == ["D1", "D3"]
    assert idx.search("музей NEAR/1 анализ") == ["D1", "D3"]


def test_precedence():
    idx = PositionalIndex.build(DOCS)
    assert idx.search("археология OR музей AND обзор") == ["D2", "D3"]


def test_update_delete():
    idx = PositionalIndex.build(DOCS)
    idx.add_or_update({"doc_id": "D2", "title": "", "text": "архитектура 1480"})
    assert idx.search("обзор") == []
    assert idx.search("1480") == ["D2"]
    idx.delete("D1")
    assert "D1" not in idx.universe


def test_bad_syntax():
    with pytest.raises(QuerySyntaxError):
        parse_query("(музей AND")
