from indexing_exam.common import read_jsonl
from indexing_exam.task03 import PositionalIndex

docs = read_jsonl("data/task03/small_corpus.jsonl")
idx = PositionalIndex.build(docs)

print("До обновления, обзор:", idx.search("обзор"))

idx.add_or_update({
    "doc_id": "S3",
    "title": "Музей",
    "text": "Музей истории живописи хранит каталог 1905 года."
})

print("После обновления, обзор:", idx.search("обзор"))
print("После обновления, 1905:", idx.search("1905"))
