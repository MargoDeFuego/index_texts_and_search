from indexing_exam.task02 import MorphIndex
from indexing_exam.common import tokenize

docs = [
    {"doc_id": "A", "title": "", "text": "старый замок на холме"},
    {"doc_id": "B", "title": "", "text": "сломался дверной замок"},
]

print("Проверка токенизации:", tokenize("старый замок на холме"))

for mode in ("surface", "stem", "lemma"):
    idx = MorphIndex.build(docs, mode, False)
    print(mode, "->", idx.search("замок")[0])
