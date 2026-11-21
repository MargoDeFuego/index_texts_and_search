from indexing_exam.task02 import MorphIndex
from indexing_exam.common import tokenize

print("Tokens:", tokenize("музей"))

docs = [
    {"doc_id": "D1", "title": "", "text": "музей"},
    {"doc_id": "D2", "title": "", "text": "музея"},
]

for mode in ("surface", "stem", "lemma"):
    idx = MorphIndex.build(docs, mode, False)
    ranked, terms, missing = idx.search("музей")
    print(f"{mode} -> {ranked}; terms={terms}; missing={missing}")
