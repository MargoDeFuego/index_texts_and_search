import json
from indexing_exam.common import read_jsonl
from indexing_exam.task03 import PositionalIndex, parse_query, ast_to_dict

query = '"культурное наследие" OR (археология AND NOT реставрация)'

print("Дерево запроса:")
print(json.dumps(
    ast_to_dict(parse_query(query)),
    ensure_ascii=False,
    indent=2
))

docs = read_jsonl("data/task03/small_corpus.jsonl")
idx = PositionalIndex.build(docs)

print("\nПозиции терминов:")
print(json.dumps(
    idx.dump_positions(["музей", "обзор", "1917", "пушкин"]),
    ensure_ascii=False,
    indent=2
))
