from collections import Counter
from pathlib import Path
from .tokenizer import tokenize, normalize
from .io_utils import load_json, save_json


def build_word_dictionary(corpus_dir):
    counts = Counter()
    display = {}
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for token in tokenize(text):
            key = normalize(token.text)
            counts[key] += 1
            display.setdefault(key, token.text)
    return dict(sorted(counts.items())), display


def build_entity_dictionary(catalog, dev_records):
    # Справочник строится из независимого каталога сущностей корпуса + dev.
    # Из test ничего не добавляется.
    result = {item["id"]: dict(item) for item in catalog}

    # Допустимо пополнять справочник по dev-настройке.
    for rec in dev_records:
        for ent in rec.get("correct_entities", []):
            norm_text = normalize(ent["text"])
            found = None
            for item in result.values():
                if item["type"] != ent["type"]:
                    continue
                if any(normalize(f) == norm_text for f in item.get("forms", [])):
                    found = item
                    break
            if found is None:
                eid = f"dev_{ent['type'].lower()}_{len(result)+1:03d}"
                result[eid] = {
                    "id": eid,
                    "name": ent["text"],
                    "type": ent["type"],
                    "forms": [ent["text"]],
                    "source": "queries_dev.json"
                }
            elif ent["text"] not in found["forms"]:
                found["forms"].append(ent["text"])
    return sorted(result.values(), key=lambda x: x["id"])


def rebuild_all(root):
    root = Path(root)
    word_dict, display = build_word_dictionary(root / "data" / "corpus")
    catalog = load_json(root / "data" / "entity_catalog.json")
    dev = load_json(root / "data" / "queries_dev.json")
    entities = build_entity_dictionary(catalog, dev)
    save_json(root / "data" / "word_dictionary.json", word_dict)
    save_json(root / "data" / "word_display.json", display)
    save_json(root / "data" / "entity_dictionary.json", entities)
    return word_dict, entities
