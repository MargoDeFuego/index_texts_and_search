import re
from .tokenizer import normalize, tokenize


def _boundary_ok(text, start, end):
    left = text[start - 1] if start > 0 else ""
    right = text[end] if end < len(text) else ""
    return (not left or not (left.isalnum() or left == "_")) and (not right or not (right.isalnum() or right == "_"))


def dictionary_entities(text, entity_dictionary):
    norm = normalize(text)
    candidates = []
    for ent in entity_dictionary:
        for form in ent.get("forms", []):
            f = normalize(form)
            for m in re.finditer(re.escape(f), norm):
                if _boundary_ok(norm, m.start(), m.end()):
                    candidates.append({
                        "start": m.start(), "end": m.end(),
                        "text": text[m.start():m.end()], "type": ent["type"],
                        "source": "dictionary", "entity_id": ent["id"]
                    })
    # При совпадениях на одной позиции выбираем самое длинное; затем неперекрывающиеся.
    candidates.sort(key=lambda x: (x["start"], -(x["end"] - x["start"]), x["type"]))
    selected = []
    for c in candidates:
        if any(not (c["end"] <= s["start"] or c["start"] >= s["end"]) for s in selected):
            continue
        selected.append(c)
    return sorted(selected, key=lambda x: x["start"])


def context_entities(text, occupied):
    tokens = tokenize(text)
    occ = [(e["start"], e["end"]) for e in occupied]
    trigger_type = {
        "компания": "ORG", "университет": "ORG", "завод": "ORG", "банк": "ORG",
        "город": "LOC", "остров": "LOC", "поселок": "LOC", "посёлок": "LOC",
        "профессор": "PERSON", "ученый": "PERSON", "учёный": "PERSON", "писатель": "PERSON"
    }
    out = []
    for i, tok in enumerate(tokens[:-1]):
        typ = trigger_type.get(normalize(tok.text))
        if not typ:
            continue
        nxt = tokens[i + 1]
        if any(nxt.start >= a and nxt.end <= b for a, b in occ):
            continue
        if not (nxt.text[:1].isupper() or nxt.text.isupper()):
            continue
        end_i = i + 1
        while end_i + 1 < len(tokens) and end_i - i < 3:
            cand = tokens[end_i + 1]
            gap = text[tokens[end_i].end:cand.start]
            if gap.strip() or not (cand.text[:1].isupper() or cand.text.isupper()):
                break
            end_i += 1
        start, end = nxt.start, tokens[end_i].end
        if any(not (end <= a or start >= b) for a, b in occ):
            continue
        ent = {"start": start, "end": end, "text": text[start:end], "type": typ, "source": "context_rule"}
        out.append(ent)
        occ.append((start, end))
    return out


def recognize_rule(text, entity_dictionary):
    base = dictionary_entities(text, entity_dictionary)
    extra = context_entities(text, base)
    return sorted(base + extra, key=lambda x: x["start"])
