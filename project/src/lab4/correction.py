from dataclasses import dataclass, asdict
from .tokenizer import tokenize, normalize, restore_case
from .levenshtein import levenshtein

@dataclass
class Candidate:
    word: str
    distance: int
    frequency: int

@dataclass
class Change:
    start: int
    end: int
    wrong: str
    correct: str
    distance: int
    alternatives: list
    new_start: int = 0
    new_end: int = 0

    def to_dict(self):
        d = asdict(self)
        d["alternatives"] = [asdict(x) if isinstance(x, Candidate) else x for x in self.alternatives]
        return d


def find_candidates(word, word_dictionary, display, threshold=1, use_frequencies=True, limit=3):
    key = normalize(word)
    out = []
    for cand, freq in word_dictionary.items():
        if abs(len(cand) - len(key)) > threshold:
            continue
        dist = levenshtein(key, cand)
        if dist <= threshold:
            out.append(Candidate(display.get(cand, cand), dist, int(freq)))
    if use_frequencies:
        out.sort(key=lambda x: (x.distance, -x.frequency, normalize(x.word)))
    else:
        out.sort(key=lambda x: (x.distance, normalize(x.word)))
    return out[:limit]


def _inside(span_start, span_end, protected):
    return any(span_start >= a and span_end <= b for a, b in protected)


def correct_text(text, word_dictionary, display, threshold=1, use_frequencies=True, protected_spans=None):
    protected_spans = protected_spans or []
    tokens = tokenize(text)
    replacements = {}
    changes = []
    for tok in tokens:
        if _inside(tok.start, tok.end, protected_spans):
            continue
        key = normalize(tok.text)
        # Однобуквенные/двухбуквенные служебные слова не корректируем.
        if len(key) < 3 or key in word_dictionary:
            continue
        candidates = find_candidates(tok.text, word_dictionary, display, threshold, use_frequencies)
        if not candidates:
            continue
        best = candidates[0]
        repl = restore_case(tok.text, best.word)
        if normalize(repl) == key:
            continue
        replacements[(tok.start, tok.end)] = repl
        changes.append(Change(tok.start, tok.end, tok.text, repl, best.distance, candidates))

    parts = []
    cursor = 0
    new_pos = 0
    change_by_span = {(c.start, c.end): c for c in changes}
    for tok in tokens:
        if (tok.start, tok.end) not in replacements:
            continue
        # добавляем участок до замены
        prefix = text[cursor:tok.start]
        parts.append(prefix)
        new_pos += len(prefix)
        repl = replacements[(tok.start, tok.end)]
        ch = change_by_span[(tok.start, tok.end)]
        ch.new_start = new_pos
        parts.append(repl)
        new_pos += len(repl)
        ch.new_end = new_pos
        cursor = tok.end
    tail = text[cursor:]
    parts.append(tail)
    corrected = "".join(parts)
    return corrected, [c.to_dict() for c in changes]


def map_span(start, end, changes):
    """Переносит [start,end) из исходного текста в текст после токеновых замен."""
    delta_start = 0
    delta_end = 0
    for ch in sorted(changes, key=lambda x: x["start"]):
        delta = len(ch["correct"]) - (ch["end"] - ch["start"])
        if ch["end"] <= start:
            delta_start += delta
        if ch["start"] < end:
            delta_end += delta
    return start + delta_start, end + delta_end


def suspicious_inside_entities(text, protected_spans, entity_dictionary, word_dictionary, threshold=1):
    # Рекомендации внутри защищённых сущностей: не применяются автоматически.
    vocab = set()
    for ent in entity_dictionary:
        for form in ent.get("forms", []):
            for t in tokenize(form):
                vocab.add(normalize(t.text))
    recommendations = []
    for tok in tokenize(text):
        if not _inside(tok.start, tok.end, protected_spans):
            continue
        key = normalize(tok.text)
        if key in word_dictionary:
            continue
        cands = []
        for v in vocab:
            if abs(len(v) - len(key)) <= threshold:
                d = levenshtein(key, v)
                if d <= threshold and v != key:
                    cands.append((d, v))
        cands.sort()
        if cands:
            recommendations.append({
                "start": tok.start, "end": tok.end, "word": tok.text,
                "candidates": [{"word": w, "distance": d} for d, w in cands[:3]]
            })
    return recommendations
