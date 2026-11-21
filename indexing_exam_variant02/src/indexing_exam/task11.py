from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .common import corpus_word_frequencies, prf, read_jsonl, tokenize, write_csv, write_json

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)*", re.UNICODE)


def levenshtein(a: str, b: str) -> int:
    """Classic dynamic-programming Levenshtein distance: insert/delete/replace."""
    a = a.lower(); b = b.lower()
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, 1):
        cur = [j]
        for i, ca in enumerate(a, 1):
            cur.append(min(
                cur[-1] + 1,
                prev[i] + 1,
                prev[i - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


@dataclass
class Replacement:
    start: int
    end: int
    old: str
    new: str
    distance: int


def overlaps(span: tuple[int, int], protected: list[tuple[int, int]]) -> bool:
    a, b = span
    return any(a < d and c < b for c, d in protected)


def candidate_forms(freq: Counter[str], extra: Counter[str], token: str, max_dist: int) -> list[tuple[int, int, str]]:
    low = token.lower()
    # Length filter greatly reduces comparisons while preserving edit distance <= max_dist.
    candidates = set(freq) | set(extra)
    out = []
    for cand in candidates:
        if abs(len(cand) - len(low)) > max_dist:
            continue
        if not cand.isalpha() and "-" not in cand:
            continue
        d = levenshtein(low, cand)
        if d <= max_dist:
            out.append((d, -(freq.get(cand, 0) + extra.get(cand, 0)), cand))
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    return out


def preserve_case(src: str, replacement: str) -> str:
    if src.istitle():
        return replacement[:1].upper() + replacement[1:]
    if src.isupper():
        return replacement.upper()
    return replacement


def correct_text(text: str, freq: Counter[str], extra: Counter[str], threshold: int,
                 protected: list[tuple[int, int]] | None = None) -> tuple[str, list[Replacement]]:
    protected = protected or []
    reps: list[Replacement] = []
    for m in WORD_RE.finditer(text):
        if overlaps((m.start(), m.end()), protected):
            continue
        token = m.group()
        low = token.lower()
        if low in freq or low in extra or len(low) <= 2:
            continue
        cands = candidate_forms(freq, extra, token, threshold)
        if not cands:
            continue
        d, _, cand = cands[0]
        new = preserve_case(token, cand)
        if new != token:
            reps.append(Replacement(m.start(), m.end(), token, new, d))

    if not reps:
        return text, []
    parts = []
    pos = 0
    for r in reps:
        parts.append(text[pos:r.start])
        parts.append(r.new)
        pos = r.end
    parts.append(text[pos:])
    return "".join(parts), reps


def remap_position(pos: int, reps: list[Replacement], prefer_end: bool = False) -> int:
    delta = 0
    for r in reps:
        if pos < r.start:
            break
        if r.start <= pos <= r.end:
            if pos == r.end or prefer_end:
                return r.start + delta + len(r.new)
            return r.start + delta
        delta += len(r.new) - (r.end - r.start)
    return pos + delta


def remap_entities(entities: list[dict], reps: list[Replacement], new_text: str) -> list[dict]:
    out = []
    for e in entities:
        ns = remap_position(e["start"], reps, False)
        ne = remap_position(e["end"], reps, True)
        out.append({"start": ns, "end": ne, "text": new_text[ns:ne], "type": e["type"]})
    return out


def normalize_entity_type(label: str) -> str | None:
    mapping = {"PER": "PERSON", "PERSON": "PERSON", "ORG": "ORG", "LOC": "LOC", "GPE": "LOC"}
    return mapping.get(label)


def gazetteer_forms(gazetteer: list[dict]) -> Counter[str]:
    c = Counter()
    for e in gazetteer:
        for form in e.get("forms", []):
            for token in tokenize(form):
                if token.isalpha() or "-" in token:
                    c[token] += 1000
    return c


def gazetteer_ner(text: str, gazetteer: list[dict]) -> list[dict]:
    """Longest exact dictionary matching with a small explicit ambiguity rule for Пушкин."""
    candidates = []
    low = text.lower()
    for entry in gazetteer:
        for form in entry.get("forms", []):
            pattern = re.compile(r"(?<![A-Za-zА-Яа-яЁё])" + re.escape(form) + r"(?![A-Za-zА-Яа-яЁё])", re.I)
            for m in pattern.finditer(text):
                candidates.append({"start": m.start(), "end": m.end(), "text": m.group(), "type": entry["type"], "id": entry["id"]})
    candidates.sort(key=lambda e: (e["start"], -(e["end"] - e["start"]), e["type"]))
    chosen = []
    occupied = []
    for c in candidates:
        if overlaps((c["start"], c["end"]), occupied):
            continue
        if c["text"].lower() == "пушкин":
            ctx = low[max(0, c["start"]-25): min(len(low), c["end"]+25)]
            if "город" in ctx or "дворц" in ctx:
                c["type"] = "LOC"
            elif "поэт" in ctx or "александр" in ctx or "литератур" in ctx:
                c["type"] = "PERSON"
        chosen.append({k: c[k] for k in ("start", "end", "text", "type")})
        occupied.append((c["start"], c["end"]))
    return sorted(chosen, key=lambda e: e["start"])


def load_spacy_model():
    try:
        import spacy
        return spacy.load("ru_core_news_sm")
    except Exception as exc:
        raise RuntimeError(
            "spaCy Russian model is required. Install it with: python -m spacy download ru_core_news_sm"
        ) from exc


def spacy_ner(text: str, nlp) -> list[dict]:
    doc = nlp(text)
    out = []
    for ent in doc.ents:
        t = normalize_entity_type(ent.label_)
        if t:
            out.append({"start": ent.start_char, "end": ent.end_char, "text": ent.text, "type": t})
    return out


def ner_counts(pred: list[dict], gold: list[dict]) -> tuple[int, int, int]:
    p = {(e["start"], e["end"], e["type"]) for e in pred}
    g = {(e["start"], e["end"], e["type"]) for e in gold}
    return len(p & g), len(p - g), len(g - p)


def typo_correction_counts(item: dict, reps: list[Replacement]) -> tuple[int, int]:
    # Match replacement to original typo span and expected token.
    correct = 0
    total = len(item.get("typos", []))
    by_span = {(r.start, r.end): r for r in reps}
    for t in item.get("typos", []):
        r = by_span.get((t["start"], t["end"]))
        if r and r.new.lower() == t["correct"].lower():
            correct += 1
    return correct, total


def count_correct_tokens(text: str, typo_spans: list[tuple[int, int]]) -> int:
    return sum(1 for m in WORD_RE.finditer(text) if not overlaps((m.start(), m.end()), typo_spans))


def count_damage(reps: list[Replacement], typo_spans: list[tuple[int, int]]) -> int:
    return sum(1 for r in reps if not overlaps((r.start, r.end), typo_spans))


def build_gazetteer_from_tune(seed_gazetteer: list[dict], queries: list[dict]) -> list[dict]:
    # The assignment allows the directory to be built from corpus or tuning data.
    # Start from seed entries that are actually observed in tune text, then add exact
    # surface variants annotated in the tuning portion. Test annotations are never used.
    tune = [q for q in queries if q["split"] == "tune"]
    tune_text = " ".join(q["correct_text"] for q in tune).lower()
    selected = []
    for e in seed_gazetteer:
        if any(form.lower() in tune_text for form in e.get("forms", [])):
            selected.append({**e, "forms": list(dict.fromkeys(e.get("forms", [])))})

    # Add inflected/observed surfaces from tune annotation as independent gazetteer records.
    # This models the allowed expert-built directory without leaking test labels.
    seen = {(e["type"], f.lower()) for e in selected for f in e.get("forms", [])}
    auto_id = 1
    for q in tune:
        for ent in q.get("entities", []):
            surface = ent["text"]
            key = (ent["type"], surface.lower())
            if key in seen:
                continue
            selected.append({"id": f"TUNE{auto_id}", "name": surface, "type": ent["type"], "forms": [surface]})
            seen.add(key)
            auto_id += 1
    return selected


def pipeline(item: dict, threshold: int, ner_kind: str, order: str, freq: Counter[str], extra: Counter[str], gazetteer: list[dict], nlp=None):
    text = item["text"]
    ner_fn = (lambda t: gazetteer_ner(t, gazetteer)) if ner_kind == "gazetteer" else (lambda t: spacy_ner(t, nlp))
    if order == "A":
        corrected, reps = correct_text(text, freq, extra, threshold)
        pred = ner_fn(corrected)
    elif order == "B":
        first_ner = ner_fn(text)
        protected = [(e["start"], e["end"]) for e in first_ner]
        corrected, reps = correct_text(text, freq, extra, threshold, protected)
        pred = ner_fn(corrected)
    else:
        raise ValueError(order)
    gold = remap_entities(item.get("entities", []), reps, corrected)
    return corrected, reps, pred, gold


def build_entity_catalog(corpus: list[dict], nlp, limit: int = 100) -> dict:
    catalog: dict[str, dict[str, set[str]]] = {"PERSON": defaultdict(set), "ORG": defaultdict(set), "LOC": defaultdict(set)}
    for d in corpus[:limit]:
        text = f"{d.get('title','')} {d.get('text','')}"
        for e in spacy_ner(text, nlp):
            catalog[e["type"]][e["text"]].add(str(d["doc_id"]))
    return {t: {name: sorted(ids) for name, ids in sorted(names.items())} for t, names in catalog.items()}


def run(corpus_path: str, queries_path: str, gazetteer_path: str, out_dir: str, no_spacy: bool = False) -> dict:
    corpus = read_jsonl(corpus_path)
    queries = read_jsonl(queries_path)
    if len(queries) < 60:
        raise ValueError("Task 11 requires at least 60 queries")
    seed_gazetteer = json.loads(Path(gazetteer_path).read_text(encoding="utf-8"))
    gazetteer = build_gazetteer_from_tune(seed_gazetteer, queries)
    freq = corpus_word_frequencies(corpus)
    # tuning query words can supplement the common vocabulary; test text is not used.
    tune_extra = Counter()
    for q in queries:
        if q["split"] == "tune":
            tune_extra.update(tokenize(q["correct_text"]))
    extra = tune_extra + gazetteer_forms(gazetteer)
    nlp = None if no_spacy else load_spacy_model()

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    configs = []
    details = []
    ner_kinds = ["gazetteer"] + ([] if no_spacy else ["spacy"])
    for threshold in (1, 2):
        for ner_kind in ner_kinds:
            for order in ("A", "B"):
                tp = fp = fn = 0
                corrected_ok = typo_total = damages = correct_tokens = 0
                error_examples = []
                for item in queries:
                    if item["split"] != "test":
                        continue
                    corrected, reps, pred, gold = pipeline(item, threshold, ner_kind, order, freq, extra, gazetteer, nlp)
                    c, t = typo_correction_counts(item, reps)
                    corrected_ok += c; typo_total += t
                    typo_spans = [(x["start"], x["end"]) for x in item.get("typos", [])]
                    damages += count_damage(reps, typo_spans)
                    correct_tokens += count_correct_tokens(item["text"], typo_spans)
                    a, b, cfn = ner_counts(pred, gold)
                    tp += a; fp += b; fn += cfn
                    if len(error_examples) < 5 and (c < t or b or cfn or count_damage(reps, typo_spans)):
                        error_examples.append({
                            "qid": item["qid"], "input": item["text"], "output": corrected,
                            "replacements": [r.__dict__ for r in reps], "gold_entities": gold, "pred_entities": pred
                        })
                    details.append({
                        "qid": item["qid"], "threshold": threshold, "ner": ner_kind, "order": order,
                        "category": item["category"], "input": item["text"], "output": corrected,
                        "replacements": json.dumps([r.__dict__ for r in reps], ensure_ascii=False),
                        "gold_entities": json.dumps(gold, ensure_ascii=False),
                        "pred_entities": json.dumps(pred, ensure_ascii=False),
                    })
                p, r, f1 = prf(tp, fp, fn)
                configs.append({
                    "threshold": threshold,
                    "ner": ner_kind,
                    "order": order,
                    "exact_typo_top1_accuracy": corrected_ok / typo_total if typo_total else 0.0,
                    "damage_fraction": damages / correct_tokens if correct_tokens else 0.0,
                    "ner_precision_exact_span_type": p,
                    "ner_recall_exact_span_type": r,
                    "ner_f1_exact_span_type": f1,
                    "typos_evaluated": typo_total,
                    "correct_tokens_evaluated": correct_tokens,
                    "error_examples": error_examples,
                })

    homonym_examples = []
    for item in queries:
        if "Пушкин" in item["text"]:
            homonym_examples.append({"text": item["text"], "gazetteer_ner": gazetteer_ner(item["text"], gazetteer), "gold": item["entities"]})
            if len(homonym_examples) >= 4:
                break

    catalog = build_entity_catalog(corpus, nlp, 100) if nlp is not None else {}
    write_json(out / "entity_catalog.json", catalog)
    write_json(out / "gazetteer_from_tune.json", gazetteer)
    write_csv(out / "query_details.csv", details)
    summary = {
        "task": 11,
        "queries": len(queries),
        "test_queries": sum(1 for q in queries if q["split"] == "test"),
        "levenshtein_examples": {
            "музй->музей": levenshtein("музй", "музей"),
            "эрмитаж->эрмитаж": levenshtein("эрмитаж", "эрмитаж"),
        },
        "configs": configs,
        "homonym_surname_city_examples": homonym_examples,
        "catalog_note": "entity_catalog.json supports lookup by type -> normalized surface -> doc_id list",
    }
    write_json(out / "summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Task 11 - typo correction and named entities")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--queries", default="data/task11/queries.jsonl")
    ap.add_argument("--gazetteer", default="data/task11/gazetteer_seed.json")
    ap.add_argument("--out", default="results/task11")
    ap.add_argument("--no-spacy", action="store_true", help="Smoke-test only; full exam comparison requires spaCy ru_core_news_sm.")
    args = ap.parse_args()
    result = run(args.corpus, args.queries, args.gazetteer, args.out, args.no_spacy)
    print(json.dumps({"task": result["task"], "configs": result["configs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
