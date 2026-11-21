from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

TOKEN_RE = re.compile(
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{3,4}|[A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*",
    re.UNICODE,
)

RU_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только",
    "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда",
    "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас",
    "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они",
    "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб",
    "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того",
    "потому", "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы",
    "нее", "сейчас", "были", "куда", "зачем", "сказать", "всех", "никогда", "сегодня", "можно", "при",
    "наконец", "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти", "нас", "про",
    "всего", "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
    "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда", "конечно", "всю",
    "между", "это", "является", "также", "которые", "который", "которая", "которое", "которых"
}


def tokenize(text: str, lower: bool = True) -> list[str]:
    tokens = TOKEN_RE.findall(text or "")
    return [t.lower() for t in tokens] if lower else tokens


def is_date_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{3,4}", token))


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    items: list[dict] = []
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {p}:{line_no}: {exc}") from exc
    return items


def write_jsonl(path: str | Path, items: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def precision_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / len(relevant)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2


def ensure_corpus(corpus: list[dict], minimum: int) -> None:
    if len(corpus) < minimum:
        raise ValueError(f"Need at least {minimum} documents, got {len(corpus)}")
    ids = [str(d.get("doc_id", "")) for d in corpus]
    if len(ids) != len(set(ids)):
        raise ValueError("doc_id values must be unique")


def corpus_word_frequencies(corpus: list[dict]) -> Counter[str]:
    c: Counter[str] = Counter()
    for doc in corpus:
        c.update(tokenize((doc.get("title") or "") + " " + (doc.get("text") or "")))
    return c
