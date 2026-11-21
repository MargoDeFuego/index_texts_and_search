from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict

from indexing_exam.common import read_jsonl, write_json, write_jsonl, tokenize

QUERY_TEMPLATES = {
    "археология": [
        "археологические раскопки", "культурный слой", "древнее поселение", "археология памятников",
        "датировка археологических находок", "курганы и раскопки"
    ],
    "архитектура": [
        "история архитектуры", "архитектурный ансамбль", "реставрация собора", "древнерусское зодчество",
        "историческая архитектура города", "архитектурное наследие"
    ],
    "живопись": [
        "русская живопись", "история живописи", "иконопись", "художник и полотно", "музейная живопись", "портрет в искусстве"
    ],
    "история_литературы": [
        "история русской литературы", "литературное наследие", "рукописи писателей", "история изданий", "поэт и эпоха", "архив писателя"
    ],
    "культурное_наследие": [
        "культурное наследие", "охрана памятников", "музей и наследие", "историческая память", "реставрация памятников", "сохранение наследия"
    ],
}

GAZETTEER = [
    {"id": "P1", "name": "Александр Пушкин", "type": "PERSON", "forms": ["Александр Пушкин", "Пушкин"]},
    {"id": "P2", "name": "Лев Толстой", "type": "PERSON", "forms": ["Лев Толстой", "Толстой"]},
    {"id": "P3", "name": "Илья Репин", "type": "PERSON", "forms": ["Илья Репин", "Репин"]},
    {"id": "P4", "name": "Андрей Рублёв", "type": "PERSON", "forms": ["Андрей Рублёв", "Рублёв"]},
    {"id": "P5", "name": "Василий Баженов", "type": "PERSON", "forms": ["Василий Баженов", "Баженов"]},
    {"id": "L1", "name": "Москва", "type": "LOC", "forms": ["Москва", "Москве", "Москвы"]},
    {"id": "L2", "name": "Санкт-Петербург", "type": "LOC", "forms": ["Санкт-Петербург", "Петербург"]},
    {"id": "L3", "name": "Пушкин", "type": "LOC", "forms": ["Пушкин"]},
    {"id": "L4", "name": "Новгород", "type": "LOC", "forms": ["Новгород", "Новгороде"]},
    {"id": "L5", "name": "Псков", "type": "LOC", "forms": ["Псков", "Пскове"]},
    {"id": "O1", "name": "Государственный Эрмитаж", "type": "ORG", "forms": ["Государственный Эрмитаж", "Эрмитаж"]},
    {"id": "O2", "name": "Третьяковская галерея", "type": "ORG", "forms": ["Третьяковская галерея", "Третьяковке"]},
    {"id": "O3", "name": "Русский музей", "type": "ORG", "forms": ["Русский музей"]},
    {"id": "O4", "name": "Московский Кремль", "type": "ORG", "forms": ["Московский Кремль", "Кремль"]},
]

SPECIAL_RARE = ["Феофан Грек", "Дионисий", "Аристотель Фиораванти", "Осип Бове", "Алексей Щусев"]


def prepare_task02(corpus: list[dict], out: str) -> None:
    topic_docs = defaultdict(list)
    for d in corpus:
        topic_docs[d.get("topic", "")].append(d["doc_id"])

    rows = []
    qnum = 1
    all_topics = [t for t in QUERY_TEMPLATES if topic_docs.get(t)]
    if not all_topics:
        all_topics = sorted(k for k, v in topic_docs.items() if v)
    for topic in all_topics:
        templates = QUERY_TEMPLATES.get(topic, [topic, f"история {topic}", f"исследование {topic}"])
        for q in templates:
            rows.append({
                "qid": f"Q{qnum:02d}",
                "query": q,
                "relevant_doc_ids": topic_docs[topic][:40],
                "topic": topic,
                "split": "tune" if qnum <= 10 else "test",
                "annotation": "topic label assigned before indexing; review if using external corpus"
            })
            qnum += 1
            if qnum > 29:
                break
        if qnum > 29:
            break
    while qnum <= 29:
        topic = all_topics[(qnum - 1) % len(all_topics)]
        rows.append({
            "qid": f"Q{qnum:02d}",
            "query": f"историческое исследование {topic.replace('_', ' ')} {qnum}",
            "relevant_doc_ids": topic_docs[topic][:40],
            "topic": topic,
            "split": "tune" if qnum <= 10 else "test",
            "annotation": "topic label assigned before indexing"
        })
        qnum += 1
    rows.append({
        "qid": "Q30",
        "query": "квантовая механика полупроводников",
        "relevant_doc_ids": [],
        "topic": "none",
        "split": "test",
        "annotation": "required no-relevant-documents control query"
    })
    write_jsonl(out, rows)


def prepare_task03() -> None:
    docs = [
        {"doc_id": "S1", "title": "Кремль", "text": "Московский Кремль был перестроен в 1480 году и стал важным памятником архитектуры."},
        {"doc_id": "S2", "title": "Новгород", "text": "В 1044 году в Новгороде началось строительство каменных укреплений и собора."},
        {"doc_id": "S3", "title": "Музей", "text": "Музей истории архитектуры хранит чертежи и обзор реставрации."},
        {"doc_id": "S4", "title": "Живопись", "text": "Русская живопись XIX века представлена работами Ильи Репина и других художников."},
        {"doc_id": "S5", "title": "Петроград", "text": "Петроград в 1917 году стал центром событий; позднее город назывался Ленинград."},
        {"doc_id": "S6", "title": "Собор", "text": "Собор древнего Пскова находится рядом с крепостью и датируется XII веком."},
        {"doc_id": "S7", "title": "Пушкин", "text": "Город Пушкин связан с дворцовым ансамблем, а Александр Пушкин связан с историей литературы."},
        {"doc_id": "S8", "title": "Археология", "text": "Археология изучает культурный слой древнего поселения и фиксирует дату 1380 года."},
    ]
    checks = [
        ("C01", "музей", ["S3"]),
        ("C02", '"истории архитектуры"', ["S3"]),
        ("C03", "музей AND NOT обзор", []),
        ("C04", "музей AND обзор", ["S3"]),
        ("C05", "живопись OR археология", ["S4", "S8"]),
        ("C06", "Москва OR Московский", ["S1"]),
        ("C07", "1480", ["S1"]),
        ("C08", "1917 AND Петроград", ["S5"]),
        ("C09", "Петроград NEAR/3 1917", ["S5"]),
        ("C10", "1917 NEAR/3 центром", ["S5"]),
        ("C11", '"культурный слой"', ["S8"]),
        ("C12", "культурный NEAR/2 слой", ["S8"]),
        ("C13", "Пушкин AND город", ["S7"]),
        ("C14", '"Александр Пушкин"', ["S7"]),
        ("C15", "(собор OR собора) AND (Пскова OR Новгород)", ["S2", "S6"]),
        ("C16", "NOT музей", ["S1", "S2", "S4", "S5", "S6", "S7", "S8"]),
        ("C17", "неизвестныйтермин", []),
        ("C18", '"музей музей"', []),
        ("C19", "(археология AND 1380) OR (живопись AND Репина)", ["S4", "S8"]),
        ("C20", "Ленинград OR Петроград", ["S5"]),
    ]
    write_jsonl("data/task03/small_corpus.jsonl", docs)
    write_jsonl("data/task03/checks.jsonl", [
        {"id": cid, "query": q, "expected": exp} for cid, q, exp in checks
    ])


def typo_one(word: str) -> str:
    if len(word) < 4:
        return word + "а"
    i = max(1, len(word) // 2 - 1)
    return word[:i] + word[i+1:]


def entity_spans(text: str, entities: list[tuple[str, str]]) -> list[dict]:
    spans = []
    used = set()
    for surface, etype in entities:
        start = text.find(surface)
        if start < 0:
            continue
        key = (start, start + len(surface), etype)
        if key in used:
            continue
        used.add(key)
        spans.append({"start": start, "end": start + len(surface), "text": surface, "type": etype})
    return sorted(spans, key=lambda x: (x["start"], x["end"]))


def build_task11_queries(out: str) -> None:
    rng = random.Random(17)
    base = [
        ("Покажи материалы об Александре Пушкине в Москве", [("Александре Пушкине", "PERSON"), ("Москве", "LOC")]),
        ("Найди картины Ильи Репина в Третьяковской галерее", [("Ильи Репина", "PERSON"), ("Третьяковской галерее", "ORG")]),
        ("Что известно об Андрее Рублёве и Русском музее", [("Андрее Рублёве", "PERSON"), ("Русском музее", "ORG")]),
        ("История Государственного Эрмитажа в Санкт-Петербурге", [("Государственного Эрмитажа", "ORG"), ("Санкт-Петербурге", "LOC")]),
        ("Архитектура Московского Кремля и Москвы", [("Московского Кремля", "ORG"), ("Москвы", "LOC")]),
        ("Город Пушкин и его дворцовые ансамбли", [("Пушкин", "LOC")]),
        ("Поэт Пушкин и история литературы", [("Пушкин", "PERSON")]),
        ("Памятники Новгорода и Пскова", [("Новгорода", "LOC"), ("Пскова", "LOC")]),
    ]
    rows = []
    idx = 1
    categories = ["correct", "ordinary_typo", "entity_typo", "rare_correct"]
    for category in categories:
        target = 15
        for j in range(target):
            text, ents = base[j % len(base)]
            correct = text
            typos = []
            if category == "ordinary_typo":
                candidates = [w for w in re.finditer(r"[А-Яа-яЁё]+", text) if len(w.group()) >= 5]
                # avoid entity spans where possible
                gold = entity_spans(text, ents)
                def in_entity(m):
                    return any(m.start() >= e["start"] and m.end() <= e["end"] for e in gold)
                cand = next((m for m in candidates if not in_entity(m)), candidates[0])
                old = cand.group()
                new = typo_one(old)
                text = text[:cand.start()] + new + text[cand.end():]
                typos.append({"start": cand.start(), "end": cand.start()+len(new), "original": new, "correct": old})
            elif category == "entity_typo":
                surface, _ = ents[0]
                pos = text.find(surface)
                words = list(re.finditer(r"[А-Яа-яЁё]+", surface))
                w = words[-1]
                old_word = w.group()
                new_word = typo_one(old_word)
                absolute = pos + w.start()
                text = text[:absolute] + new_word + text[absolute + len(old_word):]
                typos.append({"start": absolute, "end": absolute+len(new_word), "original": new_word, "correct": old_word})
                # replace entity surface in annotation with error form
                bad_surface = surface[:w.start()] + new_word + surface[w.end():]
                ents = [(bad_surface if s == surface else s, t) for s, t in ents]
            elif category == "rare_correct":
                rare = SPECIAL_RARE[j % len(SPECIAL_RARE)]
                text = f"Найди сведения о {rare} в истории архитектуры и искусства"
                correct = text
                ents = [(rare, "PERSON")]
            gold = entity_spans(text, ents)
            tune_limit = 8 if category in {"correct", "ordinary_typo"} else 7
            rows.append({
                "qid": f"N{idx:02d}",
                "text": text,
                "correct_text": correct,
                "category": category,
                "typos": typos,
                "entities": gold,
                "split": "tune" if j < tune_limit else "test"
            })
            idx += 1
    write_jsonl(out, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    args = ap.parse_args()
    corpus = read_jsonl(args.corpus)
    prepare_task02(corpus, "data/task02_queries.jsonl")
    prepare_task03()
    build_task11_queries("data/task11/queries.jsonl")
    write_json("data/task11/gazetteer_seed.json", GAZETTEER)
    print("Prepared task 02 queries, task 03 checks, and task 11 query set.")


if __name__ == "__main__":
    main()
