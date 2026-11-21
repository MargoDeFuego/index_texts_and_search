from __future__ import annotations

import argparse
import random
from pathlib import Path

from indexing_exam.common import write_jsonl

TOPICS = {
    "археология": [
        "археологические раскопки", "культурный слой", "древнее поселение", "керамика", "курган", "экспедиция"
    ],
    "архитектура": [
        "архитектурный ансамбль", "собор", "зодчество", "реставрация фасада", "усадьба", "кремль"
    ],
    "живопись": [
        "русская живопись", "художник", "полотно", "иконопись", "портрет", "Третьяковская галерея"
    ],
    "история_литературы": [
        "история литературы", "поэт", "писатель", "рукопись", "литературное наследие", "издание"
    ],
    "культурное_наследие": [
        "культурное наследие", "музей", "памятник", "охрана наследия", "историческая память", "реставрация"
    ],
}

NAMES = [
    "Александр Пушкин", "Лев Толстой", "Илья Репин", "Виктор Васнецов", "Андрей Рублёв",
    "Василий Баженов", "Матвей Казаков", "Николай Карамзин", "Сергей Соловьёв", "Иван Забелин"
]
PLACES = ["Москва", "Санкт-Петербург", "Новгород", "Псков", "Владимир", "Суздаль", "Казань", "Ярославль"]
YEARS = ["988", "1147", "1380", "1480", "1612", "1703", "1812", "1861", "1917", "1945"]
HISTORICAL = ["Царьград", "Константинополь", "Петроград", "Ленинград", "град", "уезд", "губерния"]


def make_doc(i: int, topic: str, rng: random.Random) -> dict:
    terms = TOPICS[topic]
    name = rng.choice(NAMES)
    place = rng.choice(PLACES)
    year = rng.choice(YEARS)
    historical = rng.choice(HISTORICAL)
    t = rng.sample(terms, min(4, len(terms)))
    title = f"Исследование: {t[0]} и исторический контекст {place}"
    body = (
        f"Статья посвящена теме «{t[0]}» в контексте истории и культурного наследия. "
        f"В источниках упоминаются {name}, {place} и дата {year}. "
        f"Авторы анализируют {t[1]}, {t[2]} и {t[3]}, сопоставляя материальные свидетельства, архивные описания и музейные коллекции. "
        f"Особое внимание уделено тому, как историческое название «{historical}» сохраняется в документах и не должно смешиваться с современной формой. "
        f"В работе обсуждаются методы датировки, атрибуции, сохранения памятников и интерпретации источников. "
        f"Результаты показывают связь между локальной историей, художественной традицией и практиками сохранения наследия. "
        f"Для проверки выводов приводятся сравнительные примеры, описания объектов и ссылки на предыдущие исследования."
    )
    return {
        "doc_id": f"D{i:04d}",
        "title": title,
        "text": body,
        "language": "ru",
        "topic": topic,
        "source": "demo://generated",
        "acquisition": "synthetic demo corpus; use fetch_openalex.py for exam data",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=220)
    ap.add_argument("--out", default="data/corpus.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    topics = list(TOPICS)
    docs = [make_doc(i, topics[(i - 1) % len(topics)], rng) for i in range(1, args.count + 1)]
    write_jsonl(args.out, docs)
    print(f"Saved {len(docs)} demo documents -> {args.out}")


if __name__ == "__main__":
    main()
