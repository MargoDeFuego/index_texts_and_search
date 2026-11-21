from collections import Counter
from indexing_exam.task11 import correct_text, remap_entities

text = "Покажи материалы об Александре Пукине в Москве"

freq = Counter({
    "пушкине": 10,
    "покажи": 5,
    "материалы": 5,
    "об": 10,
    "александре": 10,
    "в": 20,
    "москве": 10,
})

corrected, replacements = correct_text(
    text, freq, Counter(), threshold=1
)

start = text.index("Москве")
entities = [{
    "start": start,
    "end": start + len("Москве"),
    "text": "Москве",
    "type": "LOC",
}]

print("Исходный текст:", text)
print("Исправленный текст:", corrected)
print("Замены:", [r.__dict__ for r in replacements])
print("Сущность до:", entities[0])
print("Сущность после:", remap_entities(entities, replacements, corrected)[0])
