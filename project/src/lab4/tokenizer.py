import re
from dataclasses import dataclass, asdict

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)*")

@dataclass
class Token:
    text: str
    start: int
    end: int

    def to_dict(self):
        return asdict(self)


def tokenize(text: str):
    return [Token(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def normalize(text: str) -> str:
    # Политика лабораторной: регистр не учитываем; ё и е считаем эквивалентными.
    # Дефисы сохраняем, чтобы составные названия не распадались при нормализации.
    return text.lower().replace("ё", "е")


def restore_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper() and source[1:].islower():
        return replacement[:1].upper() + replacement[1:]
    if source.islower():
        return replacement.lower()
    return replacement
