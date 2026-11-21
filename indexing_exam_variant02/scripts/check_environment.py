from __future__ import annotations
import sys

print("Python:", sys.version)
mods = ["nltk", "pymorphy3", "requests", "spacy"]
for name in mods:
    try:
        mod = __import__(name)
        print(f"{name}: OK", getattr(mod, "__version__", ""))
    except Exception as exc:
        print(f"{name}: ERROR - {exc}")
try:
    import spacy
    nlp = spacy.load("ru_core_news_sm")
    print("ru_core_news_sm: OK", nlp.meta.get("version", ""))
except Exception as exc:
    print("ru_core_news_sm: NOT READY -", exc)
