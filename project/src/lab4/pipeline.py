from .correction import correct_text, suspicious_inside_entities
from .ner_rule import recognize_rule


def recognize(text, recognizer, entity_dictionary, model=None):
    if recognizer == "rule":
        return recognize_rule(text, entity_dictionary)
    if recognizer == "model":
        if model is None:
            raise RuntimeError("Offline NER model is not loaded")
        return model.recognize(text)
    raise ValueError(f"Unknown recognizer: {recognizer}")


def process(text, recognizer, order, word_dictionary, display, entity_dictionary, model=None,
            threshold=1, use_frequencies=True, protect_entities=True):
    if not text.strip():
        return {
            "original_text": text, "proposed_text": text, "changes": [],
            "alternatives": [], "entities": [], "entity_recommendations": [],
            "recognizer": recognizer, "order": order, "note": "Пустой ввод"
        }

    if order == "A":
        proposed, changes = correct_text(text, word_dictionary, display, threshold, use_frequencies)
        entities = recognize(proposed, recognizer, entity_dictionary, model)
        protected = []
        recs = []
    elif order == "B":
        first_entities = recognize(text, recognizer, entity_dictionary, model)
        protected = [(e["start"], e["end"]) for e in first_entities] if protect_entities else []
        recs = suspicious_inside_entities(text, protected, entity_dictionary, word_dictionary, threshold)
        proposed, changes = correct_text(
            text, word_dictionary, display, threshold, use_frequencies, protected_spans=protected
        )
        entities = recognize(proposed, recognizer, entity_dictionary, model)
    else:
        raise ValueError("order must be A or B")

    return {
        "original_text": text,
        "proposed_text": proposed,
        "changes": changes,
        "entities": entities,
        "recognizer": recognizer,
        "order": order,
        "threshold": threshold,
        "use_frequencies": bool(use_frequencies),
        "protect_entities": bool(protect_entities),
        "protected_spans": protected,
        "entity_recommendations": recs,
        "entity_positions_refer_to": "proposed_text"
    }
