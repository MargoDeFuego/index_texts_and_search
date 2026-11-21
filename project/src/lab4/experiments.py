from .pipeline import process
from .evaluation import correction_metrics, run_combination


def run_dev_experiments(records, word_dictionary, display, entity_dictionary, model):
    result = {"distance_threshold": {}, "frequency_ranking": {}, "protection": {}}
    # Порог: меняем только threshold, коррекция без защиты.
    for th in (1, 2):
        outs = [process(r["original_text"], "rule", "A", word_dictionary, display, entity_dictionary,
                        model=model, threshold=th, use_frequencies=True) for r in records]
        result["distance_threshold"][str(th)] = correction_metrics(records, outs)
    # Частоты: фиксируем threshold=1 и меняем только ранжирование.
    for use_freq in (True, False):
        outs = [process(r["original_text"], "rule", "A", word_dictionary, display, entity_dictionary,
                        model=model, threshold=1, use_frequencies=use_freq) for r in records]
        result["frequency_ranking"]["with_frequency" if use_freq else "without_frequency"] = correction_metrics(records, outs)
    # Защита: B, один и тот же recognizer/config, меняется только protection.
    for rec in ("rule", "model"):
        result["protection"][rec] = {}
        for protect in (True, False):
            combo = run_combination(records, rec, "B", word_dictionary, display, entity_dictionary, model,
                                    threshold=1, use_frequencies=True, protect_entities=protect)
            result["protection"][rec]["protected" if protect else "unprotected"] = {
                "correction": combo["correction"], "ner": combo["ner"]
            }
    return result
