from collections import defaultdict
from .tokenizer import tokenize
from .correction import map_span
from .pipeline import process, recognize


def correction_metrics(records, outputs):
    total_typos = 0
    correct_typos = 0
    total_correct_tokens = 0
    false_changes = 0
    top3_hits = 0
    for rec, out in zip(records, outputs):
        typo_spans = {(t["start"], t["end"]): t for t in rec.get("typos", [])}
        total_typos += len(typo_spans)
        token_spans = {(t.start, t.end) for t in tokenize(rec["original_text"])}
        total_correct_tokens += len(token_spans) - len(typo_spans)
        changes = {(c["start"], c["end"]): c for c in out.get("changes", [])}
        for span, typo in typo_spans.items():
            c = changes.get(span)
            if c and c["correct"].casefold().replace("ё","е") == typo["correct"].casefold().replace("ё","е"):
                correct_typos += 1
            if c:
                alt_words = [a["word"].casefold().replace("ё","е") for a in c.get("alternatives", [])]
                if typo["correct"].casefold().replace("ё","е") in alt_words:
                    top3_hits += 1
        for span in changes:
            if span not in typo_spans:
                false_changes += 1
    return {
        "typos_total": total_typos,
        "typos_correct_top1": correct_typos,
        "typo_accuracy": correct_typos / total_typos if total_typos else 0.0,
        "top3_hits": top3_hits,
        "top3_recall": top3_hits / total_typos if total_typos else 0.0,
        "correct_tokens_total": total_correct_tokens,
        "false_changes": false_changes,
        "false_change_rate": false_changes / total_correct_tokens if total_correct_tokens else 0.0,
    }


def _entity_key(e):
    return (e["start"], e["end"], e["type"])


def entity_metrics(records, outputs=None, baseline_predictions=None):
    overall = {"TP":0,"FP":0,"FN":0}
    per_type = {t:{"TP":0,"FP":0,"FN":0} for t in ("PERSON","ORG","LOC")}
    for idx, rec in enumerate(records):
        if baseline_predictions is not None:
            pred = baseline_predictions[idx]
            gold = rec.get("original_entities", [])
        else:
            out = outputs[idx]
            pred = out.get("entities", [])
            gold = []
            for e in rec.get("original_entities", []):
                ns, ne = map_span(e["start"], e["end"], out.get("changes", []))
                gold.append({"start":ns,"end":ne,"type":e["type"]})
        pset = {_entity_key(e) for e in pred}
        gset = {_entity_key(e) for e in gold}
        for item in pset & gset:
            overall["TP"] += 1; per_type[item[2]]["TP"] += 1
        for item in pset - gset:
            overall["FP"] += 1
            if item[2] in per_type: per_type[item[2]]["FP"] += 1
        for item in gset - pset:
            overall["FN"] += 1; per_type[item[2]]["FN"] += 1

    def calc(d):
        tp, fp, fn = d["TP"], d["FP"], d["FN"]
        p = tp/(tp+fp) if tp+fp else 0.0
        r = tp/(tp+fn) if tp+fn else 0.0
        f = 2*p*r/(p+r) if p+r else 0.0
        return {**d, "precision":p, "recall":r, "f1":f}
    return {"overall":calc(overall), "per_type":{k:calc(v) for k,v in per_type.items()}}


def run_combination(records, recognizer, order, word_dictionary, display, entity_dictionary, model,
                    threshold=1, use_frequencies=True, protect_entities=True):
    outputs = [process(
        r["original_text"], recognizer, order, word_dictionary, display, entity_dictionary,
        model=model, threshold=threshold, use_frequencies=use_frequencies,
        protect_entities=protect_entities
    ) for r in records]
    return {
        "outputs": outputs,
        "correction": correction_metrics(records, outputs),
        "ner": entity_metrics(records, outputs=outputs)
    }


def baseline_ner(records, recognizer, entity_dictionary, model):
    predictions = [recognize(r["original_text"], recognizer, entity_dictionary, model) for r in records]
    return {"predictions": predictions, "metrics": entity_metrics(records, baseline_predictions=predictions)}
