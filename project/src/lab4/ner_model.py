import json
from pathlib import Path
from .tokenizer import tokenize, normalize

LABELS = ["O", "PERSON", "ORG", "LOC"]


def token_features(tokens, i):
    w = tokens[i].text
    low = normalize(w)
    prev = normalize(tokens[i-1].text) if i > 0 else "<BOS>"
    nxt = normalize(tokens[i+1].text) if i + 1 < len(tokens) else "<EOS>"
    feats = {
        "bias": 1,
        f"w={low}": 1,
        f"p1={low[:1]}": 1,
        f"p2={low[:2]}": 1,
        f"p3={low[:3]}": 1,
        f"s1={low[-1:]}": 1,
        f"s2={low[-2:]}": 1,
        f"s3={low[-3:]}": 1,
        f"prev={prev}": 1,
        f"next={nxt}": 1,
        f"title={int(w[:1].isupper())}": 1,
        f"upper={int(w.isupper())}": 1,
        f"hyphen={int('-' in w)}": 1,
        f"lenbin={min(len(w)//3,5)}": 1,
        f"prev_title={int(i>0 and tokens[i-1].text[:1].isupper())}": 1,
        f"next_title={int(i+1<len(tokens) and tokens[i+1].text[:1].isupper())}": 1,
        f"prev+title={prev}|{int(w[:1].isupper())}": 1,
    }
    return feats


class OfflineNERModel:
    def __init__(self, weights, metadata=None):
        self.weights = weights
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["weights"], data.get("metadata", {}))

    def _score(self, feats, label):
        score = 0.0
        lw = self.weights.get(label, {})
        for feat in feats:
            score += lw.get(feat, 0.0)
        return score

    def predict_labels(self, text):
        tokens = tokenize(text)
        labels = []
        for i in range(len(tokens)):
            feats = token_features(tokens, i)
            scores = {lab: self._score(feats, lab) for lab in LABELS}
            # Детеминированный tie-break по порядку LABELS.
            label = max(LABELS, key=lambda lab: (scores[lab], -LABELS.index(lab)))
            labels.append(label)
        return tokens, labels

    def recognize(self, text):
        tokens, labels = self.predict_labels(text)
        entities = []
        i = 0
        while i < len(tokens):
            lab = labels[i]
            if lab == "O":
                i += 1
                continue
            j = i + 1
            while j < len(tokens) and labels[j] == lab:
                # Не объединяем через существенную пунктуацию/разрыв.
                gap = text[tokens[j-1].end:tokens[j].start]
                if gap not in (" ", "-"):
                    break
                j += 1
            start, end = tokens[i].start, tokens[j-1].end
            entities.append({
                "start": start, "end": end, "text": text[start:end],
                "type": lab, "source": "offline_model"
            })
            i = j
        return entities
