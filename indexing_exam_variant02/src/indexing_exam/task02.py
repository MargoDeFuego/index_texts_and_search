from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from nltk.stem.snowball import SnowballStemmer

from .common import RU_STOPWORDS, ensure_corpus, is_date_token, precision_at_k, read_jsonl, recall_at_k, tokenize, write_csv, write_json

_STEM = SnowballStemmer("russian")
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("ru_core_news_sm", disable=["parser", "ner"])
        except Exception as exc:
            raise RuntimeError(
                "Russian spaCy model is required for lemmatization. Install: python -m spacy download ru_core_news_sm"
            ) from exc
    return _NLP


def norm_surface(token: str) -> str:
    return token.lower()


def norm_stem(token: str) -> str:
    if is_date_token(token):
        return token.lower()
    return _STEM.stem(token.lower())


def lemma_tokens(tokens: list[str]) -> list[str]:
    """Batch lemmatize tokens while preserving dates as distinct tokens."""
    if not tokens:
        return []
    nlp = _get_nlp()
    # Use tokenizer on space-joined tokens; dates are restored position-wise.
    doc = nlp(" ".join(tokens))
    lemmas = []
    doc_words = [t for t in doc if not t.is_space]
    if len(doc_words) != len(tokens):
        # Robust fallback: process token-by-token if spaCy tokenization differs (e.g. hyphens).
        for token in tokens:
            if is_date_token(token):
                lemmas.append(token.lower())
            else:
                d = nlp(token)
                lemmas.append((d[0].lemma_ if len(d) else token).lower())
        return lemmas
    for original, sp in zip(tokens, doc_words):
        lemmas.append(original.lower() if is_date_token(original) else (sp.lemma_ or original).lower())
    return lemmas


@dataclass
class MorphIndex:
    mode: str
    remove_stopwords: bool
    postings: dict[str, set[str]]
    documents: dict[str, dict]

    @classmethod
    def build(cls, corpus: list[dict], mode: str, remove_stopwords: bool) -> "MorphIndex":
        postings: dict[str, set[str]] = defaultdict(set)
        docs = {}
        for doc in corpus:
            doc_id = str(doc["doc_id"])
            docs[doc_id] = doc
            text = f"{doc.get('title','')} {doc.get('text','')}"
            tokens = tokenize(text)
            if remove_stopwords:
                tokens = [t for t in tokens if t not in RU_STOPWORDS]
            if mode == "surface":
                terms = [norm_surface(t) for t in tokens]
            elif mode == "stem":
                terms = [norm_stem(t) for t in tokens]
            elif mode == "lemma":
                terms = lemma_tokens(tokens)
            else:
                raise ValueError(mode)
            for term in terms:
                if term:
                    postings[term].add(doc_id)
        return cls(mode, remove_stopwords, dict(postings), docs)

    def normalize_query(self, query: str) -> list[str]:
        tokens = tokenize(query)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in RU_STOPWORDS]
        if self.mode == "surface":
            return [norm_surface(t) for t in tokens]
        if self.mode == "stem":
            return [norm_stem(t) for t in tokens]
        return lemma_tokens(tokens)

    def search(self, query: str) -> tuple[list[str], list[str], list[str]]:
        terms = self.normalize_query(query)
        unique_terms = list(dict.fromkeys(terms))
        missing = [t for t in unique_terms if t not in self.postings]
        scores = defaultdict(int)
        for term in unique_terms:
            for doc_id in self.postings.get(term, ()):
                scores[doc_id] += 1
        ranked = sorted(scores, key=lambda d: (-scores[d], d))
        return ranked, unique_terms, missing


def evaluate(index: MorphIndex, queries: list[dict]) -> tuple[dict, list[dict]]:
    rows = []
    p5s, r5s = [], []
    total_query_terms = 0
    missing_terms = 0
    for q in queries:
        if q.get("split") != "test":
            continue
        ranked, terms, missing = index.search(q["query"])
        rel = set(q.get("relevant_doc_ids", []))
        p5 = precision_at_k(ranked, rel, 5)
        r5 = recall_at_k(ranked, rel, 5)
        if rel:
            p5s.append(p5)
            r5s.append(r5)
        total_query_terms += len(set(terms))
        missing_terms += len(set(missing))
        rows.append({
            "qid": q["qid"],
            "query": q["query"],
            "mode": index.mode,
            "stopwords_removed": index.remove_stopwords,
            "precision@5": round(p5, 4),
            "recall@5": round(r5, 4),
            "top5": " ".join(ranked[:5]),
            "missing_terms": " ".join(missing),
            "relevant_count": len(rel),
        })
    summary = {
        "mode": index.mode,
        "stopwords_removed": index.remove_stopwords,
        "vocabulary_size": len(index.postings),
        "mean_precision@5_nonempty_gold": sum(p5s) / len(p5s) if p5s else 0.0,
        "mean_recall@5_nonempty_gold": sum(r5s) / len(r5s) if r5s else 0.0,
        "query_term_oov_fraction": missing_terms / total_query_terms if total_query_terms else 0.0,
    }
    return summary, rows


def run(corpus_path: str, query_path: str, out_dir: str) -> dict:
    corpus = read_jsonl(corpus_path)
    ensure_corpus(corpus, 120)
    queries = read_jsonl(query_path)
    if len(queries) < 30:
        raise ValueError("Task 02 requires at least 30 queries")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_rows = []
    for remove_stopwords in (False, True):
        for mode in ("surface", "stem", "lemma"):
            idx = MorphIndex.build(corpus, mode, remove_stopwords)
            summary, rows = evaluate(idx, queries)
            summaries.append(summary)
            all_rows.extend(rows)
            serializable = {term: sorted(ids) for term, ids in sorted(idx.postings.items())}
            suffix = "nostop" if remove_stopwords else "withstop"
            write_json(out / f"index_{mode}_{suffix}.json", serializable)

    historical_case = {}
    for mode in ("surface", "stem", "lemma"):
        idx = MorphIndex.build(corpus, mode, False)
        historical_case[mode] = {
            "град": idx.search("град")[0][:10],
            "город": idx.search("город")[0][:10],
            "note": "Different historical names/forms are not forcibly synonymized; dates and original text are preserved."
        }

    write_json(out / "summary.json", {
        "task": 2,
        "requirement": "three independent representations + stopword comparison",
        "results": summaries,
        "historical_modern_case": historical_case,
    })
    write_csv(out / "query_metrics.csv", all_rows)
    return {"results": summaries, "out_dir": str(out)}


def main():
    ap = argparse.ArgumentParser(description="Task 02 - morphological indexing")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--queries", default="data/task02_queries.jsonl")
    ap.add_argument("--out", default="results/task02")
    args = ap.parse_args()
    result = run(args.corpus, args.queries, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
