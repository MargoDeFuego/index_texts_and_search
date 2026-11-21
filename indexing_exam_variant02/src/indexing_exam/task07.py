from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .common import ensure_corpus, median, prf, read_jsonl, tokenize, write_csv, write_json, write_jsonl


def word_shingles(text: str, k: int) -> set[tuple[str, ...]]:
    tokens = tokenize(text)
    if len(tokens) < k:
        return set()
    return {tuple(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set, b: set) -> float | None:
    if not a or not b:
        return None
    return len(a & b) / len(a | b)


def stable_hash(shingle: tuple[str, ...], seed: int) -> int:
    payload = f"{seed}|{' '.join(shingle)}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def minhash_signature(shingles: set[tuple[str, ...]], num_hashes: int, seed: int = 42) -> list[int] | None:
    if not shingles:
        return None
    sig = []
    for i in range(num_hashes):
        s = seed + i * 1000003
        sig.append(min(stable_hash(sh, s) for sh in shingles))
    return sig


def minhash_similarity(sig_a: list[int] | None, sig_b: list[int] | None) -> float | None:
    if not sig_a or not sig_b:
        return None
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def edit_text(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 20:
        return text
    out = []
    for i, w in enumerate(words):
        if i % 17 == 5:
            continue
        if i % 23 == 9:
            out.append("исторический")
        out.append(w)
    # keep order and most content; add a short editorial sentence
    out.extend(["Текст", "содержит", "небольшие", "редакторские", "правки."])
    return " ".join(out)


def make_pairs(corpus: list[dict], out_path: str, seed: int = 42) -> list[dict]:
    ensure_corpus(corpus, 100)
    rng = random.Random(seed)
    docs = corpus[:120]
    by_topic: dict[str, list[dict]] = {}
    for d in docs:
        by_topic.setdefault(d.get("topic", "unknown"), []).append(d)

    # Pre-assign split per source document, so pairs sharing a source cannot cross parts.
    split_by_doc = {}
    for d in docs:
        h = int(hashlib.sha1(str(d["doc_id"]).encode()).hexdigest()[:8], 16)
        split_by_doc[d["doc_id"]] = "tune" if h % 2 == 0 else "test"

    pairs = []
    used = set()

    def add_pair(a, b_text, label, kind, b_source=None, note=""):
        split = split_by_doc[a["doc_id"]]
        if b_source is not None and split_by_doc[b_source["doc_id"]] != split:
            return False
        key = (a["doc_id"], b_source["doc_id"] if b_source else kind, kind)
        if key in used:
            return False
        used.add(key)
        pairs.append({
            "pair_id": f"P{len(pairs)+1:03d}",
            "source_a": a["doc_id"],
            "source_b": b_source["doc_id"] if b_source else a["doc_id"],
            "text_a": f"{a.get('title','')} {a.get('text','')}",
            "text_b": b_text,
            "duplicate": label,
            "kind": kind,
            "split": split,
            "note": note,
        })
        return True

    # positives: exact copies and lightly edited variants
    pos_docs = docs[:40]
    for i, d in enumerate(pos_docs):
        if len(pairs) >= 30:
            break
        text = f"{d.get('title','')} {d.get('text','')}"
        if i % 2 == 0:
            add_pair(d, text, True, "copy")
        else:
            add_pair(d, edit_text(text, rng), True, "edited_copy", note="main content and fragment order preserved")

    # thematic negatives: same topic, different source docs
    for topic, arr in by_topic.items():
        for i in range(len(arr) - 1):
            if len([p for p in pairs if p["kind"] == "thematic_similar"]) >= 15:
                break
            a, b = arr[i], arr[i+1]
            if split_by_doc[a["doc_id"]] != split_by_doc[b["doc_id"]]:
                continue
            btxt = f"{b.get('title','')} {b.get('text','')}"
            add_pair(a, btxt, False, "thematic_similar", b_source=b, note="same topic but different document")

    # unrelated negatives from different topics
    topic_names = list(by_topic)
    attempts = 0
    while len([p for p in pairs if p["kind"] == "unrelated"]) < 14 and attempts < 1000:
        attempts += 1
        ta, tb = rng.sample(topic_names, 2)
        a, b = rng.choice(by_topic[ta]), rng.choice(by_topic[tb])
        if split_by_doc[a["doc_id"]] != split_by_doc[b["doc_id"]]:
            continue
        btxt = f"{b.get('title','')} {b.get('text','')}"
        add_pair(a, btxt, False, "unrelated", b_source=b)

    # required quote-inside-different-texts negative case
    quote = "Культурное наследие требует бережного отношения к источникам и памятникам."
    found_quote_case = False
    for ta in topic_names:
        for tb in topic_names:
            if ta == tb:
                continue
            for a in by_topic[ta]:
                for b in by_topic[tb]:
                    if split_by_doc[a["doc_id"]] == split_by_doc[b["doc_id"]]:
                        btxt = f"{b.get('title','')} {b.get('text','')} {quote}"
                        atxt = f"{a.get('title','')} {a.get('text','')}"
                        a2 = dict(a)
                        a2["text"] = atxt + " " + quote
                        if add_pair(a2, btxt, False, "shared_quote_negative", b_source=b, note="shared quotation must not turn thematic texts into duplicates"):
                            found_quote_case = True
                            break
                if found_quote_case:
                    break
            if found_quote_case:
                break
        if found_quote_case:
            break

    # pad to exactly >=60 if some category was sparse
    attempts = 0
    while len(pairs) < 60 and attempts < 2000:
        attempts += 1
        d = rng.choice(docs)
        text = f"{d.get('title','')} {d.get('text','')}"
        add_pair(d, edit_text(text, rng), True, "edited_copy")

    write_jsonl(out_path, pairs[:60])
    return pairs[:60]


def classify_scores(rows: list[dict], score_key: str, threshold: float) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for r in rows:
        score = r.get(score_key)
        pred = score is not None and score >= threshold
        gold = bool(r["duplicate"])
        if pred and gold: tp += 1
        elif pred and not gold: fp += 1
        elif not pred and gold: fn += 1
    return tp, fp, fn


def best_threshold(rows: list[dict], score_key: str) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for i in range(10, 20):
        t = i / 20  # 0.50..0.95
        p, r, f1 = prf(*classify_scores(rows, score_key, t))
        if (f1, -t) > (best_f1, -best_t):
            best_t, best_f1 = t, f1
    return best_t, best_f1


def run(corpus_path: str, pairs_path: str, out_dir: str, seed: int = 42) -> dict:
    corpus = read_jsonl(corpus_path)
    ensure_corpus(corpus, 100)
    if not Path(pairs_path).exists():
        make_pairs(corpus, pairs_path, seed)
    pairs = read_jsonl(pairs_path)
    if len(pairs) < 60:
        raise ValueError("Task 07 requires at least 60 labeled pairs")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    measurements = []
    summaries = []
    for k in (2, 3, 5):
        for num_hashes in (64, 128):
            rows = []
            jac_times, mh_times = [], []
            errors = []
            for pair in pairs:
                sa = word_shingles(pair["text_a"], k)
                sb = word_shingles(pair["text_b"], k)
                t0 = time.perf_counter(); exact = jaccard(sa, sb); jac_times.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                siga = minhash_signature(sa, num_hashes, seed)
                sigb = minhash_signature(sb, num_hashes, seed)
                approx = minhash_similarity(siga, sigb)
                mh_times.append(time.perf_counter() - t0)
                if exact is not None and approx is not None:
                    errors.append(abs(exact - approx))
                rows.append({**pair, "jaccard": exact, "minhash": approx, "status": "ok" if sa and sb else "insufficient_tokens"})

            tune = [r for r in rows if r["split"] == "tune" and r["status"] == "ok"]
            test = [r for r in rows if r["split"] == "test" and r["status"] == "ok"]
            tj, _ = best_threshold(tune, "jaccard")
            tm, _ = best_threshold(tune, "minhash")
            pj, rj, f1j = prf(*classify_scores(test, "jaccard", tj))
            pm, rm, f1m = prf(*classify_scores(test, "minhash", tm))
            summary = {
                "shingle_k": k,
                "num_hashes": num_hashes,
                "jaccard_threshold": tj,
                "minhash_threshold": tm,
                "jaccard_precision": pj,
                "jaccard_recall": rj,
                "jaccard_f1": f1j,
                "minhash_precision": pm,
                "minhash_recall": rm,
                "minhash_f1": f1m,
                "mean_abs_approx_error": sum(errors)/len(errors) if errors else 0.0,
                "median_jaccard_ms": median(jac_times) * 1000,
                "median_minhash_ms": median(mh_times) * 1000,
                "insufficient_pairs": sum(1 for r in rows if r["status"] != "ok"),
            }
            summaries.append(summary)
            for r in rows:
                measurements.append({
                    "pair_id": r["pair_id"], "split": r["split"], "kind": r["kind"], "duplicate": r["duplicate"],
                    "shingle_k": k, "num_hashes": num_hashes, "jaccard": r["jaccard"], "minhash": r["minhash"], "status": r["status"]
                })

    shared_quote = [p for p in pairs if p["kind"] == "shared_quote_negative"]
    report = {
        "task": 7,
        "seed": seed,
        "pair_count": len(pairs),
        "configurations": summaries,
        "shared_quote_case": shared_quote[0] if shared_quote else None,
        "note": "Thresholds are chosen only on tune pairs; test pairs are used once for final metrics."
    }
    write_json(out / "summary.json", report)
    write_csv(out / "pair_scores.csv", measurements)
    return report


def main():
    ap = argparse.ArgumentParser(description="Task 07 - duplicate and near-duplicate detection")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--pairs", default="data/task07/pairs.jsonl")
    ap.add_argument("--out", default="results/task07")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prepare-pairs", action="store_true")
    args = ap.parse_args()
    corpus = read_jsonl(args.corpus)
    if args.prepare_pairs or not Path(args.pairs).exists():
        make_pairs(corpus, args.pairs, args.seed)
    print(json.dumps(run(args.corpus, args.pairs, args.out, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
