from __future__ import annotations

import argparse
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import requests

from indexing_exam.common import write_jsonl

TOPICS = {
    "археология": "archaeology history",
    "архитектура": "architecture history cultural heritage",
    "живопись": "painting art history",
    "история_литературы": "literary history literature history",
    "культурное_наследие": "cultural heritage conservation",
    "музеи": "museum history heritage",
    "реставрация": "restoration conservation heritage",
    "исторические_города": "historic city architecture heritage",
    "древнерусская_культура": "russian medieval culture history",
    "история_искусства": "art history painting"
}


def abstract_from_inverted(inv: dict | None) -> str:
    if not inv:
        return ""
    pairs = []
    for word, positions in inv.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort()
    return " ".join(word for _, word in pairs)


def fetch_topic(session: requests.Session, topic: str, search: str, target: int, language: str, mailto: str | None, api_key: str | None, fulltext: bool):
    url = "https://api.openalex.org/works"
    cursor = "*"
    seen = set()
    out = []
    while len(out) < target:
        filters = ["has_abstract:true", "type:article", f"language:{language}"]
        if fulltext:
            filters.append("has_content.grobid_xml:true")
        params = {
            "search": search,
            "filter": ",".join(filters),
            "per_page": min(100, max(25, target * 2)),
            "cursor": cursor,
            "select": "id,title,abstract_inverted_index,publication_year,doi,language,primary_location,content_urls",
        }
        if mailto:
            params["mailto"] = mailto
        if api_key:
            params["api_key"] = api_key
        r = session.get(url, params=params, timeout=45)
        r.raise_for_status()
        payload = r.json()
        for w in payload.get("results", []):
            oid = w.get("id")
            text = abstract_from_inverted(w.get("abstract_inverted_index"))
            if not oid or oid in seen or len(text.split()) < 35:
                continue
            seen.add(oid)
            loc = w.get("primary_location") or {}
            out.append({
                "openalex_id": oid,
                "title": w.get("title") or "Без названия",
                "text": text,
                "year": w.get("publication_year"),
                "doi": w.get("doi"),
                "language": w.get("language") or language,
                "source": (loc.get("landing_page_url") if isinstance(loc, dict) else None) or oid,
                "topic": topic,
                "content_urls": w.get("content_urls") or {},
                "acquisition": "OpenAlex API; scholarly article metadata/abstract"
            })
            if len(out) >= target:
                break
        cursor = (payload.get("meta") or {}).get("next_cursor")
        if not cursor or not payload.get("results"):
            break
        time.sleep(0.1)
    return out



def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def extract_grobid_text(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    bodies = [e for e in root.iter() if local_name(e.tag) == "body"]
    scope = bodies[0] if bodies else next((e for e in root.iter() if local_name(e.tag) == "text"), root)
    parts = []
    for e in scope.iter():
        if local_name(e.tag) in {"head", "p", "item", "quote"}:
            txt = " ".join("".join(e.itertext()).split())
            if txt:
                parts.append(txt)
    # Deduplicate nested repeats while preserving order.
    clean = []
    seen = set()
    for x in parts:
        if x not in seen:
            clean.append(x); seen.add(x)
    return "\n".join(clean)


def enrich_fulltext(session: requests.Session, docs: list[dict], api_key: str) -> None:
    for i, d in enumerate(docs, 1):
        url = (d.get("content_urls") or {}).get("grobid_xml")
        if not url:
            continue
        try:
            r = session.get(url, params={"api_key": api_key}, timeout=60)
            r.raise_for_status()
            text = extract_grobid_text(r.content)
            if len(text.split()) >= 100:
                d["text"] = text
                d["acquisition"] = "OpenAlex API + OpenAlex GROBID full text"
            else:
                d["acquisition"] += "; full-text parse too short, abstract kept"
        except Exception as exc:
            d["acquisition"] += f"; full-text fallback: {type(exc).__name__}"
        if i % 20 == 0:
            print(f"  full text processed: {i}/{len(docs)}")
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description="Download a Russian scholarly corpus from OpenAlex.")
    ap.add_argument("--limit", type=int, default=220, help="Total documents (recommended >=220).")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--out", default="data/corpus.jsonl")
    ap.add_argument("--fulltext", action="store_true", help="Fetch OpenAlex GROBID full text (requires OPENALEX_API_KEY).")
    args = ap.parse_args()

    per_topic = max(25, (args.limit + len(TOPICS) - 1) // len(TOPICS))
    mailto = os.environ.get("OPENALEX_MAILTO")
    api_key = os.environ.get("OPENALEX_API_KEY")
    if args.fulltext and not api_key:
        raise SystemExit("--fulltext requires OPENALEX_API_KEY (free key from OpenAlex settings).")
    session = requests.Session()
    session.headers["User-Agent"] = "indexing-exam-variant02/1.0"

    merged = {}
    counts = defaultdict(int)
    for topic, search in TOPICS.items():
        print(f"[OpenAlex] {topic} ...")
        try:
            works = fetch_topic(session, topic, search, per_topic, args.language, mailto, api_key, args.fulltext)
        except requests.RequestException as exc:
            print(f"  warning: {exc}")
            continue
        for w in works:
            if w["openalex_id"] not in merged:
                merged[w["openalex_id"]] = w
                counts[topic] += 1
        print(f"  collected {counts[topic]}")

    docs = list(merged.values())[: args.limit]
    if args.fulltext and api_key:
        print("[OpenAlex] downloading GROBID full text ...")
        enrich_fulltext(session, docs, api_key)
    for i, d in enumerate(docs, 1):
        d["doc_id"] = f"D{i:04d}"
        d.pop("content_urls", None)
    if len(docs) < args.limit:
        print(f"WARNING: requested {args.limit}, collected {len(docs)}. You can rerun or use another language/source.")
    write_jsonl(args.out, docs)
    print(f"Saved {len(docs)} documents -> {args.out}")


if __name__ == "__main__":
    main()
