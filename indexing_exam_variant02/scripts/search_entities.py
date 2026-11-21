from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Search extracted entity catalog by type and name substring")
    ap.add_argument("--catalog", default="results/task11/entity_catalog.json")
    ap.add_argument("--type", choices=["PERSON", "ORG", "LOC"], required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    q = args.name.lower()
    hits = []
    for name, doc_ids in data.get(args.type, {}).items():
        if q in name.lower():
            hits.append({"name": name, "type": args.type, "doc_ids": doc_ids})
    print(json.dumps(hits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
