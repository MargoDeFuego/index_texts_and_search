from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Run tasks 2,3,4,7,11 sequentially")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--no-spacy", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    if not Path(args.corpus).exists():
        raise SystemExit(f"Corpus not found: {args.corpus}. Run a corpus preparation command first.")
    run([py, "scripts/prepare_exam_data.py", "--corpus", args.corpus])
    run([py, "-m", "indexing_exam.task02", "--corpus", args.corpus])
    run([py, "-m", "indexing_exam.task03", "--corpus", args.corpus])
    run([py, "-m", "indexing_exam.task04", "--corpus", args.corpus])
    run([py, "-m", "indexing_exam.task07", "--corpus", args.corpus, "--prepare-pairs"])
    cmd = [py, "-m", "indexing_exam.task11", "--corpus", args.corpus]
    if args.no_spacy:
        cmd.append("--no-spacy")
    run(cmd)
    print("\nDone. Results are in results/task02, task03, task04, task07, task11")


if __name__ == "__main__":
    main()
