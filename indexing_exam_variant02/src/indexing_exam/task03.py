from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .common import read_jsonl, tokenize, write_json

# Query grammar documented in README:
# OR lowest, AND middle, NOT higher, NEAR/k highest binary operator.
# NEAR/k is intentionally restricted to term-term operands for transparent positional semantics.


class QuerySyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class QToken:
    kind: str
    value: str


def lex(query: str) -> list[QToken]:
    out: list[QToken] = []
    i = 0
    n = len(query)
    while i < n:
        if query[i].isspace():
            i += 1
            continue
        ch = query[i]
        if ch == '(':
            out.append(QToken("LPAREN", ch)); i += 1; continue
        if ch == ')':
            out.append(QToken("RPAREN", ch)); i += 1; continue
        if ch == '"':
            j = i + 1
            buf = []
            while j < n and query[j] != '"':
                buf.append(query[j])
                j += 1
            if j >= n:
                raise QuerySyntaxError("Unclosed quote")
            phrase = "".join(buf).strip()
            if not phrase:
                raise QuerySyntaxError("Empty phrase")
            out.append(QToken("PHRASE", phrase))
            i = j + 1
            continue
        m = re.match(r"(?i)NEAR/(\d+)", query[i:])
        if m:
            k = int(m.group(1))
            if k <= 0:
                raise QuerySyntaxError("NEAR/k requires k > 0")
            out.append(QToken("NEAR", str(k)))
            i += m.end()
            continue
        m = re.match(r"[A-Za-zА-Яа-яЁё0-9./'-]+", query[i:])
        if not m:
            raise QuerySyntaxError(f"Unexpected character at position {i}: {query[i]!r}")
        word = m.group(0)
        upper = word.upper()
        if upper in {"AND", "OR", "NOT"}:
            out.append(QToken(upper, upper))
        else:
            # One lexical item must map to exactly one indexed token.
            parts = tokenize(word)
            if len(parts) != 1:
                raise QuerySyntaxError(f"Invalid term: {word!r}")
            out.append(QToken("TERM", parts[0]))
        i += m.end()
    return out


class Node:
    pass


@dataclass
class TermNode(Node):
    term: str


@dataclass
class PhraseNode(Node):
    terms: list[str]


@dataclass
class NotNode(Node):
    child: Node


@dataclass
class BinNode(Node):
    op: str
    left: Node
    right: Node
    k: int | None = None


class Parser:
    def __init__(self, tokens: list[QToken]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> QToken | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self, kind: str | None = None) -> QToken:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("Unexpected end of query")
        if kind and tok.kind != kind:
            raise QuerySyntaxError(f"Expected {kind}, got {tok.kind}")
        self.i += 1
        return tok

    def parse(self) -> Node:
        if not self.tokens:
            raise QuerySyntaxError("Empty query")
        node = self.parse_or()
        if self.peek() is not None:
            raise QuerySyntaxError(f"Unexpected token {self.peek()}")
        return node

    def parse_or(self) -> Node:
        node = self.parse_and()
        while self.peek() and self.peek().kind == "OR":
            self.take("OR")
            node = BinNode("OR", node, self.parse_and())
        return node

    def parse_and(self) -> Node:
        node = self.parse_not()
        while self.peek() and self.peek().kind == "AND":
            self.take("AND")
            node = BinNode("AND", node, self.parse_not())
        return node

    def parse_not(self) -> Node:
        if self.peek() and self.peek().kind == "NOT":
            self.take("NOT")
            return NotNode(self.parse_not())
        return self.parse_near()

    def parse_near(self) -> Node:
        node = self.parse_primary()
        while self.peek() and self.peek().kind == "NEAR":
            k = int(self.take("NEAR").value)
            right = self.parse_primary()
            if not isinstance(node, TermNode) or not isinstance(right, TermNode):
                raise QuerySyntaxError("NEAR/k currently supports TERM NEAR/k TERM")
            node = BinNode("NEAR", node, right, k)
        return node

    def parse_primary(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("Expected term, phrase or '('")
        if tok.kind == "TERM":
            return TermNode(self.take("TERM").value)
        if tok.kind == "PHRASE":
            terms = tokenize(self.take("PHRASE").value)
            if not terms:
                raise QuerySyntaxError("Phrase has no tokens")
            return PhraseNode(terms)
        if tok.kind == "LPAREN":
            self.take("LPAREN")
            node = self.parse_or()
            self.take("RPAREN")
            return node
        raise QuerySyntaxError(f"Unexpected token {tok.kind}")


def parse_query(query: str) -> Node:
    return Parser(lex(query)).parse()


class PositionalIndex:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.tokens_by_doc: dict[str, list[str]] = {}
        self.index: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    @property
    def universe(self) -> set[str]:
        return set(self.docs)

    def add_or_update(self, doc: dict) -> None:
        doc_id = str(doc["doc_id"])
        if doc_id in self.docs:
            self.delete(doc_id)
        self.docs[doc_id] = dict(doc)
        # Keep title and body in one positional stream but insert a non-indexed boundary
        # so an exact phrase cannot accidentally cross from the title into the body.
        tokens = tokenize(doc.get("title", "")) + ["__FIELD_BOUNDARY__"] + tokenize(doc.get("text", ""))
        self.tokens_by_doc[doc_id] = tokens
        for pos, term in enumerate(tokens):
            if term == "__FIELD_BOUNDARY__":
                continue
            self.index[term][doc_id].append(pos)

    def delete(self, doc_id: str) -> None:
        if doc_id not in self.docs:
            return
        for term in set(self.tokens_by_doc.get(doc_id, [])):
            postings = self.index.get(term)
            if postings and doc_id in postings:
                del postings[doc_id]
                if not postings:
                    del self.index[term]
        self.docs.pop(doc_id, None)
        self.tokens_by_doc.pop(doc_id, None)

    @classmethod
    def build(cls, docs: Iterable[dict]) -> "PositionalIndex":
        obj = cls()
        for d in docs:
            obj.add_or_update(d)
        return obj

    def phrase_docs(self, terms: list[str]) -> set[str]:
        if not terms:
            return set()
        candidate = set(self.index.get(terms[0], {}))
        for term in terms[1:]:
            candidate &= set(self.index.get(term, {}))
        result = set()
        for doc_id in candidate:
            starts = set(self.index[terms[0]][doc_id])
            for offset, term in enumerate(terms[1:], 1):
                posset = set(self.index[term][doc_id])
                starts = {s for s in starts if s + offset in posset}
                if not starts:
                    break
            if starts:
                result.add(doc_id)
        return result

    def near_docs(self, left: str, right: str, k: int) -> set[str]:
        candidate = set(self.index.get(left, {})) & set(self.index.get(right, {}))
        result = set()
        for doc_id in candidate:
            a = self.index[left][doc_id]
            b = self.index[right][doc_id]
            j = 0
            for pa in a:
                while j < len(b) and b[j] <= pa:
                    j += 1
                if j < len(b) and 0 < b[j] - pa <= k:
                    result.add(doc_id)
                    break
        return result

    def evaluate(self, node: Node) -> set[str]:
        if isinstance(node, TermNode):
            return set(self.index.get(node.term, {}))
        if isinstance(node, PhraseNode):
            return self.phrase_docs(node.terms)
        if isinstance(node, NotNode):
            return self.universe - self.evaluate(node.child)
        if isinstance(node, BinNode):
            if node.op == "AND":
                return self.evaluate(node.left) & self.evaluate(node.right)
            if node.op == "OR":
                return self.evaluate(node.left) | self.evaluate(node.right)
            if node.op == "NEAR":
                assert isinstance(node.left, TermNode) and isinstance(node.right, TermNode) and node.k is not None
                return self.near_docs(node.left.term, node.right.term, node.k)
        raise TypeError(node)

    def search(self, query: str) -> list[str]:
        return sorted(self.evaluate(parse_query(query)))

    def dump_positions(self, terms: list[str]) -> dict:
        return {t: {d: p for d, p in sorted(self.index.get(t, {}).items())} for t in terms}


def sequential_search(docs: list[dict], query: str) -> list[str]:
    # Independent correctness oracle: rebuild one-document positional structure per document.
    ast = parse_query(query)
    out = []
    for d in docs:
        idx = PositionalIndex.build([d])
        if idx.evaluate(ast):
            out.append(str(d["doc_id"]))
    return sorted(out)


def ast_to_dict(node: Node):
    if isinstance(node, TermNode):
        return {"TERM": node.term}
    if isinstance(node, PhraseNode):
        return {"PHRASE": node.terms}
    if isinstance(node, NotNode):
        return {"NOT": ast_to_dict(node.child)}
    if isinstance(node, BinNode):
        d = {node.op: [ast_to_dict(node.left), ast_to_dict(node.right)]}
        if node.k is not None:
            d["k"] = node.k
        return d
    raise TypeError(node)


def run(corpus_path: str, small_path: str, checks_path: str, out_dir: str, demo_query: str | None = None) -> dict:
    corpus = read_jsonl(corpus_path)
    if len(corpus) < 120:
        raise ValueError("Task 03 requires at least 120 documents")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    idx = PositionalIndex.build(corpus)
    sample_terms = ["музей", "архитектура", "1917", "пушкин"]
    write_json(out / "positions_sample.json", idx.dump_positions(sample_terms))

    small = read_jsonl(small_path)
    checks = read_jsonl(checks_path)
    sidx = PositionalIndex.build(small)
    check_rows = []
    passed = 0
    for c in checks:
        try:
            got = sidx.search(c["query"])
            seq = sequential_search(small, c["query"])
            expected = sorted(c["expected"])
            ok = got == expected and got == seq
            err = ""
        except Exception as exc:
            got, seq, expected, ok, err = [], [], sorted(c["expected"]), False, str(exc)
        passed += int(ok)
        check_rows.append({"id": c["id"], "query": c["query"], "expected": expected, "index": got, "sequential": seq, "ok": ok, "error": err})

    # Update / delete / add tests and comparison with full rebuild.
    update_doc = {"doc_id": "S3", "title": "Музей", "text": "Музей истории живописи хранит каталог 1905 года."}
    sidx.add_or_update(update_doc)
    small2 = [update_doc if d["doc_id"] == "S3" else d for d in small]
    rebuilt = PositionalIndex.build(small2)
    update_ok = sidx.search("архитектуры") == rebuilt.search("архитектуры") and sidx.search("1905") == rebuilt.search("1905")
    sidx.delete("S4")
    small3 = [d for d in small2 if d["doc_id"] != "S4"]
    rebuilt2 = PositionalIndex.build(small3)
    delete_ok = sidx.search("живопись") == rebuilt2.search("живопись")
    added = {"doc_id": "S9", "title": "Дата", "text": "Название Царьград встречается рядом с датой 1453."}
    sidx.add_or_update(added)
    small4 = small3 + [added]
    rebuilt3 = PositionalIndex.build(small4)
    add_ok = sidx.search("Царьград AND 1453") == rebuilt3.search("Царьград AND 1453")

    syntax_cases = {}
    for q in ["", "(музей AND", "музей AND OR обзор", '""']:
        try:
            parse_query(q)
            syntax_cases[q] = "unexpectedly accepted"
        except QuerySyntaxError as exc:
            syntax_cases[q] = f"rejected: {exc}"

    result = {
        "task": 3,
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": check_rows,
        "update_matches_rebuild": update_ok,
        "delete_matches_rebuild": delete_ok,
        "add_matches_rebuild": add_ok,
        "syntax_edge_cases": syntax_cases,
        "date_near_name_example": {
            "query": "Царьград AND 1453",
            "result": sidx.search("Царьград AND 1453"),
        }
    }
    if demo_query:
        ast = parse_query(demo_query)
        result["demo_query"] = {"query": demo_query, "ast": ast_to_dict(ast), "results": idx.search(demo_query)[:50]}
    write_json(out / "summary.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description="Task 03 - positional index and query language")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--small", default="data/task03/small_corpus.jsonl")
    ap.add_argument("--checks", default="data/task03/checks.jsonl")
    ap.add_argument("--out", default="results/task03")
    ap.add_argument("--demo-query", default='"культурное наследие" OR (археология AND NOT реставрация)')
    args = ap.parse_args()
    print(json.dumps(run(args.corpus, args.small, args.checks, args.out, args.demo_query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
