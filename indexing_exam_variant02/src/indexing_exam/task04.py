from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .common import ensure_corpus, median, read_jsonl, tokenize, write_csv, write_json


def delta_encode(ids: list[int]) -> list[int]:
    ids = sorted(set(ids))
    if not ids:
        return []
    out = [ids[0]]
    out.extend(ids[i] - ids[i - 1] for i in range(1, len(ids)))
    return out


def delta_decode(deltas: list[int]) -> list[int]:
    out = []
    acc = 0
    for d in deltas:
        if d < 0:
            raise ValueError("Negative delta")
        acc += d
        out.append(acc)
    return out


def vb_encode_number(n: int) -> bytes:
    if n < 0:
        raise ValueError("Variable Byte supports non-negative integers")
    groups = [n % 128]
    n //= 128
    while n:
        groups.append(n % 128)
        n //= 128
    groups.reverse()  # most significant 7-bit group first
    groups[-1] |= 0x80  # MSB=1 on the final byte
    return bytes(groups)


def vb_encode(numbers: list[int]) -> bytes:
    return b"".join(vb_encode_number(n) for n in numbers)


def vb_decode(data: bytes) -> list[int]:
    out = []
    n = 0
    in_number = False
    for b in data:
        in_number = True
        if b & 0x80:
            n = 128 * n + (b & 0x7F)
            out.append(n)
            n = 0
            in_number = False
        else:
            n = 128 * n + b
    if in_number:
        raise ValueError("Truncated Variable Byte sequence")
    return out


def vb_decode_one(data: bytes, offset: int) -> tuple[int, int]:
    if offset >= len(data):
        raise EOFError
    n = 0
    while offset < len(data):
        b = data[offset]
        offset += 1
        if b & 0x80:
            return 128 * n + (b & 0x7F), offset
        n = 128 * n + b
    raise ValueError("Truncated Variable Byte sequence")


def compress_postings(ids: list[int]) -> bytes:
    return vb_encode(delta_encode(ids))


def decompress_postings(data: bytes) -> list[int]:
    return delta_decode(vb_decode(data))


def intersect_two_pointer(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def build_skip_map(ids: list[int]) -> dict[int, int]:
    n = len(ids)
    if n < 4:
        return {}
    step = max(2, int(math.sqrt(n)))
    skips = {}
    i = 0
    while i + step < n:
        skips[i] = i + step
        i += step
    return skips


def intersect_with_skips(a: list[int], b: list[int]) -> list[int]:
    sa, sb = build_skip_map(a), build_skip_map(b)
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            ni = sa.get(i)
            if ni is not None and a[ni] <= b[j]:
                i = ni
            else:
                i += 1
        else:
            nj = sb.get(j)
            if nj is not None and b[nj] <= a[i]:
                j = nj
            else:
                j += 1
    return out


@dataclass
class Checkpoint:
    index: int
    doc_id: int
    next_offset: int


class CompressedPosting:
    """Delta+VB list with restart checkpoints.

    Each checkpoint stores an absolute doc_id and the byte offset where the NEXT delta starts.
    Therefore seek_geq() can resume without decoding from byte zero.
    """
    def __init__(self, ids: list[int], checkpoint_step: int | None = None):
        self.ids = sorted(set(ids))
        self.data = b""
        self.checkpoints: list[Checkpoint] = []
        if not self.ids:
            return
        step = checkpoint_step or max(1, int(math.sqrt(len(self.ids))))
        chunks = []
        offset = 0
        prev = 0
        for i, doc_id in enumerate(self.ids):
            delta = doc_id if i == 0 else doc_id - prev
            enc = vb_encode_number(delta)
            chunks.append(enc)
            offset += len(enc)
            if i % step == 0:
                self.checkpoints.append(Checkpoint(i, doc_id, offset))
            prev = doc_id
        self.data = b"".join(chunks)
        if self.checkpoints[-1].index != len(self.ids) - 1:
            # Last checkpoint is useful for seek near the end.
            # Find offset after the last encoded number by decoding once; size is negligible and deterministic.
            self.checkpoints.append(Checkpoint(len(self.ids) - 1, self.ids[-1], len(self.data)))
        self._checkpoint_docids = [c.doc_id for c in self.checkpoints]

    @property
    def storage_bytes(self) -> int:
        # 3 uint32 per checkpoint: index, absolute doc_id, byte offset.
        return len(self.data) + 12 * len(self.checkpoints)

    def seek_geq(self, target: int) -> int | None:
        if not self.ids:
            return None
        if target <= self.ids[0]:
            return self.ids[0]
        pos = bisect.bisect_right(self._checkpoint_docids, target) - 1
        pos = max(0, pos)
        cp = self.checkpoints[pos]
        current = cp.doc_id
        if current >= target:
            return current
        offset = cp.next_offset
        idx = cp.index
        while idx + 1 < len(self.ids) and offset < len(self.data):
            delta, offset = vb_decode_one(self.data, offset)
            current += delta
            idx += 1
            if current >= target:
                return current
        return None


def intersect_compressed(a: CompressedPosting, b: CompressedPosting) -> list[int]:
    # Iterate the smaller logical list and use checkpointed seeking in the other compressed list.
    if len(a.ids) > len(b.ids):
        a, b = b, a
    out = []
    for x in a.ids:
        y = b.seek_geq(x)
        if y == x:
            out.append(x)
    return out


def build_postings(corpus: list[dict]) -> tuple[dict[str, list[int]], dict[str, int]]:
    doc_num = {str(d["doc_id"]): i + 1 for i, d in enumerate(corpus)}
    postings = defaultdict(set)
    for d in corpus:
        did = doc_num[str(d["doc_id"])]
        for term in set(tokenize(f"{d.get('title','')} {d.get('text','')}")):
            postings[term].add(did)
    return {t: sorted(v) for t, v in postings.items()}, doc_num


def synthetic_pair(n: int, density: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    universe = max(n * 20, 1000)
    a = sorted(rng.sample(range(1, universe + 1), n))
    common_n = max(0, min(n, int(n * density)))
    common = set(rng.sample(a, common_n)) if common_n else set()
    b = set(common)
    while len(b) < n:
        b.add(rng.randint(1, universe))
    return a, sorted(b)


def timed(fn, *args, repeats: int = 5):
    values = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        values.append(time.perf_counter() - t0)
    return result, median(values)


def run(corpus_path: str, out_dir: str, repeats: int = 5) -> dict:
    corpus = read_jsonl(corpus_path)
    ensure_corpus(corpus, 200)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    postings, doc_num = build_postings(corpus)

    # Codec correctness including required boundary values.
    codec_cases = [[], [0], [127], [128], [3, 10, 11, 25], [1, 128, 16384, 999999]]
    codec_results = []
    for ids in codec_cases:
        sorted_ids = sorted(set(ids))
        enc = compress_postings(sorted_ids)
        dec = decompress_postings(enc)
        codec_results.append({"input": sorted_ids, "bytes_hex": enc.hex(), "decoded": dec, "ok": dec == sorted_ids})
    truncated_detected = False
    try:
        vb_decode(bytes([1]))  # no final byte with MSB=1
    except ValueError:
        truncated_detected = True

    total_ids = sum(len(v) for v in postings.values())
    raw_bytes = total_ids * 4  # explicit uint32 baseline
    compressed_bytes = 0
    compressed_plus_cp = 0
    for ids in postings.values():
        cp = CompressedPosting(ids)
        compressed_bytes += len(cp.data)
        compressed_plus_cp += cp.storage_bytes

    rows = []
    for n in (1_000, 10_000, 100_000):
        for density in (0.001, 0.01, 0.10):
            a, b = synthetic_pair(n, density, seed=42 + n + int(density * 1000))
            expected, t_plain = timed(intersect_two_pointer, a, b, repeats=repeats)
            skip_res, t_skip = timed(intersect_with_skips, a, b, repeats=repeats)
            ca, cb = CompressedPosting(a), CompressedPosting(b)
            comp_res, t_comp = timed(intersect_compressed, ca, cb, repeats=repeats)
            # Full path: read bytes already in memory + decode + two-pointer.
            def full_decode_intersection():
                da = decompress_postings(ca.data)
                db = decompress_postings(cb.data)
                return intersect_two_pointer(da, db)
            full_res, t_full = timed(full_decode_intersection, repeats=repeats)
            ok = expected == skip_res == comp_res == full_res
            rows.append({
                "n_each": n,
                "target_intersection_density": density,
                "actual_intersection": len(expected),
                "plain_ms_median": round(t_plain * 1000, 4),
                "skip_ms_median": round(t_skip * 1000, 4),
                "compressed_checkpoint_ms_median": round(t_comp * 1000, 4),
                "full_decode_plus_intersect_ms_median": round(t_full * 1000, 4),
                "raw_uint32_bytes": 8 * n,
                "compressed_bytes_plus_checkpoints": ca.storage_bytes + cb.storage_bytes,
                "all_results_equal": ok,
            })

    # Variant-specific example: repeated names and dates.
    repeated_example = {}
    for term in ("пушкин", "1917", "1812"):
        ids = postings.get(term, [])
        repeated_example[term] = {
            "df": len(ids),
            "first_ids": ids[:20],
            "delta_first": delta_encode(ids)[:20],
            "compressed_hex_prefix": compress_postings(ids)[:32].hex(),
        }

    write_csv(out / "measurements.csv", rows)
    summary = {
        "task": 4,
        "uint32_baseline_bytes": raw_bytes,
        "vb_compressed_bytes": compressed_bytes,
        "vb_plus_checkpoint_bytes": compressed_plus_cp,
        "compression_ratio_raw_over_vb": raw_bytes / compressed_bytes if compressed_bytes else 0.0,
        "compression_ratio_raw_over_vb_plus_checkpoints": raw_bytes / compressed_plus_cp if compressed_plus_cp else 0.0,
        "codec_cases": codec_results,
        "truncated_sequence_detected": truncated_detected,
        "synthetic_measurements": rows,
        "variant_repeated_names_dates": repeated_example,
        "note": "Baseline stores every posting as uint32 (4 bytes). Checkpoint cost is included as 12 bytes per restart point."
    }
    write_json(out / "summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Task 04 - index compression and acceleration")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--out", default="results/task04")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    result = run(args.corpus, args.out, args.repeats)
    print(json.dumps({k: v for k, v in result.items() if k != "synthetic_measurements"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
