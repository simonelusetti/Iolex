"""Occurrence-specific, maximum-support deterministic patterns.

    python -m utils.occurrence_masks --output analysis/context_masks/fewnerd.sqlite

All original training words and full sentences are used. Offset 0 is always
visible; any subset of other *present* word positions is admissible, including
noncontiguous subsets. Match exact word identities at focal-relative offsets.
Keep ALL inclusion-minimal deterministic masks with globally maximum support,
not just the shortest ones. Never enumerate redundant deterministic supersets.

The self-contained SQLite file stores original sentence words, global token
occurrence IDs, labels, and results. Each result payload consists of n_extents
groups: uint32 support, uint32 mask count, support uint32 occurrence IDs, then
each mask as uint16 length followed by length int16 relative offsets. All values
are little-endian. Offset 0 is implicit. Use decode_result to read the payload.
Each group associates its masks with exactly their matching occurrences.

Commits every 500 results allow restart with the same command. Only exact and
proven-impossible results are skipped; interrupted/limited cases are retried.
Limits are resource safeguards, not restrictions on masks or sentence length.
"""
import argparse
import functools
import heapq
import json
import operator
import sqlite3
import struct
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.data import LABEL_DISPLAY_NAMES, resolve_dataset
from utils.context_masks import Limit, bits, check, enumerate_masks


def bitmap(array):
    return int.from_bytes(np.packbits(array, bitorder="little").tobytes(), "little")


def best_extents(matches, positive, n, deadline, max_states=20000):
    """Exact best-first search over distinct matching occurrence sets."""
    whole = (1 << n) - 1
    negative = whole ^ positive
    if not negative:
        return [whole], n, "exact"
    if functools.reduce(operator.and_, matches, whole) & negative:
        return [], 0, "impossible"
    features = sorted(set(matches) - {whole})
    queue = [(-positive.bit_count(), whole)]
    seen, winners, best = {whole}, set(), 0
    while queue:
        check(deadline)
        bound, extent = heapq.heappop(queue)
        if -bound < best:
            break
        wrong = extent & negative
        if not wrong:
            support = extent.bit_count()
            if support > best:
                winners, best = set(), support
            winners.add(extent)
            continue
        # A pure descendant must exclude this negative using at least one word.
        bad = wrong & -wrong
        for child in {extent & f for f in features if not f & bad}:
            upper = (child & positive).bit_count()
            if upper >= best and child not in seen:
                seen.add(child)
                heapq.heappush(queue, (-upper, child))
                if len(seen) > min(max_states, 128 * 2**20 // max(1, (n + 7) // 8)):
                    raise Limit("state_limit")
    return sorted(winners), best, "exact"


def context_matrix(occurrences, sentences):
    left = max(p for s, p, l, oid in occurrences)
    right = max(len(sentences[s]) - p - 1 for s, p, l, oid in occurrences)
    x = np.zeros((len(occurrences), left + right + 1), dtype=np.int32)
    for i, (sid, pos, _, _) in enumerate(occurrences):
        sentence = sentences[sid]
        x[i, left - pos:left - pos + len(sentence)] = sentence
    return x, left, np.array([l for s, p, l, oid in occurrences])


def matching_columns(occurrences, sentences, focal, cache):
    x, left, labels = cache
    sid, pos, label, _ = occurrences[focal]
    sentence = sentences[sid]
    offsets = np.delete(np.arange(len(sentence)) - pos, pos)
    equality = x[:, left - pos:left - pos + len(sentence)] == sentence
    packed = np.packbits(equality.T, axis=1, bitorder="little")
    matches = [int.from_bytes(row.tobytes(), "little") for j, row in enumerate(packed) if j != pos]
    return offsets, matches, bitmap(labels == label)


def solve(occurrences, sentences, focal, cache, seconds):
    offsets, matches, positive = matching_columns(occurrences, sentences, focal, cache)
    support, stage = None, "support"
    try:
        extents, support, status = best_extents(matches, positive, len(occurrences), time.monotonic() + seconds)
        if status == "impossible":
            return status, 0, 0, 0, b""
        stage, deadline = "masks", time.monotonic() + seconds
        negative = ((1 << len(occurrences)) - 1) ^ positive
        ids = np.array([oid for s, p, l, oid in occurrences], dtype="<u4")
        payload, n_masks = bytearray(), 0
        for extent in extents:
            check(deadline)
            closure = [j for j, match in enumerate(matches) if match & extent == extent]
            edges = set()
            for bad in bits(negative):
                check(deadline)
                edge = sum(1 << j for j in closure if not matches[j] & bad)
                assert edge
                edges.add(edge)
            masks, status, _ = enumerate_masks(sorted(edges, key=int.bit_count), deadline,
                                                10000, 100000, minimum=False)
            if status != "exact":
                raise Limit(status)
            selected = [b.bit_length() - 1 for b in bits(extent)]
            payload.extend(struct.pack("<II", support, len(masks)))
            payload.extend(ids[selected].tobytes())
            for mask in masks:
                positions = [int(offsets[b.bit_length() - 1]) for b in bits(mask)]
                payload.extend(struct.pack("<H", len(positions)))
                payload.extend(struct.pack(f"<{len(positions)}h", *positions))
            n_masks += len(masks)
        return "exact", support, n_masks, len(extents), bytes(payload)
    except Limit as exc:
        return f"{stage}_{exc}", support, None, None, None


def decode_result(payload):
    """Yield (matching global occurrence IDs, masks of relative offsets)."""
    pos = 0
    while pos < len(payload):
        support, count = struct.unpack_from("<II", payload, pos)
        pos += 8
        ids = struct.unpack_from(f"<{support}I", payload, pos)
        pos += 4 * support
        masks = []
        for _ in range(count):
            size, = struct.unpack_from("<H", payload, pos)
            pos += 2
            masks.append(struct.unpack_from(f"<{size}h", payload, pos))
            pos += 2 * size
        yield ids, masks


def load_corpus(connection):
    """Preserve original training sentence indices and flattened token IDs."""
    if connection.execute("SELECT count(*) FROM sentences").fetchone()[0] == 0:
        display = LABEL_DISPLAY_NAMES["fewnerd"]
        oid = 0
        for sid, example in enumerate(resolve_dataset("fewnerd")["train"]):
            tokens, labels = example["tokens"], example["labels"]
            assert len(tokens) == len(labels) and len(tokens) < 32768
            connection.execute("INSERT INTO sentences VALUES (?, ?)", (sid, json.dumps(tokens)))
            connection.executemany("INSERT INTO occurrences VALUES (?, ?, ?, ?, ?)",
                                   [(oid + pos, sid, pos, w, display.get(str(l), str(l)))
                                    for pos, (w, l) in enumerate(zip(tokens, labels))])
            oid += len(tokens)
        connection.commit()
    vocab, sentences, words = {}, [], defaultdict(list)
    for sid, encoded in connection.execute("SELECT * FROM sentences ORDER BY id"):
        assert sid == len(sentences)
        sentences.append(np.array([vocab.setdefault(w, len(vocab) + 1)
                                   for w in json.loads(encoded)], dtype=np.int32))
    for oid, sid, pos, word, label in connection.execute("SELECT * FROM occurrences ORDER BY id"):
        words[word].append((sid, pos, label, oid))
    return sentences, words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10, help="Budget per occurrence per search stage")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.output)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS sentences (id INTEGER PRIMARY KEY, words TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS occurrences
            (id INTEGER PRIMARY KEY, sentence INTEGER, position INTEGER, word TEXT, label TEXT);
        CREATE TABLE IF NOT EXISTS results
            (occurrence INTEGER PRIMARY KEY, status TEXT, support INTEGER,
             n_masks INTEGER, n_extents INTEGER, payload BLOB);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
    """)
    definition = "fewnerd/train/coarse; exact relative words; maximum support; all irredundant masks; v1"
    previous = connection.execute("SELECT value FROM settings WHERE key='definition'").fetchone()
    if previous and previous[0] != definition:
        raise ValueError("Existing results use a different experiment definition")
    connection.executemany("INSERT OR REPLACE INTO settings VALUES (?, ?)",
                           [("definition", definition), ("seconds_per_stage", str(args.seconds))])
    connection.commit()
    started = time.monotonic()
    sentences, words = load_corpus(connection)
    done = {r[0] for r in connection.execute("SELECT occurrence FROM results WHERE status IN ('exact','impossible')")}
    total, processed = sum(map(len, words.values())), 0
    counts = Counter()
    print(f"START {total} occurrences, {len(words)} types, {len(done)} already resolved; "
          f"{args.seconds}s/stage; output={args.output}", flush=True)
    for word, occurrences in words.items():
        pending = [i for i, v in enumerate(occurrences) if v[3] not in done]
        if not pending:
            continue
        lexical = len({v[2] for v in occurrences}) == 1
        if lexical:
            ids = np.array([v[3] for v in occurrences], dtype="<u4")
            payload = struct.pack("<II", len(ids), 1) + ids.tobytes() + struct.pack("<H", 0)
            result = ("exact", len(ids), 1, 1, payload)
        else:
            cache = context_matrix(occurrences, sentences)
        for focal in pending:
            if not lexical:
                result = solve(occurrences, sentences, focal, cache, args.seconds)
            oid = occurrences[focal][3]
            connection.execute("INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, ?, ?)", (oid, *result))
            counts[result[0]] += 1
            processed += 1
            if result[0] not in ("exact", "impossible"):
                print(f"LIMIT occurrence={oid} word={word!r} status={result[0]}", flush=True)
            if processed % 500 == 0:
                connection.commit()
            if processed % 5000 == 0:
                print(f"PROGRESS {processed+len(done)}/{total} elapsed={time.monotonic()-started:.1f}s "
                      f"{dict(counts)}", flush=True)
    connection.commit()
    counts = dict(connection.execute("SELECT status, count(*) FROM results GROUP BY status"))
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    print(f"DONE elapsed={time.monotonic()-started:.1f}s statuses={counts} "
          f"database_bytes={args.output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
