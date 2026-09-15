"""Distance-weighted occurrence masks on Few-NERD's training split.

For focal occurrence i, another occurrence j of the same word remains a
contextual match under M while

    sum(1 / abs(offset) for offset in M if word_i[offset] != word_j[offset]) < 1.

A mask is sufficient when every remaining match has the focal label. Choose
the fewest positions, then the smallest sum of absolute offsets, then greatest
support; save every complete tie. Offset zero is implicit, positions may be
noncontiguous, and only positions present in the focal sentence are candidates.

    python -m utils.weighted_occurrence_masks \
        --output analysis/context_masks/fewnerd_weighted_occurrences.sqlite

CUDA evaluates candidate masks against occurrences. SQLite is restartable and
uses the payload format documented by utils.occurrence_masks.decode_result.
Resource-limit statuses are explicit and are never treated as impossibility.
"""
import argparse
import json
import multiprocessing as mp
import os
import sqlite3
import struct
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack

from utils.context_masks import Limit, check
from utils.occurrence_masks import context_matrix, load_corpus


THRESHOLD = 1.0
TOLERANCE = 1e-10
WORKER_SENTENCES = None
WORKER_WORDS = None


def optimize(cost, matrix, lower, upper, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise Limit("time_limit")
    result = milp(cost, integrality=np.ones(len(cost)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={"time_limit": remaining, "mip_rel_gap": 0.0})
    if result.success:
        return result
    if result.status == 2:
        return None
    raise Limit("optimizer_limit")


def solve(occurrences, sentences, focal, cache, device, seconds, max_masks):
    """CUDA builds evidence; exact mixed-integer optimization finds all ties."""
    started = time.monotonic()
    deadline = started + seconds
    x, left, labels = cache
    sid, pos, label, _ = occurrences[focal]
    sentence = sentences[sid]
    offsets = np.delete(np.arange(len(sentence), dtype=np.int16) - pos, pos)
    columns = np.delete(np.arange(left - pos, left - pos + len(sentence)), pos)
    negative_np = labels != label
    positive_np = ~negative_np
    if not negative_np.any():
        raise AssertionError("Lexically deterministic types must bypass solve")

    weights_np = 1.0 / np.abs(offsets).astype(np.float64)
    if device.type == "cuda":
        context = torch.as_tensor(x[:, columns], device=device)
        target = torch.as_tensor(np.delete(sentence, pos), device=device)
        mismatch = (context != target).T
        weights = torch.as_tensor(weights_np, device=device)
        negative = torch.as_tensor(negative_np, device=device)
        full = (mismatch * weights[:, None]).sum(dim=0)
        if torch.any(full[negative] < THRESHOLD - TOLERANCE).item():
            return "impossible", 0, 0, 0, b"", 0, time.monotonic() - started
        mismatch_np = mismatch.cpu().numpy()
    else:
        mismatch_np = (x[:, columns] != np.delete(sentence, pos)).T
        full = (mismatch_np * weights_np[:, None]).sum(axis=0, dtype=np.float64)
        if np.any(full[negative_np] < THRESHOLD - TOLERANCE):
            return "impossible", 0, 0, 0, b"", 0, time.monotonic() - started
    p = len(offsets)
    # One binary variable per visible relative position. Each wrong-label
    # occurrence supplies a weighted covering constraint >= 1.
    evidence = csr_matrix((mismatch_np[:, negative_np].T * weights_np).astype(np.float64))
    solver_calls = 0

    try:
        lower = np.ones(evidence.shape[0])
        upper = np.full(evidence.shape[0], np.inf)
        result = optimize(np.ones(p), evidence, lower, upper, deadline)
        solver_calls += 1
        assert result is not None  # full-mask CUDA check proved feasibility
        cardinality = int(round(result.fun))

        cardinality_row = csr_matrix(np.ones((1, p)))
        matrix = vstack([evidence, cardinality_row], format="csr")
        lower2 = np.r_[lower, cardinality]
        upper2 = np.r_[upper, cardinality]
        result = optimize(np.abs(offsets).astype(float), matrix, lower2, upper2, deadline)
        solver_calls += 1
        assert result is not None
        distance = int(round(result.fun))

        # Enumerate every mask tied on cardinality and distance using no-good
        # cuts. With cardinality fixed, sum(selected variables) <= k-1 excludes
        # exactly the current solution without excluding any other solution.
        base = vstack([matrix, csr_matrix(np.abs(offsets)[None, :])], format="csr")
        base_lower = np.r_[lower2, distance]
        base_upper = np.r_[upper2, distance]
        cuts = []
        best_support, winners = -1, []
        ids = np.asarray([oid for _, _, _, oid in occurrences], dtype="<u4")
        while True:
            check(deadline)
            constraints = vstack([base, *cuts], format="csr") if cuts else base
            cut_count = len(cuts)
            result = optimize(np.zeros(p), constraints,
                              np.r_[base_lower, np.full(cut_count, -np.inf)],
                              np.r_[base_upper, np.full(cut_count, cardinality - 1)], deadline)
            solver_calls += 1
            if result is None:
                break
            selected = np.flatnonzero(result.x > 0.5)
            if len(selected) != cardinality:
                raise Limit("numeric_failure")
            cuts.append(csr_matrix((np.ones(cardinality), (np.zeros(cardinality), selected)),
                                   shape=(1, p)))
            if len(cuts) > max_masks:
                raise Limit("solution_limit")
            scores = (mismatch_np[selected] * weights_np[selected, None]).sum(axis=0,
                                                                              dtype=np.float64)
            if np.any(scores[negative_np] < THRESHOLD - TOLERANCE):
                continue
            matches = positive_np & (scores < THRESHOLD - TOLERANCE)
            support = int(np.count_nonzero(matches))
            mask = tuple(int(offsets[j]) for j in selected)
            members = tuple(ids[matches].tolist())
            if support > best_support:
                best_support, winners = support, []
            if support == best_support:
                winners.append((mask, members))

        grouped = defaultdict(list)
        for mask, members in winners:
            grouped[members].append(mask)
        payload = bytearray()
        for members, masks in grouped.items():
            payload.extend(struct.pack("<II", len(members), len(masks)))
            payload.extend(np.asarray(members, dtype="<u4").tobytes())
            for mask in masks:
                payload.extend(struct.pack("<H", len(mask)))
                payload.extend(struct.pack(f"<{len(mask)}h", *mask))
        return ("exact", best_support, len(winners), len(grouped), bytes(payload),
                solver_calls, time.monotonic() - started)
    except Limit as exc:
        return str(exc), None, None, None, None, solver_calls, time.monotonic() - started


def solve_chunk(task):
    """Fork worker: reuse one aligned word matrix for a chunk of occurrences."""
    word, focals, seconds, max_masks = task
    occurrences = WORKER_WORDS[word]
    cache = context_matrix(occurrences, WORKER_SENTENCES)
    device = torch.device("cpu")
    return [(occurrences[focal][3], *solve(occurrences, WORKER_SENTENCES, focal, cache,
                                           device, seconds, max_masks))
            for focal in focals]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Occurrences of one word solved per worker task")
    parser.add_argument("--seconds", type=float, default=60,
                        help="Maximum seconds per unresolved occurrence")
    parser.add_argument("--max-masks", type=int, default=10_000)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "cuda" and args.workers != 1:
        raise ValueError("CUDA mode uses one process; use CPU for parallel exact optimization")
    device = torch.device(args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.output)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS sentences (id INTEGER PRIMARY KEY, words TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS occurrences
            (id INTEGER PRIMARY KEY, sentence INTEGER, position INTEGER, word TEXT, label TEXT);
        CREATE TABLE IF NOT EXISTS results
            (occurrence INTEGER PRIMARY KEY, status TEXT, support INTEGER,
             n_masks INTEGER, n_extents INTEGER, payload BLOB,
             combinations INTEGER, seconds REAL);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
    """)
    definition = ("fewnerd/train/coarse; exact relative words; evidence=sum(difference/abs(offset)); "
                  "threshold=1; minimize cardinality,distance; maximize support; all ties; v2")
    previous = connection.execute("SELECT value FROM settings WHERE key='definition'").fetchone()
    if previous and previous[0] != definition:
        raise ValueError("Existing results use a different experiment definition")
    connection.executemany("INSERT OR REPLACE INTO settings VALUES (?, ?)", [
        ("definition", definition), ("device", str(device)),
        ("cuda_name", torch.cuda.get_device_name(device) if device.type == "cuda" else ""),
        ("workers", str(args.workers)),
        ("seconds_per_occurrence", str(args.seconds)),
        ("optimizer", "scipy.optimize.milp/HiGHS"),
        ("max_masks", str(args.max_masks)),
    ])
    connection.commit()

    started = time.monotonic()
    sentences, words = load_corpus(connection)
    done = {row[0] for row in connection.execute(
        "SELECT occurrence FROM results WHERE status IN ('exact','impossible')")}
    total, processed, counts = sum(map(len, words.values())), 0, Counter()
    device_description = (f"{device} ({torch.cuda.get_device_name(device)})"
                          if device.type == "cuda" else f"cpu x {args.workers} workers")
    print(f"START {total} occurrences, {len(words)} types, {len(done)} already resolved; "
          f"device={device_description}; output={args.output}", flush=True)

    # Empty masks need no optimizer. Write these in the parent while building
    # dynamically scheduled chunks for ambiguous word types.
    tasks = []
    for word, occurrences in words.items():
        pending = [i for i, occurrence in enumerate(occurrences) if occurrence[3] not in done]
        if not pending:
            continue
        lexical = len({occurrence[2] for occurrence in occurrences}) == 1
        if lexical:
            ids = np.asarray([occurrence[3] for occurrence in occurrences], dtype="<u4")
            payload = struct.pack("<II", len(ids), 1) + ids.tobytes() + struct.pack("<H", 0)
            result = ("exact", len(ids), 1, 1, payload, 0, 0.0)
            for focal in pending:
                oid = occurrences[focal][3]
                connection.execute("INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                   (oid, *result))
                counts["exact"] += 1
                processed += 1
        else:
            tasks.extend((word, pending[start:start + args.chunk_size], args.seconds, args.max_masks)
                         for start in range(0, len(pending), args.chunk_size))
    connection.commit()

    global WORKER_SENTENCES, WORKER_WORDS
    WORKER_SENTENCES, WORKER_WORDS = sentences, words
    context = mp.get_context("fork")
    with context.Pool(args.workers) as pool:
        for batch in pool.imap_unordered(solve_chunk, tasks, chunksize=1):
            for oid, *result in batch:
                connection.execute("INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                   (oid, *result))
                counts[result[0]] += 1
                processed += 1
                if result[0] not in ("exact", "impossible"):
                    print(f"LIMIT occurrence={oid} status={result[0]} "
                          f"solver_calls={result[-2]} seconds={result[-1]:.2f}", flush=True)
            connection.commit()
            if processed // 5000 != (processed - len(batch)) // 5000:
                print(f"PROGRESS {processed + len(done)}/{total} elapsed={time.monotonic()-started:.1f}s "
                      f"{dict(counts)}", flush=True)
    connection.commit()
    statuses = dict(connection.execute("SELECT status, count(*) FROM results GROUP BY status"))
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    print(f"DONE elapsed={time.monotonic()-started:.1f}s statuses={statuses} "
          f"database_bytes={args.output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
