"""Exact empirical context masks, using unbounded focal-relative word offsets.

Run a frequency-stratified tractability pilot (not a dataset-wide estimate):
    python -m utils.context_masks --per-bin 3 --seconds 5
Use --per-bin 0 to search every ambiguous training word. Lexical determinism
is always counted over every training word, including singletons. No encoder,
tokenization, sentence truncation, or new dataset subsampling is used.

Opposite-label occurrence pairs form a hypergraph: each edge contains the
offsets where that pair differs. Deterministic masks are exactly its hitting
sets. A missing position has the reserved value 0. Globally constant offsets
cannot occur in any inclusion-minimal mask and are removed exactly.

Resource limits are computational safeguards, NOT a context window. Incomplete
searches have explicit statuses and never masquerade as nondeterminism.
"""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.data import LABEL_DISPLAY_NAMES, resolve_dataset
from tagger.tagging import TAGGED_DATASETS


BINS = [(2, 2), (3, 5), (6, 20), (21, 100), (101, 1000), (1001, float("inf"))]


class Limit(Exception):
    pass


def check(deadline):
    if time.monotonic() > deadline:
        raise Limit("time_limit")


def bits(mask):
    while mask:
        bit = mask & -mask
        yield bit
        mask ^= bit


def constraints(x, y, deadline, max_edges):
    """All discordant-pair constraints; deduplicate but never approximate."""
    edges = set()
    for i in range(len(y) - 1):
        check(deadline)
        other = np.flatnonzero(y[i + 1:] != y[i]) + i + 1
        # Bound temporary arrays even when the original context matrix is wide.
        chunk = max(1, 8_000_000 // max(1, x.shape[1]))
        for start in range(0, len(other), chunk):
            check(deadline)
            different = x[other[start:start + chunk]] != x[i]
            packed = np.packbits(different, axis=1, bitorder="little")
            edges.update(int.from_bytes(row.tobytes(), "little") for row in packed)
            if len(edges) > max_edges:
                raise Limit("edge_limit")
    if 0 in edges:
        raise ValueError("Conflicting identical contexts should be detected first")
    return sorted(edges, key=lambda e: (e.bit_count(), e))


def enumerate_masks(edges, deadline, max_masks, max_nodes, minimum=True):
    """Disjoint branch-and-bound; exact only if the entire search completes.

    Branch on an unhit edge. Branch j selects its j-th available coordinate
    and excludes its earlier coordinates, so no hitting set is visited twice.
    Every selected coordinate must have a private edge at a minimal solution.
    Once a coordinate has no possible private edge, adding more cannot fix it.
    """
    found, best, nodes = [], float("inf"), 0
    available = 0
    for edge in edges:
        available |= edge
    stack = [(0, available, edges)]
    pending_edges = len(edges)
    try:
        while stack:
            check(deadline)
            nodes += 1
            if nodes > max_nodes:
                raise Limit("node_limit")
            mask, available, unhit = stack.pop()
            pending_edges -= len(unhit)
            size = mask.bit_count()
            if minimum and size > best:
                continue
            private = {edge & mask for edge in edges if (edge & mask).bit_count() == 1}
            if len(private) != size:
                continue
            if not unhit:
                if minimum and size < best:
                    found, best = [], size
                found.append(mask)
                if len(found) > max_masks:
                    raise Limit("solution_limit")
                continue
            if minimum and size >= best:
                continue
            edge = min((e & available for e in unhit), key=int.bit_count)
            if not edge:
                continue
            for bit in bits(edge):
                available ^= bit
                remaining = [e for e in unhit if not e & bit]
                pending_edges += len(remaining)
                if pending_edges > 2_000_000:
                    raise Limit("search_memory_limit")
                stack.append((mask | bit, available, remaining))
                if len(stack) > max_nodes:
                    raise Limit("node_limit")
        return sorted(found), "exact", nodes
    except Limit as exc:
        # A discovered solution only gives an upper bound, not an exact k*.
        return [], str(exc), nodes


def mask_stats(masks, offsets, prefix):
    lower, upper = masks[0], 0
    for mask in masks:
        lower &= mask
        upper |= mask
    coordinates = lambda mask: [int(offsets[b.bit_length() - 1]) for b in bits(mask)]
    return {
        f"{prefix}_count": len(masks),
        f"{prefix}_L": json.dumps(coordinates(lower)),
        f"{prefix}_U": json.dumps(coordinates(upper)),
        f"{prefix}_L_size": lower.bit_count(),
        f"{prefix}_U_size": upper.bit_count(),
        f"{prefix}_A_size": (upper ^ lower).bit_count(),
        f"{prefix}_J": lower.bit_count() / upper.bit_count() if upper else "",
        f"{prefix}_masks": json.dumps([coordinates(m) for m in masks]),
    }


def entropy(labels):
    counts = np.array(list(Counter(labels).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def search_word(occurrences, sentences, args):
    started = time.monotonic()
    row = {"n": len(occurrences), "entropy_bits": entropy([v[2] for v in occurrences])}
    try:
        deadline = started + args.seconds
        # A full relative context identifies (complete sentence, focal position).
        # Sentences are deduplicated on exact original words at corpus loading.
        unique = {}
        for sid, pos, label in occurrences:
            check(deadline)
            key = (sid, pos)
            if key in unique and unique[key] != label:
                row.update(card_status="no_deterministic_mask", incl_status="no_deterministic_mask")
                row["total_seconds"] = time.monotonic() - started
                return row
            unique[key] = label
        row["unique_contexts"] = len(unique)
        left = max(pos for sid, pos in unique)
        right = max(len(sentences[sid]) - pos - 1 for sid, pos in unique)
        row["domain_offsets"] = left + right
        if len(unique) * (left + right + 1) * 4 > args.matrix_mib * 2**20:
            raise Limit("matrix_memory_limit")
        x = np.zeros((len(unique), left + right + 1), dtype=np.int32)
        for i, (sid, pos) in enumerate(unique):
            check(deadline)
            sentence = sentences[sid]
            x[i, left - pos:left - pos + len(sentence)] = sentence
        variable = np.any(x != x[0], axis=0)
        variable[left] = False  # focal word is known, never a selectable feature
        offsets = np.arange(-left, right + 1)[variable]
        x = x[:, variable]
        row["variable_offsets"] = len(offsets)
        edges = constraints(x, np.array(list(unique.values())), deadline, args.max_edges)
        row["edges"] = len(edges)
        row["constraint_seconds"] = time.monotonic() - started
    except Limit as exc:
        row.update(card_status=str(exc), incl_status="not_started")
        row["total_seconds"] = time.monotonic() - started
        return row
    for prefix, minimum in (("card", True), ("incl", False)):
        tick = time.monotonic()
        masks, status, nodes = enumerate_masks(
            edges, tick + args.seconds, args.max_masks, args.max_nodes, minimum)
        row.update({f"{prefix}_status": status, f"{prefix}_seconds": time.monotonic() - tick,
                    f"{prefix}_nodes": nodes})
        if status == "exact":
            row.update(mask_stats(masks, offsets, prefix))
            if minimum:
                row["k"] = masks[0].bit_count()
    row["total_seconds"] = time.monotonic() - started
    return row


def load_corpus(dataset):
    sentences, sequence_ids, vocab, words = [], {}, {}, defaultdict(list)
    display = LABEL_DISPLAY_NAMES[dataset]
    for example in resolve_dataset(dataset)["train"]:
        tokens, labels = example["tokens"], example["labels"]
        assert len(tokens) == len(labels)
        ids = np.array([vocab.setdefault(w, len(vocab) + 1) for w in tokens], dtype=np.int32)
        key = ids.tobytes()
        if key not in sequence_ids:
            sequence_ids[key] = len(sentences)
            sentences.append(ids)
        sid = sequence_ids[key]
        for pos, (word, label) in enumerate(zip(tokens, labels)):
            label = display.get(str(label), str(label))
            words[word].append((sid, pos, label))
    return sentences, words


def write_row(path, row, fields):
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=sorted(TAGGED_DATASETS), choices=sorted(TAGGED_DATASETS))
    parser.add_argument("--per-bin", type=int, default=3, help="Ambiguous types per frequency bin; 0 = all")
    parser.add_argument("--seconds", type=float, default=5, help="Budget EACH for constraints, cardinality, inclusion")
    parser.add_argument("--max-edges", type=int, default=100_000)
    parser.add_argument("--max-masks", type=int, default=10_000)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--matrix-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fields = ["dataset", "labels", "word", "frequency_bin", "n", "entropy_bits", "unique_contexts",
              "domain_offsets", "variable_offsets", "edges", "constraint_seconds", "k", "total_seconds"]
    for prefix in ("card", "incl"):
        fields += [f"{prefix}_{key}" for key in ("status", "seconds", "nodes", "count", "L", "U",
                                                 "L_size", "U_size", "A_size", "J", "masks")]
    print(f"Unbounded TRAIN contexts; seed={args.seed}; per-bin={args.per_bin}; budgets={args.seconds}s/stage; "
          f"edges={args.max_edges}; masks={args.max_masks}; nodes={args.max_nodes}; matrix={args.matrix_mib}MiB", flush=True)
    for dataset in args.datasets:
        print(f"Loading {dataset}", flush=True)
        sentences, original = load_corpus(dataset)
        has_boundaries = any(label.startswith(("B-", "I-")) for occ in original.values() for _, _, label in occ)
        for regime in (["original", "semantic"] if has_boundaries else ["original"]):
            words = original if regime == "original" else {
                w: [(s, p, l[2:] if l.startswith(("B-", "I-")) else l) for s, p, l in occ]
                for w, occ in original.items()}
            ambiguous = {w: occ for w, occ in words.items() if len({v[2] for v in occ}) > 1}
            total = sum(map(len, words.values()))
            census = dict(dataset=dataset, labels=regime, types=len(words), occurrences=total,
                          ambiguous_types=len(ambiguous),
                          lexical_types=len(words) - len(ambiguous),
                          lexical_occurrences=total - sum(map(len, ambiguous.values())))
            write_row(args.output / "census.csv", census, list(census))
            rng = np.random.default_rng(args.seed)
            selected = []
            for low, high in BINS:
                pool = sorted(w for w, occ in ambiguous.items() if low <= len(occ) <= high)
                chosen = rng.choice(len(pool), min(args.per_bin or len(pool), len(pool)), replace=False)
                selected.extend((pool[i], f"{low}-{high}") for i in chosen)
            statuses = Counter()
            for index, (word, frequency_bin) in enumerate(selected, 1):
                row = dict(dataset=dataset, labels=regime, word=word, frequency_bin=frequency_bin)
                row.update(search_word(words[word], sentences, args))
                write_row(args.output / "words.csv", row, fields)
                statuses[row["card_status"]] += 1
                print(f"{dataset}/{regime} {index}/{len(selected)} {word!r} n={row['n']} "
                      f"d={row.get('domain_offsets', '-')} k={row.get('k', '-')} "
                      f"card={row['card_status']} incl={row['incl_status']}", flush=True)
            print(f"SUMMARY {dataset}/{regime}: {dict(statuses)}; lexical types "
                  f"{census['lexical_types']}/{census['types']}; lexical occurrences "
                  f"{census['lexical_occurrences']}/{total}", flush=True)


if __name__ == "__main__":
    main()
