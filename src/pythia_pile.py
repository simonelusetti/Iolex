"""Matched Pile windows for Pythia selector training and home-task evaluation.

Pythia-410m's home task is causal next-token prediction on the (non-deduped)
Pile, not LAMBADA.  Both splits below use the official ``pile_val_test``
repository, Pythia's own tokenizer, and the same deterministic window builder.
Validation windows train/select the selector; test windows are reserved for the
home-task evaluation.

The original model used 2048-token sequences.  Differentiating through several
masked Pythia forwards at that length does not fit the available GPU, so this
experiment uses fixed 128-token contexts.  This is the only intentional change
from the pretraining input protocol, and it is identical on both sides.
"""
from __future__ import annotations

import random
import re
from collections.abc import Iterable

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

PILE_REPO = "EleutherAI/pile_val_test"
CONTEXT_TOKENS = 128
TRAIN_WINDOWS = 6_000
EVAL_WINDOWS = 1_000
SHUFFLE_BUFFER = 10_000
WINDOW_SEED = 1729


def text_fragments(text: str) -> list[str]:
    """Whitespace-prefixed words whose concatenation reproduces ``text``.

    Pythia's byte-level tokenizer gives the same ids for this pre-tokenized
    representation as for the raw string, while the fragments provide stable
    word ids for whole-word selection.
    """
    return re.findall(r"\s*\S+", text)


def pile_windows(
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    count: int,
    *,
    context_tokens: int = CONTEXT_TOKENS,
    seed: int = WINDOW_SEED,
) -> list[dict]:
    """Build deterministic fixed-token windows from an official Pile split.

    One randomly positioned window is taken per shuffled document.  ``target_id``
    is retained for held-out home-task scoring; selector training intentionally
    ignores it and distils Pythia's full-context next-token distribution.
    """
    stream: Iterable[dict] = load_dataset(PILE_REPO, split=split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)
    rng = random.Random(seed)
    rows: list[dict] = []
    min_fragments = context_tokens + 1
    candidate_fragments = 2 * context_tokens + 32

    for example in stream:
        fragments = text_fragments(example["text"])
        if len(fragments) < min_fragments:
            continue
        # Leave at least context_tokens+1 whitespace words.  Since many words
        # split into several BPE tokens, this always leaves enough candidates
        # while avoiding an expensive tokenization of the whole document.
        last_start = len(fragments) - min_fragments
        start = rng.randint(0, last_start) if last_start else 0
        pieces = fragments[start:start + candidate_fragments]
        enc = tokenizer(pieces, is_split_into_words=True, add_special_tokens=False)
        if len(enc["input_ids"]) <= context_tokens:
            continue

        ids = enc["input_ids"][:context_tokens]
        word_ids = enc.word_ids()[:context_tokens]
        rows.append({
            "ids": ids,
            "attn_mask": [1] * context_tokens,
            "word_ids": word_ids,
            "tokens": tokenizer.convert_ids_to_tokens(ids),
            "target_id": int(enc["input_ids"][context_tokens]),
            "pile_set_name": str(example.get("meta", {}).get("pile_set_name", "unknown")),
        })
        if len(rows) >= count:
            break

    if len(rows) != count:
        raise RuntimeError(f"requested {count} Pile {split} windows, constructed {len(rows)}")
    return rows
