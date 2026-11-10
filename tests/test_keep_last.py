"""keep_last: the last word is a protected non-word, and `last` pooling reads
the true last token.  Run: python tests/test_keep_last.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sentence import masked_pool  # noqa: E402
from src.words import protect_last_word, word_valid  # noqa: E402


def test_protect_last_word_marks_every_piece_of_the_last_word():
    # word 2 has two pieces; the second row is shorter (padding -1 at the end)
    wids = torch.tensor([[0, 1, 2, 2, -1], [0, 0, 1, -1, -1]])
    assert protect_last_word(wids).tolist() == [[0, 1, -1, -1, -1], [0, 0, -1, -1, -1]]


def test_protect_last_word_skips_trailing_non_words():
    # E5-style: [CLS] prefix prefix w0 w1 w1 [SEP] -> w1 protected, specials untouched
    wids = torch.tensor([[-1, -1, -1, 0, 1, 1, -1]])
    assert protect_last_word(wids).tolist() == [[-1, -1, -1, 0, -1, -1, -1]]


def test_protect_last_word_is_not_idempotent_so_apply_once():
    # documented: a second call protects the next word back
    wids = torch.tensor([[0, 1, 2]])
    assert protect_last_word(protect_last_word(wids)).tolist() == [[0, -1, -1]]


def test_budget_counts_the_other_words():
    wids = torch.tensor([[0, 1, 2, 3, 3]])
    assert word_valid(protect_last_word(wids)).sum().item() == 3


def test_last_pooling_reads_the_true_last_token_even_if_masked():
    emb = torch.arange(8, dtype=torch.float32).view(2, 4, 1)          # value = flat position
    valid = torch.tensor([[1., 1., 1., 1.], [1., 1., 1., 0.]])       # row 1: 3 real tokens
    kept = torch.tensor([[1., 1., 0., 0.], [1., 0., 0., 0.]])        # the tail was dropped
    assert masked_pool(emb, kept, "last", valid_mask=valid).squeeze(-1).tolist() == [3.0, 6.0]


def test_last_pooling_without_valid_mask_uses_the_pool_mask():
    emb = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    assert masked_pool(emb, torch.tensor([[1., 1., 1., 0.]]), "last").item() == 2.0


if __name__ == "__main__":
    failed = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn(); print(f"  ok    {_name}")
            except AssertionError as exc:
                failed += 1; print(f"  FAIL  {_name}: {exc}")
    print("all passed" if not failed else f"{failed} failure(s)")
    sys.exit(1 if failed else 0)
