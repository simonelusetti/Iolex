"""Matched Pile window construction for Pythia selector training/evaluation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pythia_pile import text_fragments  # noqa: E402
from src.sentence import resolve_tokenizer  # noqa: E402


def test_fragments_preserve_raw_text_except_irrelevant_trailing_space():
    text = "  First\nline, with\todd  spacing!  "
    assert "".join(text_fragments(text)) == text.rstrip()


def test_fragment_tokenization_matches_raw_pythia_tokenization():
    tokenizer = resolve_tokenizer("llm")
    examples = [
        "The Pile keeps punctuation, CAPS, and spaces exactly.",
        "\nA second line has naïve Unicode and  two spaces.",
        "code(x += 1);\t# byte-level BPE",
    ]
    for text in examples:
        pieces = text_fragments(text)
        split = tokenizer(pieces, is_split_into_words=True, add_special_tokens=False)
        raw = tokenizer("".join(pieces), add_special_tokens=False)
        assert split["input_ids"] == raw["input_ids"]
        assert len(split.word_ids()) == len(split["input_ids"])


if __name__ == "__main__":
    test_fragments_preserve_raw_text_except_irrelevant_trailing_space()
    test_fragment_tokenization_matches_raw_pythia_tokenization()
    print("all passed")
