"""gpt_token_embeddings must match HuggingFace exactly AND stay differentiable.

The second property is the reason the function is hand-written. HF's
create_causal_mask() treats attention_mask as a boolean keep/drop and detaches
it, so a soft selector weight reached the model as a constant: the output came
back with requires_grad=False and the selector never trained. Every llm run in
the store before this was an oracle for that reason.

    .venv/bin/python tests/test_gpt_mask.py        # no pytest needed
    .venv/bin/python -m pytest tests/test_gpt_mask.py -q   # if pytest is installed

pytest is not in requirements.txt and not in this venv, so the module-level
fixtures are optional and the file runs standalone via its __main__ block.
"""
import sys
from pathlib import Path

import torch

try:
    import pytest
except ModuleNotFoundError:                   # standalone mode
    pytest = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sentence import gpt_token_embeddings  # noqa: E402

MODEL = "EleutherAI/pythia-410m"


def _fixture(fn):
    return pytest.fixture(scope="module")(fn) if pytest else fn


@_fixture
def model():
    from transformers import AutoModel
    m = AutoModel.from_pretrained(MODEL).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@_fixture
def ids():
    torch.manual_seed(0)
    return torch.randint(0, 50000, (2, 12))


def test_matches_huggingface_forward(model, ids):
    """A 0/1 mask must reproduce HF's own output on the real tokens."""
    am = torch.ones(2, 12)
    am[1, 9:] = 0.0                       # right-padding, as this repo tokenises
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=am).last_hidden_state
        got = gpt_token_embeddings(model, ids, am)
    valid = am.bool()                     # padded positions are unconstrained in both
    assert torch.allclose(ref[valid], got[valid], atol=1e-4)


def test_gradient_reaches_a_fractional_mask(model, ids):
    """The property the old implementation lacked."""
    w = torch.full((2, 12), 0.6, requires_grad=True)
    out = gpt_token_embeddings(model, ids, w)
    assert out.requires_grad, "no graph: the selector would get no gradient"
    out.sum().backward()
    assert torch.isfinite(w.grad).all()
    assert not torch.all(w.grad == 0), "zero gradient everywhere: mask is inert"


def test_uniform_weight_is_a_no_op(model, ids):
    """Softmax is shift-invariant, so down-weighting everything equally does nothing.

    Pinned because it is the easy thing to mistake for a broken mask. The check is
    relative, not an absolute tolerance: 24 layers of float accumulation leave a
    residual of ~1e-4 (5e-5 of the signal) that does NOT scale with the weight --
    0.5 and 0.01 give the same 1.06e-4 -- whereas a mask that really bit would.
    So the invariant is that a uniform weight moves the output orders of magnitude
    less than a non-uniform one, which the next test measures at ~0.66.
    """
    w = torch.ones(2, 12)
    w[:, 3:7] = 0.1
    with torch.no_grad():
        base = gpt_token_embeddings(model, ids, torch.ones(2, 12))
        uniform = gpt_token_embeddings(model, ids, torch.full((2, 12), 0.5))
        non_uniform = gpt_token_embeddings(model, ids, w)
    uniform_effect = (base - uniform).abs().mean()
    real_effect = (base - non_uniform).abs().mean()
    assert uniform_effect < real_effect / 1000
    assert uniform_effect < 0.01 * base.abs().mean()


def test_non_uniform_weight_changes_the_output(model, ids):
    """...whereas down-weighting SOME tokens must move the representation."""
    w = torch.ones(2, 12)
    w[:, 3:7] = 0.1
    with torch.no_grad():
        a = gpt_token_embeddings(model, ids, torch.ones(2, 12))
        b = gpt_token_embeddings(model, ids, w)
    assert (a - b).abs().mean() > 0.01


if __name__ == "__main__":
    _model, _ids = model(), ids()
    failed = 0
    for _name, _fn in sorted(globals().items()):
        if not _name.startswith("test_"):
            continue
        try:
            _fn(_model, _ids)
            print(f"  PASS  {_name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {_name}: {exc}")
    print("all passed" if not failed else f"{failed} failure(s)")
    sys.exit(1 if failed else 0)
