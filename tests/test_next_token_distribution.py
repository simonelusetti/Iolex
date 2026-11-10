"""Distribution metrics used by the Pythia home-task evaluation."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "scripts"))
from home_task_pythia import distribution_metrics  # noqa: E402


def test_identical_distributions_have_no_change():
    lp = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, 0.0]]), -1)
    got = distribution_metrics(lp, lp)
    assert torch.allclose(got["forward_kl"], torch.zeros(2), atol=1e-7)
    assert torch.allclose(got["js_divergence"], torch.zeros(2), atol=1e-7)
    assert torch.allclose(got["total_variation"], torch.zeros(2), atol=1e-7)
    assert got["top1_agreement"].tolist() == [1.0, 1.0]


def test_known_binary_distribution_change():
    full = torch.tensor([[0.9, 0.1]]).log()
    compressed = torch.tensor([[0.1, 0.9]]).log()
    got = distribution_metrics(full, compressed)
    assert np.isclose(got["forward_kl"].item(), 0.8 * np.log(9), atol=1e-6)
    assert np.isclose(got["total_variation"].item(), 0.8, atol=1e-6)
    assert 0 < got["js_divergence"].item() < np.log(2)
    assert got["top1_agreement"].item() == 0.0
