"""The selector's persistence a as an extra probe input feature.

    a_w = mean over selector seeds s and budgets rho in {0.1..0.9} of g_s(rho)_w

where g is the hard top-k word selection of that encoder's trained selector. It
is computed on the fly from each batch's token embeddings -- the ones the probe
already has -- so it covers the train split as well as the test split (the
selection logs only cover test), and needs no join back to dataset rows.

mode="a"     the real a of each word.
mode="perm"  the baseline: a permuted among the words of its own sentence. The
             permutation is fixed per sentence (seeded from the sentence's token
             ids, so it is identical in every epoch and at train and test), which
             keeps the marginal distribution of a and each sentence's multiset of
             values, and destroys only which word carries which value.

Selectors are the family's trained series (utils.plot_grounding.discover_series):
never an oracle, never e5's prefix-masked runs.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

RHOS = [round(0.1 * i, 4) for i in range(1, 10)]
MODES = ("a", "perm")


class Saliency:
    def __init__(self, dataset: str, family: str, encoder, dim: int, device: str, mode: str = "a"):
        from src.selector import RationaleSelectorModel
        from utils.plot_grounding import discover_series

        if mode not in MODES:
            raise ValueError(f"saliency mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        entry = next((s for s in discover_series(dataset)
                      if s["family"] == family and "oracle" not in s["label"]
                      and "prefix-masked" not in s["label"]), None)
        if entry is None:
            raise SystemExit(f"no trained {family} selector on {dataset}")
        xp = OmegaConf.load(Path(entry["bias_runs"][0]).parent / "config.yaml")
        self.models = []
        for run in entry["bias_runs"]:
            ck = sorted((Path(run) / "state/models").glob("model_*.pth"),
                        key=lambda p: int(p.stem.split("_")[1]))
            if not ck:
                continue
            m = RationaleSelectorModel(dim, loss_cfg=xp.model.loss,
                                       selector_cfg=OmegaConf.select(xp, "model.selector"),
                                       sent_encoder=encoder).to(device)
            st = torch.load(ck[-1], map_location=device)
            m.load_state_dict(st.get("model", st), strict=False)
            m.eval().requires_grad_(False)
            self.models.append(m)
        if not self.models:
            raise SystemExit(f"{family}/{dataset}: selector runs have no checkpoints")
        self.label = entry["label"]

    @torch.no_grad()
    def __call__(self, token_emb: torch.Tensor, batch: dict) -> torch.Tensor:
        """[B, W] feature in the probe's word order (first-subword slots)."""
        word_ids = batch["word_ids"]
        a = torch.stack([m.hard_word_selection(token_emb, RHOS, word_ids).mean(0)
                         for m in self.models]).mean(0)                      # [B, U]
        if self.mode == "perm":
            a = self._permute(a, batch)
        return a

    @staticmethod
    def _permute(a: torch.Tensor, batch: dict) -> torch.Tensor:
        from src.words import word_valid
        n_words = word_valid(batch["word_ids"]).sum(dim=1).long().tolist()
        ids, attn = batch["ids"].cpu().numpy(), batch["attn_mask"].cpu().numpy()
        out = a.clone()
        for i, n in enumerate(n_words):
            if n < 2:
                continue
            key = zlib.crc32(ids[i][attn[i] > 0].astype(np.int64).tobytes())
            perm = torch.as_tensor(np.random.default_rng(key).permutation(n), device=a.device)
            out[i, :n] = a[i, perm]
        return out
