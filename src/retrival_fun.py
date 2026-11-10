"""STS-B sufficiency test (paper Section 4.3/5.1) and cross-corpus
generalization support. Despite the filename (kept for continuity with the
original research codebase this was ported from), this has nothing to do
with retrieval -- it evaluates whether the tokens a trained selector keeps
are sufficient to preserve an encoder's sentence-similarity behaviour.

For each retention rate rho: filter sentence1 of an STS-B pair through the
trained selector (keeping only its top-k tokens for that rho), re-encode,
and compare its cosine similarity against the *unfiltered* sentence2 to the
gold human similarity score via Spearman correlation. Compared against a
random-selection baseline of the same size, averaged over several seeds.

This module is dataset-independent of whatever the selector was *trained*
on (see src/data.py's "stsb" dataset entry for that side) -- it always
evaluates against STS-B's validation split (which has real gold labels,
unlike GLUE's STS-B test split), regardless of cfg.data.dataset. That is
what makes cross-corpus generalization (paper Fig. 3, rightmost panel)
possible: every training run gets STS-B-evaluated at the end, whatever
corpus it trained on.
"""
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from scipy.stats import spearmanr
from tqdm import tqdm
from numpy import linspace

from .sentence import ALIAS_TO_CANON
from .utils import should_disable_tqdm
from .words import token_units


def build_non_special_mask(tokenizer, input_ids, attention_mask, device):
    special_tokens_mask = torch.tensor(
        [
            tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            for ids in input_ids.detach().cpu().tolist()
        ],
        dtype=attention_mask.dtype,
        device=device,
    )
    return attention_mask * (1 - special_tokens_mask)


def get_rhos(cfg) -> list[float]:
    return list(linspace(
        cfg.sweep_range[0],
        cfg.sweep_range[1],
        cfg.sweep_range[2],
    ))


class STSBDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "sentence1": item["sentence1"],
            "sentence2": item["sentence2"],
            # sentence-transformers/stsb's column is "score" (0-1, already
            # normalized), not GLUE stsb's raw "label" (0-5) -- see the
            # comment on run_stsb_sweep's load_dataset call for why this repo
            # uses that source instead of raw GLUE.
            "label": float(item["score"]),
        }


def build_stsb_collate(tokenizer, device, max_length, family: str, keep_special: bool = False):
    """STS-B batches with word maps, tokenized exactly as src/data.py's encode_examples
    does for training: words = sentence.split(), e5's "query: " prefix inserted
    after a leading [CLS] as non-word positions, specials stripped unless
    keep_special. Each side carries `word_ids` (-1 = not a word) so selectors
    choose whole words and rho counts words -- the unit they were trained on.
    """
    is_e5 = ALIAS_TO_CANON[family.lower()] == "e5"
    prefix_ids = tokenizer.encode("query: ", add_special_tokens=False) if is_e5 else []
    special_ids = set(tokenizer.all_special_ids)

    def encode_side(sentences):
        ids_l, wid_l = [], []
        for s in sentences:
            enc = tokenizer(s.split(), truncation=True, max_length=max(1, max_length - len(prefix_ids)),
                            is_split_into_words=True)
            ids, wids = enc["input_ids"], enc.word_ids()
            lead = 1 if (prefix_ids and ids and ids[0] == tokenizer.cls_token_id) else 0
            ids = ids[:lead] + prefix_ids + ids[lead:]
            wids = wids[:lead] + [None] * len(prefix_ids) + wids[lead:]
            if not keep_special:
                keep = [not (w is None and t in special_ids) for t, w in zip(ids, wids)]
                ids = [t for t, k in zip(ids, keep) if k]
                wids = [w for w, k in zip(wids, keep) if k]
            ids_l.append(torch.tensor(ids))
            wid_l.append(torch.tensor([-1 if w is None else w for w in wids]))
        return {
            "input_ids": pad_sequence(ids_l, batch_first=True, padding_value=tokenizer.pad_token_id).to(device),
            "attention_mask": pad_sequence([torch.ones_like(x) for x in ids_l], batch_first=True).to(device),
            "word_ids": pad_sequence(wid_l, batch_first=True, padding_value=-1).to(device),
        }

    def collate_fn(batch):
        return {
            "t1": encode_side([x["sentence1"] for x in batch]),
            "t2": encode_side([x["sentence2"] for x in batch]),
            "labels": torch.tensor([x["label"] for x in batch], dtype=torch.float),
        }

    return collate_fn


@torch.no_grad()
def eval_baseline(loader, encoder):
    """Unfiltered (full-sentence, no selection) cosine-sim vs gold Spearman."""
    sims, labels = [], []

    for batch in tqdm(
        loader,
        desc="STS-B baseline",
        leave=False,
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
        file=sys.stderr,
    ):
        t1, t2 = batch["t1"], batch["t2"]

        e1 = encoder.token_embeddings(t1["input_ids"], t1["attention_mask"])
        e2 = encoder.token_embeddings(t2["input_ids"], t2["attention_mask"])

        z1 = encoder.pool_full(e1, t1["attention_mask"])
        z2 = encoder.pool_full(e2, t2["attention_mask"])

        sims.extend(F.cosine_similarity(z1, z2).cpu().tolist())
        labels.extend(batch["labels"].tolist())

    return float(spearmanr(sims, labels)[0])


@torch.no_grad()
def eval_sweep(
    loader,
    encoder,
    mask_generator,
    eval_cfg,
    desc: str = "STS-B selector",
    progress_bar=None,
):
    rhos = get_rhos(eval_cfg)

    sims = {rho: [] for rho in rhos}
    labels = []

    iterator = loader
    if progress_bar is None:
        iterator = tqdm(
            loader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            disable=should_disable_tqdm(),
            file=sys.stderr,
        )

    for batch in iterator:
        t1, t2 = batch["t1"], batch["t2"]

        a1 = t1["attention_mask"]
        a2 = t2["attention_mask"]

        e2_full = encoder.token_embeddings(t2["input_ids"], a2)
        z2 = encoder.pool_full(e2_full, a2)

        new_a1_sweep = mask_generator(t1, a1, rhos)

        for rho, new_a1 in zip(rhos, new_a1_sweep):
            e1_masked = encoder.token_embeddings(t1["input_ids"], new_a1)
            z1 = encoder.pool(e1_masked, new_a1, valid_mask=a1)
            sims[rho].extend(F.cosine_similarity(z1, z2).cpu().tolist())

        labels.extend(batch["labels"].tolist())
        if progress_bar is not None:
            progress_bar.update(1)

    return {rho: float(spearmanr(sims[rho], labels)[0]) for rho in rhos}


def build_selector_mask_generator(
    selector,
    encoder,
    tokenizer,
    device,
    hard: bool = False,
    keep_special: bool = False,
):
    """Masks from the selector's own word-level forward: whole words kept, k
    counted in words, non-word positions treated exactly as in training."""
    @torch.no_grad()
    def mask_generator(t1, a1, rhos):
        e1_full = encoder.token_embeddings(t1["input_ids"], a1)
        z, g, _ = selector(t1["input_ids"], e1_full, a1, rhos=rhos, word_ids=t1["word_ids"])
        pred = g if hard else z
        return [pred_i * a1 for pred_i in pred]

    return mask_generator


def selector_keeps_nonword(selector) -> bool:
    """Whether the selector's forward keeps attended non-word positions (specials,
    e5's prefix) at every rho -- the random baseline follows the same rule."""
    return bool(selector.keep_nonword or selector.keep_last
                or getattr(selector.sent_encoder, "pooling", None) == "cls")


def build_random_mask_generator(cfg, tokenizer, device, keep_special: bool = False, keep_nonword: bool = False,
                                unit: str = "word"):
    """Random units at the selector's budget, k = max(1, round(rho * n_units)): whole
    words, or single pieces for unit="token" (match the selector being compared)."""
    @torch.no_grad()
    def mask_generator(t1, a1, rhos):
        wids = token_units(t1["word_ids"]) if unit == "token" else t1["word_ids"]
        is_word = wids >= 0
        nonword = (a1 > 0) & ~is_word
        n_words = (wids.max(dim=1).values + 1).clamp(min=0)
        new_a1_sweep = []
        for rho in rhos:
            k = (n_words.float() * rho).round().long()
            k = torch.where(n_words > 0, k.clamp(min=1), torch.zeros_like(k))
            keep = torch.zeros_like(a1, dtype=torch.float)
            for b in range(a1.size(0)):
                n = int(n_words[b])
                if n == 0:
                    continue
                chosen = torch.randperm(n, device=device)[: int(k[b])]
                keep[b] = torch.isin(wids[b], chosen).float()
            if keep_nonword:
                keep = keep.masked_fill(nonword, 1.0)
            new_a1_sweep.append(keep * a1)
        return new_a1_sweep

    return mask_generator


@torch.no_grad()
def eval_random_sweep(loader, encoder, tokenizer, eval_cfg, device, keep_special: bool = False,
                      keep_nonword: bool = False, unit: str = "word"):
    rhos = get_rhos(eval_cfg)
    runs = eval_cfg.random_selector.runs

    acc = {rho: [] for rho in rhos}
    cached = []

    for batch in loader:
        t1, t2 = batch["t1"], batch["t2"]
        a2 = t2["attention_mask"]
        e2_full = encoder.token_embeddings(t2["input_ids"], a2)
        cached.append((t1, encoder.pool_full(e2_full, a2), batch["labels"]))

    with tqdm(
        total=runs * len(cached),
        desc="STS-B random selector",
        leave=False,
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
        file=sys.stderr,
    ) as pbar:
        for run in range(runs):
            torch.manual_seed(eval_cfg.random_selector.seed + run)
            rand_mask_gen = build_random_mask_generator(
                eval_cfg,
                tokenizer,
                device,
                keep_special=keep_special,
                keep_nonword=keep_nonword,
                unit=unit,
            )
            sims = {rho: [] for rho in rhos}
            labels = []

            for t1, z2, batch_labels in cached:
                a1 = t1["attention_mask"]
                new_a1_sweep = rand_mask_gen(t1, a1, rhos)

                for rho, new_a1 in zip(rhos, new_a1_sweep):
                    e1_masked = encoder.token_embeddings(t1["input_ids"], new_a1)
                    z1 = encoder.pool(e1_masked, new_a1, valid_mask=a1)
                    sims[rho].extend(F.cosine_similarity(z1, z2).cpu().tolist())

                labels.extend(batch_labels.tolist())
                pbar.update(1)

            out = {rho: float(spearmanr(sims[rho], labels)[0]) for rho in rhos}

            for rho in rhos:
                acc[rho].append(out[rho])

    return {rho: sum(v) / len(v) for rho, v in acc.items()}


@torch.no_grad()
def run_stsb_sweep(cfg, device, encoder, tokenizer, selector):
    hf_ds = load_dataset("sentence-transformers/stsb", split="validation")
    ds = STSBDataset(hf_ds)

    keep_special = bool(cfg.model.get("keep_special", False))
    collate_fn = build_stsb_collate(
        tokenizer,
        device,
        cfg.data.max_length,
        family=str(cfg.data.encoder.family),
        keep_special=keep_special,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg.runtime.data.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    base = eval_baseline(loader, encoder)

    selector_mask_gen = build_selector_mask_generator(
        selector,
        encoder,
        tokenizer,
        device,
        hard=bool(cfg.runtime.eval.get("hard", True)),
        keep_special=keep_special,
    )

    ours = eval_sweep(loader, encoder, selector_mask_gen, cfg.runtime.eval)
    rand = eval_random_sweep(
        loader,
        encoder,
        tokenizer,
        cfg.runtime.eval,
        device,
        keep_special=keep_special,
        keep_nonword=selector_keeps_nonword(selector),
        unit=selector.unit,
    )

    return base, ours, rand
