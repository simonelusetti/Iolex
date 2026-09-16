"""Train a focal-token NER classifier through learned per-token BERT masks.

The selector sees ordinary frozen last-layer BERT word representations and
produces one word-level mask per focal word. Each focal word is then classified
from a fresh BERT encoding of its sentence under that mask. BERT stays frozen,
but gradients pass through its soft attention mask into the selector.

Mask gates are max-anchored exponentials, exp((score - max(score)) / tau).
Consequently they lie in (0, 1], uniform scores mean a full all-ones mask, and
adding a constant to every score changes neither BERT nor the regularizer.

    python -m tagger.masked fewnerd --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Sampler

from src.data import PAD_TAG, collate, initialize_data
from src.utils import configure_runtime, to_device
from tagger.model import MLPTagger, first_subword_mask, gather_word_level
from tagger.tagging import ROOT, build_reports, headline_f1, tag_names


log = logging.getLogger(__name__)


class BucketBatchSampler(Sampler[list[int]]):
    """Length-homogeneous batches, shuffled at batch level each epoch."""

    def __init__(self, dataset, batch_size: int, seed: int):
        order = sorted(range(len(dataset)), key=lambda i: len(dataset[i]["ids"]))
        self.batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        for index in torch.randperm(len(self.batches), generator=generator).tolist():
            yield self.batches[index]

    def __len__(self):
        return len(self.batches)


class PairwiseSelector(nn.Module):
    """Contextual word representations -> focal-by-context score matrix."""

    def __init__(self, embedding_dim: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.query = nn.Linear(embedding_dim, hidden, bias=False)
        self.key = nn.Linear(embedding_dim, hidden, bias=False)
        self.scale = hidden ** -0.5

    def forward(self, word_emb, word_mask, sentence, focal, temperature):
        normalized = self.norm(word_emb)
        query = self.query(normalized[sentence, focal])
        key = self.key(normalized)[sentence]
        logits = torch.einsum("ch,cwh->cw", query, key) * self.scale
        valid = word_mask[sentence]
        logits = logits.masked_fill(~valid, -torch.inf)
        anchored = (logits - logits.amax(dim=-1, keepdim=True)) / temperature
        gates = anchored.exp().masked_fill(~valid, 0.0)
        focal_gate = torch.zeros_like(valid)
        focal_gate[torch.arange(len(focal), device=gates.device), focal] = True
        gates = gates.masked_fill(focal_gate, 1.0)
        return gates, valid


class LearnedMaskTagger(nn.Module):
    def __init__(self, embedding_dim: int, num_tags: int, selector_hidden: int,
                 tagger_hidden: int, dropout: float):
        super().__init__()
        self.selector = PairwiseSelector(embedding_dim, selector_hidden)
        self.tagger = MLPTagger(embedding_dim, num_tags, tagger_hidden, dropout)


def label_tensor(batch, label_to_idx, device):
    return torch.tensor(
        [[-1 if value == PAD_TAG else label_to_idx[str(value)] for value in row]
         for row in batch["labels"]],
        dtype=torch.long,
        device=device,
    )


def dense_word_slots(word_ids):
    """Map tokenizer word IDs to contiguous word slots within each row.

    Fast tokenizers can skip a source-word index (for example for an empty
    source token), while gather_word_level packs only words that produced at
    least one subword. The selector must use those packed slots rather than
    assuming the tokenizer IDs are contiguous.
    """
    first = first_subword_mask(word_ids)
    slots = first.cumsum(dim=1) - 1
    return slots.masked_fill(word_ids < 0, -1)


def focal_positions(batch, labels):
    mask = first_subword_mask(batch["word_ids"])
    sentence, token = mask.nonzero(as_tuple=True)
    word = dense_word_slots(batch["word_ids"])[sentence, token]
    target = labels[sentence, token]
    return sentence, token, word, target


def word_representations(encoder, batch, labels, args):
    with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=batch["ids"].is_cuda and args.amp == "bf16"):
        hidden = encoder.token_embeddings(batch["ids"], batch["attn_mask"])
    words, word_mask, _ = gather_word_level(hidden.float(), batch["word_ids"], labels)
    return words, word_mask


def expand_word_gates(gates, source_word_ids, source_attention):
    valid_word = source_word_ids >= 0
    slots = dense_word_slots(source_word_ids)
    gathered = gates.gather(1, slots.clamp(min=0))
    # Special tokens, if retained by a custom config, stay visible; padding
    # remains zero. The default NER config strips specials before batching.
    return torch.where(valid_word, gathered, source_attention.float())


def run_focals(model, encoder, batch, word_emb, word_mask, focals, args,
               train: bool):
    sentence, token, word, target = focals
    total = len(target)
    ce_sum = word_emb.new_zeros(())
    mask_sum = word_emb.new_zeros(())
    selected_sum = word_emb.new_zeros(())
    predictions = []
    max_length = batch["ids"].shape[1]
    dynamic_chunk = max(1, min(args.focal_chunk,
                               args.attention_budget // max(1, max_length * max_length)))

    for start in range(0, total, dynamic_chunk):
        stop = min(total, start + dynamic_chunk)
        source = sentence[start:stop]
        focal_word = word[start:stop]
        focal_token = token[start:stop]
        wanted = target[start:stop]

        gates, valid = model.selector(
            word_emb, word_mask, source, focal_word, args.temperature)
        context = valid.clone()
        context[torch.arange(len(source), device=source.device), focal_word] = False
        context_count = context.sum(dim=-1)
        per_focal_mean = (gates * context).sum(dim=-1) / context_count.clamp(min=1)
        per_focal_mean = per_focal_mean.masked_fill(context_count == 0, 0.0)

        ids = batch["ids"][source]
        soft_attention = expand_word_gates(
            gates, batch["word_ids"][source], batch["attn_mask"][source])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=ids.is_cuda and args.amp == "bf16"):
            masked_hidden = encoder.token_embeddings(ids, soft_attention)
            focal_hidden = masked_hidden[torch.arange(len(source), device=ids.device), focal_token]
            emissions = model.tagger.net(focal_hidden)
            ce = F.cross_entropy(emissions, wanted, reduction="sum")
        regularizer = per_focal_mean.sum()
        loss = (ce + args.mask_weight * regularizer) / total
        if train:
            loss.backward()

        ce_sum = ce_sum + ce.detach().float()
        mask_sum = mask_sum + regularizer.detach().float()
        selected_sum = selected_sum + ((gates >= 0.5) & context).sum().detach().float()
        predictions.extend(emissions.detach().argmax(dim=-1).cpu().tolist())
    return {
        "ce": ce_sum.item(),
        "mask": mask_sum.item(),
        "selected": selected_sum.item(),
        "predictions": predictions,
        "targets": target.detach().cpu().tolist(),
        "tokens": total,
    }


def train_epoch(model, encoder, loader, optimizer, label_to_idx, device, args):
    model.train()
    totals = {"ce": 0.0, "mask": 0.0, "selected": 0.0, "tokens": 0,
              "selector_grad": 0.0, "tagger_grad": 0.0}
    for step, raw in enumerate(loader, 1):
        batch = to_device(device, raw)
        labels = label_tensor(batch, label_to_idx, device)
        focals = focal_positions(batch, labels)
        words, word_mask = word_representations(encoder, batch, labels, args)
        optimizer.zero_grad(set_to_none=True)
        result = run_focals(model, encoder, batch, words, word_mask, focals, args, train=True)
        selector_grad = torch.nn.utils.clip_grad_norm_(model.selector.parameters(), float("inf"))
        tagger_grad = torch.nn.utils.clip_grad_norm_(model.tagger.parameters(), float("inf"))
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        for key in ("ce", "mask", "selected", "tokens"):
            totals[key] += result[key]
        totals["selector_grad"] += float(selector_grad)
        totals["tagger_grad"] += float(tagger_grad)
        if step % args.log_every == 0:
            n = max(1, totals["tokens"])
            log.info("batch %d/%d ce=%.4f mask=%.4f selected@.5=%.2f",
                     step, len(loader), totals["ce"] / n, totals["mask"] / n,
                     totals["selected"] / n)
    n = max(1, totals["tokens"])
    batches = max(1, len(loader))
    return {"cross_entropy": totals["ce"] / n,
            "mask_mean": totals["mask"] / n,
            "selected_at_0.5": totals["selected"] / n,
            "selector_grad_norm": totals["selector_grad"] / batches,
            "tagger_grad_norm": totals["tagger_grad"] / batches}


@torch.no_grad()
def evaluate(model, encoder, loader, label_to_idx, tags, device, args):
    model.eval()
    true, predicted = [], []
    totals = {"ce": 0.0, "mask": 0.0, "selected": 0.0, "tokens": 0}
    for raw in loader:
        batch = to_device(device, raw)
        labels = label_tensor(batch, label_to_idx, device)
        focals = focal_positions(batch, labels)
        words, word_mask = word_representations(encoder, batch, labels, args)
        result = run_focals(model, encoder, batch, words, word_mask, focals, args, train=False)
        lengths = first_subword_mask(batch["word_ids"]).sum(dim=1).tolist()
        cursor = 0
        for length in lengths:
            true.append([tags[i] for i in result["targets"][cursor:cursor + length]])
            predicted.append([tags[i] for i in result["predictions"][cursor:cursor + length]])
            cursor += length
        for key in totals:
            if key in result:
                totals[key] += result[key]
    n = max(1, totals["tokens"])
    return {
        "cross_entropy": totals["ce"] / n,
        "mask_mean": totals["mask"] / n,
        "selected_at_0.5": totals["selected"] / n,
        **build_reports(true, predicted),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", nargs="?", default="fewnerd")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=None,
                        help="PyTorch CPU threads (defaults to runtime config)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--focal-chunk", type=int, default=64)
    parser.add_argument("--attention-budget", type=int, default=2_000_000,
                        help="upper bound on focal_chunk * padded_subwords^2")
    parser.add_argument("--selector-hidden", type=int, default=128)
    parser.add_argument("--tagger-hidden", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mask-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--subset", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.temperature <= 0 or args.mask_weight < 0 or args.epochs < 1:
        parser.error("temperature and epochs must be positive; mask-weight must be non-negative")

    cfg = OmegaConf.load(ROOT / "conf/config.yaml")
    cfg.data.dataset = args.dataset
    cfg.data.encoder.family = "bert"
    cfg.data.encoder.name = "bert-base-uncased"
    cfg.data.encoder.pooling = "mean"
    cfg.data.subset = args.subset
    cfg.data.shuffle = False
    cfg.runtime.device = args.device
    cfg.runtime.seed = args.seed
    if args.threads is not None:
        cfg.runtime.threads = args.threads
    cfg.runtime.data.batch_size = args.batch_size
    cfg.runtime, fell_back = configure_runtime(cfg.runtime)
    if fell_back:
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(cfg.runtime.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_dl, test_dl, encoder, *_ = initialize_data(
        cfg.data, cfg.runtime.data, None, device=str(device), keep_special=False)
    train_dl = DataLoader(
        train_dl.dataset,
        batch_sampler=BucketBatchSampler(train_dl.dataset, args.batch_size, args.seed),
        num_workers=int(cfg.runtime.data.num_workers),
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg.runtime.data.num_workers) > 0,
    )
    tags = tag_names(args.dataset)
    label_to_idx = {str(i): i for i in range(len(tags))}
    label_to_idx.update({name: i for i, name in enumerate(tags)})
    with torch.no_grad():
        first = to_device(device, next(iter(train_dl)))
        dim = encoder.token_embeddings(first["ids"], first["attn_mask"]).shape[-1]
    model = LearnedMaskTagger(dim, len(tags), args.selector_hidden,
                              args.tagger_hidden, float(cfg.tagger.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    output = args.output or (ROOT / "outputs/masked_tagger" / args.dataset / f"seed{args.seed}")
    report_path = output / "report.json"
    if report_path.exists() and not args.retrain:
        raise FileExistsError(f"{report_path} exists; pass --retrain or choose --output")
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output"] = str(output)
    log.info("masked tagger: dataset=%s train=%d test=%d epochs=%d device=%s output=%s",
             args.dataset, len(train_dl.dataset), len(test_dl.dataset), args.epochs, device, output)

    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, encoder, train_dl, optimizer, label_to_idx, device, args)
        metrics["epoch"] = epoch
        history.append(metrics)
        log.info("epoch %d/%d ce=%.4f mask=%.4f selected@.5=%.2f selector_grad=%.4f",
                 epoch, args.epochs, metrics["cross_entropy"], metrics["mask_mean"],
                 metrics["selected_at_0.5"], metrics["selector_grad_norm"])

    test = evaluate(model, encoder, test_dl, label_to_idx, tags, device, args)
    report = {"dataset": args.dataset, "seed": args.seed, "config": config,
              "history": history, "test": test}
    torch.save({"selector": model.selector.state_dict(), "tagger": model.tagger.state_dict(),
                "config": config}, output / "model.pth")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("done: macro_f1=%.4f mask=%.4f selected@.5=%.2f",
             headline_f1(test), test["mask_mean"], test["selected_at_0.5"])


if __name__ == "__main__":
    main()
