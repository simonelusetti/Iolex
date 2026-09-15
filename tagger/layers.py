"""Train separate dataset probes on every frozen BERT hidden layer.

    python -m tagger.layers wikiann --device cuda
    python -m tagger.layers fewnerd --device cuda

Layer 0 is the embedding output; layer N follows transformer block N.
Completed layers are skipped on rerun; an interrupted layer starts over.
"""
import argparse
import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data import canonical_name, initialize_data
from src.sentence import bert_token_embeddings
from src.utils import configure_runtime
from .tagging import ROOT, _Probe, headline_f1, tag_names

log = logging.getLogger(__name__)


class BertLayer(torch.nn.Module):
    def __init__(self, model, layer_index):
        super().__init__()
        self.model, self.layer_index = model, layer_index

    def token_embeddings(self, ids, attention_mask):
        return bert_token_embeddings(self.model, ids, attention_mask, self.layer_index)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", nargs="?", default="wikiann")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/probe_layers")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dataset = canonical_name(args.dataset)
    tags = tag_names(dataset)

    cfg = OmegaConf.load(ROOT / "conf/config.yaml")
    cfg.data.dataset = dataset
    cfg.data.encoder.family = "bert"
    cfg.data.encoder.name = "bert-base-uncased"
    cfg.data.encoder.pooling = "mean"
    cfg.runtime.device = args.device
    cfg.runtime.seed = args.seed
    cfg.runtime.threads = 4
    cfg.runtime.eval.short_log = True
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.runtime.data.batch_size = args.batch_size
    if cfg.train.epochs < 1 or cfg.runtime.data.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    cfg.runtime, fell_back = configure_runtime(cfg.runtime)
    if fell_back:
        raise RuntimeError("CUDA requested but unavailable; not starting a CPU sweep.")
    config = OmegaConf.to_container(cfg, resolve=True)

    torch.manual_seed(args.seed)
    train_dl, test_dl, encoder, *_ = initialize_data(
        cfg.data, cfg.runtime.data, None, device=cfg.runtime.device,
        keep_special=bool(cfg.model.keep_special))
    # Worker startup must not advance the MLP/dropout random stream.
    for loader in (train_dl, test_dl):
        loader.generator = torch.Generator().manual_seed(args.seed)
    root = args.output_root / dataset / "bert" / f"seed{args.seed}"
    num_layers = len(encoder.model.encoder.layer) + 1
    log.info("Sweep: %d layers, %d epochs each, train=%d eval=%d, output=%s",
             num_layers, cfg.train.epochs, len(train_dl.dataset), len(test_dl.dataset), root)

    for layer_index in range(num_layers):
        out = root / f"layer{layer_index}"
        report_path = out / "report.json"
        if report_path.exists() and (out / "model.pth").exists():
            report = json.loads(report_path.read_text())
            if report.get("config") != config:
                raise ValueError(f"Config differs from {report_path}; use another --output-root.")
            log.info("Skipping completed layer %d: macro F1=%.4f", layer_index, headline_f1(report))
            continue
        # Same initial MLP and random seed for every representation.
        torch.manual_seed(args.seed)
        log.info("Training layer %d/%d", layer_index, num_layers - 1)
        probe = _Probe(cfg, BertLayer(encoder.model, layer_index), tags,
                       train_dl, test_dl, cfg.runtime.device)
        history = probe.fit(int(cfg.train.epochs))
        out.mkdir(parents=True, exist_ok=True)
        torch.save(probe.model.state_dict(), out / "model.pth")
        report = {
            "dataset": dataset, "family": "bert", "seed": args.seed,
            "layer": layer_index, "epochs": int(cfg.train.epochs), "tags": tags,
            "config": config,
            **{k: v for k, v in history[-1].items() if k not in ("epoch", "train_loss")},
            "history": history,
        }
        report_path.write_text(json.dumps(report, indent=2, default=lambda o: o.item()) + "\n")
        log.info("Saved layer %d: macro F1=%.4f, %s", layer_index, headline_f1(report), out)
        del probe
    log.info("Layer sweep complete.")


if __name__ == "__main__":
    main()
