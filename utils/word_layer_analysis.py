"""Word-type accuracy, best BERT layer, and empirical label entropy on a NER dataset.

    python -m utils.word_layer_analysis wikiann --device cuda
    python -m utils.word_layer_analysis fewnerd --device cuda

Uses existing layer probes, never trains. Binary scores collapse the original
predictions to entity/non-entity. Entropy uses the same held-out occurrences as
accuracy; words are grouped by exact original spelling, not WordPieces.
"""
import argparse
import csv
import gzip
import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import entropy, spearmanr
from sklearn.metrics import classification_report

from src.data import canonical_name, initialize_data, resolve_dataset, shuffle_and_subset, subset_split
from src.utils import configure_runtime, to_device
from tagger.layers import BertLayer
from tagger.model import first_subword_mask
from tagger.tagging import ROOT, _Probe

log = logging.getLogger(__name__)


def write_csv(path, fields, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


@torch.no_grad()
def score_words(probes, device):
    reports = [json.loads((probes / f"layer{i}/report.json").read_text()) for i in range(13)]
    assert all(r["layer"] == i and r["config"] == reports[0]["config"]
               and r["tags"] == reports[0]["tags"] for i, r in enumerate(reports))
    cfg = OmegaConf.create(reports[0]["config"])
    if "O" not in reports[0]["tags"]:
        raise ValueError("The entity/non-entity analysis requires a dataset with an O label.")
    cfg.runtime.device = device
    cfg.runtime, fell_back = configure_runtime(cfg.runtime)
    if fell_back:
        raise RuntimeError("CUDA requested but unavailable")
    train_dl, test_dl, encoder, tokenizer, *_ = initialize_data(
        cfg.data, cfg.runtime.data, None, device=cfg.runtime.device,
        keep_special=bool(cfg.model.keep_special))
    assert len(encoder.model.encoder.layer) == 12
    raw = shuffle_and_subset(resolve_dataset(cfg.data.dataset), cfg.data.subset, cfg.data.shuffle)["test"]
    raw = subset_split(raw, cfg.runtime.data.test_subset)
    assert len(raw) == len(test_dl.dataset)
    order = list(test_dl.sampler)
    sentences, word_indices, words, gold = [], [], [], []
    predictions = []

    for layer, report in enumerate(reports):
        probe = _Probe(cfg, BertLayer(encoder.model, layer), report["tags"],
                       train_dl, test_dl, cfg.runtime.device)
        probe.model.load_state_dict(torch.load(probes / f"layer{layer}/model.pth",
                                               map_location=cfg.runtime.device, weights_only=True))
        probe.model.eval()
        predicted, observed = [], []
        sentence_offset = 0
        for batch in test_dl:
            if layer == 0:
                first = first_subword_mask(batch["word_ids"])
                for row, positions in enumerate(first):
                    sentence = order[sentence_offset + row]
                    original = raw[sentence]
                    # Verify raw/cached sentence alignment, including truncation
                    # and the existing probe's removal of special tokens.
                    encoded = tokenizer(original["tokens"], is_split_into_words=True,
                                        truncation=True, max_length=int(cfg.data.max_length))
                    expected = encoded["input_ids"]
                    if not cfg.model.keep_special:
                        special = tokenizer.get_special_tokens_mask(expected, already_has_special_tokens=True)
                        expected = [v for v, s in zip(expected, special) if not s]
                    actual = batch["ids"][row][batch["attn_mask"][row].bool()].tolist()
                    assert expected == actual, f"Raw/tokenized alignment differs at sentence {sentence}"
                    for position in positions.nonzero().flatten().tolist():
                        word = int(batch["word_ids"][row, position])
                        label = probe.label_to_idx[str(batch["labels"][row][position])]
                        assert label == probe.label_to_idx[str(original["labels"][word])]
                        sentences.append(sentence)
                        word_indices.append(word)
                        words.append(original["tokens"][word])
                        gold.append(label)
            sentence_offset += batch["ids"].shape[0]
            emissions, mask, labels, _ = probe._forward(to_device(cfg.runtime.device, batch))
            predicted.extend(emissions.argmax(-1)[mask].cpu().tolist())
            observed.extend(labels[mask].cpu().tolist())
        assert observed == gold and sentence_offset == len(order)
        # An independent check against the report saved during training.
        measured = classification_report(gold, predicted, output_dict=True, zero_division=0)
        for key in ("macro avg", "weighted avg"):
            assert np.isclose(measured[key]["f1-score"], report["token_level"][key]["f1-score"], atol=1e-7)
        predictions.append(predicted)
        log.info("Scored layer %d/12: %d words; macro F1=%.4f (matches training report)",
                 layer, len(gold), measured["macro avg"]["f1-score"])
        del probe
    return (np.asarray(sentences), np.asarray(word_indices), np.asarray(words),
            np.asarray(gold), np.asarray(predictions, dtype=np.int64).T, reports[0])


def summarize(words, gold, predictions, tags, out, dataset="wikiann"):
    types, inverse, counts = np.unique(words, return_inverse=True, return_counts=True)
    layers = np.arange(predictions.shape[1])
    outside = tags.index("O")
    histogram, correlations, overview = [], [], []
    for regime in ("full", "binary"):
        labels = gold if regime == "full" else (gold != outside).astype(int)
        guesses = predictions if regime == "full" else (predictions != outside).astype(int)
        label_counts = np.zeros((len(types), len(tags) if regime == "full" else 2), dtype=int)
        np.add.at(label_counts, (inverse, labels), 1)
        h = entropy(label_counts, axis=1, base=2)
        correct = np.zeros((len(types), len(layers)), dtype=int)
        np.add.at(correct, inverse, guesses == labels[:, None])
        accuracy = correct / counts[:, None]
        winners = correct == correct.max(axis=1, keepdims=True)
        n_best = winners.sum(axis=1)
        credit = winners / n_best[:, None]
        best_mean = credit @ layers
        flat = n_best == len(layers)
        assert np.allclose(credit.sum(axis=1), 1)
        overview.append((regime, len(types), int(flat.sum()), int((n_best > 1).sum())))
        fields = ["word", "occurrences", "entropy_bits", "best_layers", "mean_best_layer", "n_best_layers"]
        fields += [f"label_count_{tag}" for tag in (tags if regime == "full" else ["non_entity", "entity"])]
        fields += [f"correct_layer{i}" for i in layers] + [f"accuracy_layer{i}" for i in layers]
        write_csv(out / f"word_types_{regime}.csv", fields,
                  ([word, int(counts[i]), float(h[i]), "|".join(map(str, layers[winners[i]])),
                    float(best_mean[i]), int(n_best[i]), *label_counts[i], *correct[i], *accuracy[i]]
                   for i, word in enumerate(types)))

        for minimum in (1, 5, 10, 20):
            for exclude_flat in (False, True):
                selected = (counts >= minimum) & (~flat if exclude_flat else True)
                n = int(selected.sum())
                shares = 100 * credit[selected].mean(axis=0) if n else np.full(len(layers), np.nan)
                histogram.extend([regime, minimum, exclude_flat, n, int(layer), float(shares[layer])]
                                 for layer in layers)
                rho = p = float("nan")
                if n > 2 and np.ptp(h[selected]) > 0 and np.ptp(best_mean[selected]) > 0:
                    rho, p = spearmanr(h[selected], best_mean[selected])
                correlations.append([regime, minimum, exclude_flat, n, float(rho), float(p)])
    write_csv(out / "best_layer_shares.csv",
              ["regime", "min_occurrences", "exclude_flat", "word_types", "layer", "percent"], histogram)
    write_csv(out / "correlations.csv",
              ["regime", "min_occurrences", "exclude_flat", "word_types", "spearman_rho", "p_two_sided"], correlations)

    lines = [f"# {dataset} word-type layer analysis", "",
             f"Held-out occurrences: {len(words):,}; distinct exact-spelling words: {len(types):,}.", "",
             "One final MLP per frozen BERT layer, all with the same seed; no retraining. Layer 0 is the embedding output.",
             "Uses the project's dataset builder and the saved training configuration for the evaluation split; training data is excluded.",
             "Each original word is scored using its first WordPiece, just as in tagger training; padding, special tokens,",
             "and words outside the encoder's truncation window are not scored. Sentence IDs index this evaluation split.", "",
             f"Full = exact original {len(tags)}-way label; binary = collapse gold and argmax predictions to O/non-O, not a separately trained classifier.",
             "Entropy is empirical Shannon entropy in bits, estimated over these same held-out occurrences.", "",
             "## Best-layer share of word types", "",
             "Each word type has total weight 1, split equally among tied best layers. These are type percentages, not occurrence percentages.",
             "All types are included below, even those tied at every layer. Columns sum to 100% before rounding.", "",
             "| Layer | Full labels (%) | Binary (%) |", "| --- | ---: | ---: |"]
    for layer in layers:
        values = [next(r[-1] for r in histogram if r[:3] == [regime, 1, False] and r[4] == layer)
                  for regime in ("full", "binary")]
        lines.append(f"| {layer} | {values[0]:.3f} | {values[1]:.3f} |")
    lines += ["", "## Ties", "", "| Regime | Any best-layer tie | Tied at all 13 layers |",
              "| --- | ---: | ---: |"]
    for regime, n, flat, tied in overview:
        lines.append(f"| {regime} | {tied:,} ({100*tied/n:.1f}%) | {flat:,} ({100*flat/n:.1f}%) |")
    lines += ["", "## Spearman: label entropy vs mean best-layer index", "",
              "Positive rho is the hypothesized direction (higher entropy, higher preferred layer).",
              "A tied word's preferred layer is the arithmetic mean of its winning layer indices; it need not itself be a winning layer.",
              "Constant-across-layer types contain no evidence of a layer preference, so both included/excluded results are reported.", "",
              "| Regime | Minimum occurrences | Exclude all-layer ties | Types | Rho | Two-sided p |",
              "| --- | ---: | --- | ---: | ---: | ---: |"]
    for regime, minimum, exclude_flat, n, rho, p in correlations:
        lines.append(f"| {regime} | {minimum} | {exclude_flat} | {n:,} | {rho:.4f} | {p:.3g} |")
    lines += ["", "These are exploratory associations from one seed. Rare words have noisy accuracy and entropy",
              "(singletons necessarily have zero empirical entropy); use the frequency-filtered rows as sensitivity checks.",
              "P-values are SciPy's asymptotic, unadjusted two-sided values and do not account for shared sentences",
              "or selection of the best layer on this same split. An association would not establish causation.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", nargs="?", default=None, help="defaults to wikiann, or the dataset saved in --probes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probes", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dataset = canonical_name(args.dataset or "wikiann")
    probes = args.probes or ROOT / "outputs/probe_layers" / dataset / "bert" / f"seed{args.seed}"
    metadata = json.loads((probes / "layer0/report.json").read_text())
    if args.dataset and dataset != metadata["dataset"]:
        parser.error("dataset does not match the reports in --probes")
    out = args.output or ROOT / "analysis/word_layers" / metadata["dataset"] / f"seed{metadata['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    sentences, indices, words, gold, predictions, metadata = score_words(probes, args.device)
    tags = metadata["tags"]
    outside = tags.index("O")
    full = predictions == gold[:, None]
    binary = (predictions != outside) == (gold[:, None] != outside)
    fields = ["sentence_index", "word_index", "word", "gold_label"]
    fields += [f"pred_layer{i}" for i in range(13)]
    fields += [f"correct_full_layer{i}" for i in range(13)]
    fields += [f"correct_binary_layer{i}" for i in range(13)]
    write_csv(out / "occurrences.csv.gz", fields,
              ([int(sentences[i]), int(indices[i]), word, tags[gold[i]],
                *[tags[p] for p in predictions[i]], *full[i].astype(int), *binary[i].astype(int)]
               for i, word in enumerate(words)))
    summarize(words, gold, predictions, tags, out, metadata["dataset"])
    log.info("Saved analysis to %s", out / "summary.md")


if __name__ == "__main__":
    main()
