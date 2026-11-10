"""Unsupervised NER from a trained selector (BERT, mean pooling).

The selectors are trained without labels -- only to keep a sentence's
embedding when words are removed -- yet they tend to keep entities. This turns
that into an NER system with the same two steps as Luo et al. (2020):

  1. Span detection   persistence a(w) = fraction of budgets rho in
                      {0.1, ..., 0.9} (and of selector seeds) at which word w is
                      kept, computed through the word-level selection path the
                      selectors were trained with. A 2-component Gaussian
                      mixture on a, fitted on TRAIN words, splits NE from non-NE
                      (the higher-mean component is NE); contiguous NE words
                      form one span.
  2. Typing           each span is the mean of its words' vectors from the
                      selector's own frozen encoder (bert-base-uncased, word =
                      mean of its subword states); PCA(50) + L2 norm, then
                      k-means with K = 4, fitted on train spans.
                      Also: Luo et al.'s DAGMM-on-GloVe typer on these spans,
                      to separate span quality from typing quality.
Types are anonymous clusters mapped one-to-one to gold types for evaluation
only, exactly as for the baseline (conll.match_clusters).

    python3 unsupervised/selector_ner.py [--label bert/mean] [--seeds 0,1,2]
Writes outputs/unsupervised/selector_<label>.json and the persistence cache.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from torch.nn.utils.rnn import pad_sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from conll import load, span_scores, typed_scores, match_clusters, apply_mapping   # noqa: E402
from src.selector import RationaleSelectorModel                                    # noqa: E402
from src.sentence import build_sentence_encoder                                    # noqa: E402
from src.utils import remap_checkpoint_state_dict                                  # noqa: E402
from src.words import gather_word_mean                                             # noqa: E402
from utils.forge_paths import discover_strategies                                  # noqa: E402

OUT = ROOT / "outputs/unsupervised"
RHOS = [round(0.1 * k, 1) for k in range(1, 10)]
SPLITS = ("train", "validation", "test")


def selectors(label, encoder, device):
    """All seeds of the conll2003 selector `label` (keep_special=false)."""
    family, pooling = label.split("/")[:2]
    for e in discover_strategies("conll2003"):
        if e["label"] != label or e["task"] != "rationale":
            continue
        xp = OmegaConf.load(Path(e["runs"][0]).parent / "config.yaml")
        if bool(OmegaConf.select(xp, "model.keep_special")):
            continue
        models, runs = [], sorted(e["runs"])
        for run in runs:
            ck = sorted((Path(run) / "state/models").glob("model_*.pth"),
                        key=lambda p: int(p.stem.split("_")[1]))[-1]
            m = RationaleSelectorModel(encoder.token_embeddings(
                    torch.tensor([[101]], device=device), torch.ones(1, 1, device=device)).shape[-1],
                loss_cfg=xp.model.loss, selector_cfg=OmegaConf.select(xp, "model.selector"),
                sent_encoder=encoder).to(device)
            st = torch.load(ck, map_location=device)
            m.load_state_dict(remap_checkpoint_state_dict(st.get("model", st), m.state_dict()))
            models.append(m.eval())
        return models, e["signature"], runs
    raise SystemExit(f"no conll2003 selector {label}")


def encode(tokenizer, sents, device):
    """The trainer's word-level inputs for pre-split words: no special tokens,
    word ids per subword, plus the slot -> word map (slot u = u-th word that
    produced tokens)."""
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids_l, wid_l, slots_l = [], [], []
    for words in sents:
        enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False,
                        truncation=True, max_length=510)
        wids = [-1 if w is None else w for w in enc.word_ids()]
        slots, prev = [], None
        for w in wids:
            if w >= 0 and w != prev:
                slots.append(w)
            prev = w
        ids_l.append(torch.tensor(enc["input_ids"])); wid_l.append(torch.tensor(wids)); slots_l.append(slots)
    ids = pad_sequence(ids_l, batch_first=True, padding_value=pad).to(device)
    attn = pad_sequence([torch.ones_like(x) for x in ids_l], batch_first=True).to(device)
    wids = pad_sequence(wid_l, batch_first=True, padding_value=-1).to(device)
    return ids, attn, wids, slots_l


@torch.no_grad()
def persistence_and_vectors(sents, models, encoder, tokenizer, device, bs=64):
    """Per sentence: a (fraction of rhos x seeds keeping each word) and the
    encoder's word vectors (mean of subword states), float16."""
    a_out, v_out = [], []
    for i in range(0, len(sents), bs):
        batch = sents[i:i + bs]
        ids, attn, wids, slots_l = encode(tokenizer, batch, device)
        emb = encoder.token_embeddings(ids, attn)                       # [B, L, D]
        kept = sum(m.hard_word_selection(emb, RHOS, wids) for m in models)  # [R, B, U]
        kept = (kept.mean(0) / len(models)).cpu().numpy()               # [B, U]
        wm = gather_word_mean(emb, wids).float().cpu().numpy()          # [B, U, D], same slots
        for b, (words, slots) in enumerate(zip(batch, slots_l)):
            a = np.zeros(len(words), np.float32)
            vec = np.zeros((len(words), emb.shape[-1]), np.float16)
            for u, w in enumerate(slots):
                a[w] = kept[b, u]; vec[w] = wm[b, u]
            a_out.append(a); v_out.append(vec)
    return a_out, v_out


def check_against_logs(A, runs):
    """The selectors' own training-time selection logs cover validation + test
    (the project concatenates them). Persistence recomputed here must match
    them word for word -- otherwise this is not the selector we think it is."""
    mine = np.concatenate(A["validation"] + A["test"])
    logged = []
    for run in runs:
        z = np.load(Path(run) / "data/selection_log.npz")
        use = ~np.isclose(z["rho"], 1.0)
        logged.append(z["selected"][use].mean(0))
    logged = np.mean(logged, 0)
    if len(logged) != len(mine):
        print(f"  CHECK: length mismatch, logs {len(logged)} vs recomputed {len(mine)}"); return
    d = np.abs(logged - mine)
    print(f"  CHECK vs selection logs: {len(mine):,} words, max |diff| {d.max():.2e}, "
          f"words differing {(d > 1e-6).sum():,}", flush=True)


def runs_as_spans(flags):
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i)); start = None
    return out


def span_vectors(vecs, sp):
    return np.stack([v[a:b].astype(np.float32).mean(0) for v, ss in zip(vecs, sp) for a, b in ss])


def kmeans_types(tr_x, te_x, seed, K=4):
    pca = PCA(50, random_state=seed).fit(tr_x)
    f = lambda x: normalize(pca.transform(x))
    km = KMeans(K, n_init=10, random_state=seed).fit(f(tr_x))
    return km.predict(f(te_x))


def attach(sp, ids):
    out, k = [], 0
    for ss in sp:
        out.append([(a, b, int(ids[k + j])) for j, (a, b) in enumerate(ss)]); k += len(ss)
    return out


def typed(gold, sp, ids):
    cl = attach(sp, ids)
    mapping = match_clusters(gold, cl)
    r = typed_scores(gold, apply_mapping(cl, mapping))
    r["mapping"] = {str(k): v for k, v in mapping.items()}
    r["cluster_sizes"] = np.bincount(ids, minlength=4).tolist()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="bert/mean")
    ap.add_argument("--seeds", default="0,1,2", help="seeds for the typing clusterers")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tag = args.label.replace("/", "_")
    data = {s: load(s) for s in SPLITS}

    cache = OUT / f"selector_{tag}_persistence.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        A, V = z["A"].item(), z["V"].item()
        print(f"loaded persistence cache {cache.name}")
    else:
        family, pooling = args.label.split("/")[:2]
        encoder, tokenizer = build_sentence_encoder(family, None, device=args.device, pooling=pooling)
        encoder.eval()
        models, sig, runs = selectors(args.label, encoder, args.device)
        print(f"{len(models)} selector seeds from experiment {sig}", flush=True)
        A, V = {}, {}
        for s in SPLITS:
            A[s], V[s] = persistence_and_vectors(data[s][0], models, encoder, tokenizer, args.device)
            print(f"  {s}: {len(A[s])} sentences", flush=True)
        check_against_logs(A, runs)
        np.savez(cache, A=np.array(A, dtype=object), V=np.array(V, dtype=object))

    # 1. span detection: 2-component GMM on train persistence
    a_tr = np.concatenate(A["train"])[:, None]
    gmm = GaussianMixture(2, random_state=0).fit(a_tr)
    hi = int(np.argmax(gmm.means_.ravel()))
    is_ne = lambda a: gmm.predict_proba(a[:, None])[:, hi] > 0.5 if len(a) else np.zeros(0, bool)
    sp = {s: [runs_as_spans(is_ne(a)) for a in A[s]] for s in SPLITS}
    thr = float(np.min(a_tr[gmm.predict_proba(a_tr)[:, hi] > 0.5])) if (gmm.predict_proba(a_tr)[:, hi] > 0.5).any() else 1.0
    res = {"label": args.label, "gmm_means": gmm.means_.ravel().tolist(), "a_threshold": thr,
           "ne_token_share_train": float(np.mean(is_ne(a_tr.ravel())))}
    for s in ("validation", "test"):
        res[f"spans_{s}"] = span_scores(data[s][1], sp[s])
    print(f"GMM on a: means {np.round(gmm.means_.ravel(), 3).tolist()} -> NE if a >= {thr:.3f} "
          f"({res['ne_token_share_train']:.1%} of train words)")
    for s in ("validation", "test"):
        r = res[f"spans_{s}"]
        print(f"spans {s:10s} P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f}", flush=True)

    # 2. typing on test spans
    te_g = data["test"][1]
    seeds = [int(x) for x in args.seeds.split(",")]
    tr_v, te_v = span_vectors(V["train"], sp["train"]), span_vectors(V["test"], sp["test"])
    km = [typed(te_g, sp["test"], kmeans_types(tr_v, te_v, sd)) for sd in seeds]
    res["typed_kmeans_bert"] = {"runs": km, "f1_mean": float(np.mean([r["f1"] for r in km])),
                                "f1_sd": float(np.std([r["f1"] for r in km]))}
    print(f"typed (k-means on BERT spans) F1 {res['typed_kmeans_bert']['f1_mean']:.4f} "
          f"± {res['typed_kmeans_bert']['f1_sd']:.4f}", flush=True)

    from luo2020 import Embedder, span_reps, dagmm_cluster
    emb = Embedder({w for s in SPLITS for x in data[s][0] for w in x})
    tr_u = span_reps(data["train"][0], sp["train"], emb.glove_sent)
    te_u = span_reps(data["test"][0], sp["test"], emb.glove_sent)
    dg = [typed(te_g, sp["test"], dagmm_cluster(tr_u, te_u, args.device, sd)) for sd in seeds]
    res["typed_dagmm_glove"] = {"runs": dg, "f1_mean": float(np.mean([r["f1"] for r in dg])),
                                "f1_sd": float(np.std([r["f1"] for r in dg]))}
    print(f"typed (DAGMM on GloVe spans, Luo et al.'s typer) F1 "
          f"{res['typed_dagmm_glove']['f1_mean']:.4f} ± {res['typed_dagmm_glove']['f1_sd']:.4f}")
    res["seconds"] = round(time.time() - t0)
    (OUT / f"selector_{tag}.json").write_text(json.dumps(res, indent=2))
    (OUT / f"selector_{tag}_spans.json").write_text(json.dumps({s: sp[s] for s in ("train", "test")}))


if __name__ == "__main__":
    main()
