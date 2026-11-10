"""Reimplementation of the BASIC stage of Luo, Zhao & Zhan, "Named Entity
Recognition Only from Word Embeddings" (EMNLP 2020, arXiv:1909.00164).

The authors' repository (github.com/cslydia/uNER) contains only a README -- no
code was released -- so this follows the paper's text and equations. It is the
"basic" row of their Table 1 (CoNLL-2003 English: P 62.57, R 56.83, F1 60.76);
the BiLSTM-CRF and RL refinement stages are not reimplemented.

  1. Clustering (Sec. 2.1)    K-means (k=2) on the SENNA 50d embeddings of the
                              vocabulary; the smaller cluster is tagged 1 (NE)
                              and its words form a coarse NE dictionary.
  2. Gaussian-HMM (Sec. 2.2)  states O / I / B. Emission = full-covariance
                              Gaussian on the GloVe 100d embedding x_i, times
                              p(cluster tag | state) -- the paper's "cluster
                              embeddings", fine-tuned. Trained by maximising the
                              forward-algorithm likelihood; Viterbi decoding.
       + single-word NEs      a predicted one-word NE is dropped if that word is
                              tagged NE in under half its corpus occurrences and
                              is not in the coarse dictionary.
       + high-quality phrases adjacent predicted spans are merged when their
                              boundary words form a word2phrase phrase.
  3. DAGMM (Sec. 2.3)         span u = [x_i; mean(x_i..x_j); x_j] (GloVe, eq. 5)
                              -> compression net [75, 15] + reconstruction
                              features (relative Euclidean distance, cosine) ->
                              estimation net (10 hidden) -> GMM, K = 4 types.
Types are anonymous clusters, mapped to gold types one-to-one for evaluation
only (conll.match_clusters), as in the paper.

Choices the paper leaves open are marked DECISION; none uses gold labels.

    python3 unsupervised/luo2020.py [--dagmm-seeds 0,1,2] [--device cuda]
Writes outputs/unsupervised/luo2020.json (scores) and luo2020_spans.json.
"""
from __future__ import annotations

import argparse, json, math, re, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from conll import (load, match_clusters, apply_mapping, span_scores,   # noqa: E402
                   typed_scores, spans as iob_spans)

EMB = ROOT / "data/embeddings"
OUT = ROOT / "outputs/unsupervised"
O, I, B = 0, 1, 2                  # the paper's column order of the cluster table


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
def senna_key(w: str) -> str:
    """SENNA's own normalisation: lowercase, every digit -> 0."""
    return re.sub(r"\d", "0", w.lower())


def load_senna() -> dict[str, np.ndarray]:
    words = (EMB / "senna/hash/words.lst").read_text(encoding="utf-8").split("\n")
    vecs = np.loadtxt(EMB / "senna/embeddings/embeddings.txt", dtype=np.float32)
    return dict(zip(words[:len(vecs)], vecs))


def load_glove(vocab: set[str]) -> dict[str, np.ndarray]:
    out = {}
    with open(EMB / "glove.6B.100d.txt", encoding="utf-8") as f:
        for line in f:
            w, rest = line.split(" ", 1)
            if w in vocab:
                out[w] = np.asarray(rest.split(), dtype=np.float32)
    return out


class Embedder:
    """Word -> vector for both tables. DECISION: unknown GloVe words get the
    mean GloVe vector; unknown SENNA words get SENNA's own UNKNOWN row."""

    def __init__(self, corpus_words: set[str]):
        self.senna = load_senna()
        self.glove = load_glove({w.lower() for w in corpus_words})
        self.glove_unk = np.mean(np.stack(list(self.glove.values())), 0)

    def senna_vec(self, w):
        return self.senna.get(senna_key(w), self.senna["UNKNOWN"])

    def glove_vec(self, w):
        return self.glove.get(w.lower(), self.glove_unk)

    def glove_sent(self, words):
        return np.stack([self.glove_vec(w) for w in words])


# --------------------------------------------------------------------------- #
# 1. clustering
# --------------------------------------------------------------------------- #
def cluster_vocabulary(emb: Embedder, sents, seed=0) -> set[str]:
    """K-means (k=2) over the vocabulary's SENNA vectors; returns the coarse NE
    dictionary (SENNA keys of the smaller cluster).
    DECISION: "the whole vocabulary" = the corpus vocabulary that HAS a SENNA
    embedding, one point per distinct key. The 12% of corpus words SENNA lacks
    all share its UNKNOWN vector; clustered, that one point mass becomes the
    small cluster on its own (cluster-span F1 0.04, vs the paper's 0.47).
    Unembedded words are therefore not clustered and so not in the dictionary
    (tag 0). Other readings checked: SENNA's full 130k vocabulary (F1 0.00),
    frequency-weighted (0.00); this one gives 0.61, stable over seeds."""
    keys = sorted({senna_key(w) for s in sents for w in s} & emb.senna.keys())
    X = np.stack([emb.senna[k] for k in keys])
    lab = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(X)
    ne = int(np.argmin(np.bincount(lab)))
    return {k for k, l in zip(keys, lab) if l == ne}


def cluster_tags(sents, ne_dict):
    return [[int(senna_key(w) in ne_dict) for w in s] for s in sents]


def runs_as_spans(tags01):
    """Contiguous 1s -> spans (how a cluster-only tagger reads)."""
    out, start = [], None
    for i, t in enumerate(list(tags01) + [0]):
        if t and start is None:
            start = i
        elif not t and start is not None:
            out.append((start, i)); start = None
    return out


# --------------------------------------------------------------------------- #
# 2. Gaussian HMM
# --------------------------------------------------------------------------- #
class GaussianHMM:
    """3-state HMM: p(z_t | z_t-1) multinomial, emission N(x; mu_z, Sigma_z)
    (full covariance) times p(cluster tag | z) -- eq. 3 of the paper."""

    def __init__(self, p):
        self.set(p)

    def set(self, p):
        self.log_start = torch.log(p["start"]); self.logA = torch.log(p["trans"])
        self.mu, self.cov = p["mu"], p["cov"]
        self.log_clus = torch.log(p["clus"])                       # [2 tags, 3 states]
        self.mvn = torch.distributions.MultivariateNormal(self.mu, covariance_matrix=self.cov)

    def emissions(self, x, c):
        """[B, T, 3] log p(x_t, c_t | z)."""
        return self.mvn.log_prob(x.unsqueeze(-2)) + self.log_clus[c]

    @torch.no_grad()
    def forward_backward(self, x, c, mask):
        """Posteriors gamma [B,T,3], pairwise xi summed over time [B,3,3], log-lik [B]."""
        e = self.emissions(x, c)
        Bn, T, _ = e.shape
        alpha = torch.empty_like(e); beta = torch.zeros_like(e)
        alpha[:, 0] = self.log_start + e[:, 0]
        for t in range(1, T):
            nxt = torch.logsumexp(alpha[:, t - 1].unsqueeze(2) + self.logA, 1) + e[:, t]
            alpha[:, t] = torch.where(mask[:, t:t + 1], nxt, alpha[:, t - 1])
        for t in range(T - 2, -1, -1):
            nxt = torch.logsumexp(self.logA + (e[:, t + 1] + beta[:, t + 1]).unsqueeze(1), 2)
            beta[:, t] = torch.where(mask[:, t + 1:t + 2], nxt, torch.zeros_like(nxt))
        logZ = torch.logsumexp(alpha[:, -1], 1)                    # alpha is frozen past the end
        gamma = torch.exp(alpha + beta - logZ[:, None, None]) * mask.unsqueeze(-1)
        xi = (alpha[:, :-1].unsqueeze(3) + self.logA + (e[:, 1:] + beta[:, 1:]).unsqueeze(2)
              - logZ[:, None, None, None])
        xi = (torch.exp(xi) * mask[:, 1:, None, None]).sum(1)
        return gamma, xi, logZ

    @torch.no_grad()
    def viterbi(self, x, c, mask):
        e = self.emissions(x, c)
        logA = self.logA
        delta = self.log_start + e[:, 0]
        back = []
        for t in range(1, x.shape[1]):
            s = delta.unsqueeze(2) + logA                          # [B, from, to]
            best, arg = s.max(1)
            new = best + e[:, t]
            m = mask[:, t:t + 1]
            delta = torch.where(m, new, delta)
            back.append(torch.where(m, arg, torch.arange(3, device=x.device).expand_as(arg)))
        path = [delta.argmax(1)]
        for bp in reversed(back):
            path.append(bp.gather(1, path[-1].unsqueeze(1)).squeeze(1))
        return torch.stack(path[::-1], 1)                          # [B, T]


def batches(sents, cl, emb, bs, device, shuffle_rng=None):
    order = np.arange(len(sents)) if shuffle_rng is None else shuffle_rng.permutation(len(sents))
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        T = max(len(sents[i]) for i in idx)
        x = np.zeros((len(idx), T, 100), np.float32)
        c = np.zeros((len(idx), T), np.int64)
        m = np.zeros((len(idx), T), bool)
        for j, i in enumerate(idx):
            n = len(sents[i])
            x[j, :n] = emb.glove_sent(sents[i]); c[j, :n] = cl[i]; m[j, :n] = True
        yield idx, torch.tensor(x, device=device), torch.tensor(c, device=device), \
            torch.tensor(m, device=device)


def hmm_init(sents, cl, emb, eps=0.05, diag=False):
    """DECISION (the paper gives no initialisation): read the cluster tags as a
    coarse IOB sequence -- a run of NE-cluster words is B I I ..., everything
    else O -- and take start/transition counts (add-one) and per-state GloVe
    mean/covariance from it. p(tag | state) starts at the paper's table with an
    epsilon so it can be fine-tuned: O <- cluster 0, I and B <- cluster 1."""
    states = []
    for s in cl:
        st, prev = [], 0
        for t in s:
            st.append(O if not t else (I if prev else B)); prev = t
        states.append(st)
    start = np.ones(3); trans = np.ones((3, 3))
    for st in states:
        start[st[0]] += 1
        for a, b in zip(st, st[1:]):
            trans[a, b] += 1
    X = np.concatenate([emb.glove_sent(s) for s in sents])
    S = np.concatenate(states)
    mu = np.stack([X[S == z].mean(0) for z in (O, I, B)])
    cov = np.stack([np.cov(X[S == z].T) + 1e-2 * np.eye(100) for z in (O, I, B)])
    if diag:
        cov = np.stack([np.diag(np.diag(c_)) for c_ in cov])
    clus = np.array([[1 - eps, eps, eps], [eps, 1 - eps, 1 - eps]])
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    return {"start": f(start / start.sum()), "trans": f(trans / trans.sum(1, keepdims=True)),
            "mu": f(mu), "cov": f(cov), "clus": f(clus)}


def train_hmm(sents, cl, emb, device, epochs=20, bs=512, reg=1e-2,
              clus_mode="em", eps=0.05, diag=False):
    """Maximise the forward-algorithm likelihood (the paper's objective) by
    Baum-Welch EM. DECISION: EM rather than gradient ascent -- gradient training
    of the 100-d full-covariance model diverged (log-lik/token -32 -> -42 in
    one epoch, after which every word was O); EM never lowers the likelihood.
    reg*I is added to each covariance in the M-step to keep it invertible.
    DECISION (validation split): clus_mode "em" re-estimates p(cluster tag |
    state) like every other parameter; "fixed" keeps the paper's initial
    table (with eps in place of its hard zeros), so the cluster signal keeps
    anchoring the states. diag=True uses diagonal covariances."""
    p = {k: v.to(device).double() for k, v in hmm_init(sents, cl, emb, eps, diag).items()}
    clus0 = p["clus"]
    hmm = GaussianHMM(p)
    n_tok = sum(map(len, sents)); eye = torch.eye(100, device=device, dtype=torch.float64)
    for ep in range(epochs):
        start = torch.zeros(3, device=device, dtype=torch.float64); trans = torch.zeros(3, 3).to(start)
        N = torch.zeros(3).to(start); S1 = torch.zeros(3, 100).to(start)
        S2 = torch.zeros(3, 100, 100).to(start); C = torch.zeros(2, 3).to(start); ll = 0.0
        for _, x, c, m in batches(sents, cl, emb, bs, device):
            x = x.double()
            g, xi, logZ = hmm.forward_backward(x, c, m)
            start += g[:, 0].sum(0); trans += xi.sum(0); ll += logZ.sum().item()
            N += g.sum((0, 1)); S1 += torch.einsum("btz,btd->zd", g, x)
            S2 += torch.einsum("btz,btd,bte->zde", g, x, x)
            C[0] += (g * (c == 0).unsqueeze(-1)).sum((0, 1)); C[1] += (g * (c == 1).unsqueeze(-1)).sum((0, 1))
        mu = S1 / N[:, None]
        cov = S2 / N[:, None, None] - torch.einsum("zd,ze->zde", mu, mu) + reg * eye
        if diag:
            cov = torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))
        clus = clus0 if clus_mode == "fixed" else (C + 1e-6) / (C + 1e-6).sum(0, keepdim=True)
        hmm.set({"start": start / start.sum(), "trans": trans / trans.sum(1, keepdim=True),
                 "mu": mu, "cov": cov, "clus": clus})
        share = (N / N.sum()).tolist()
        print(f"  EM {ep + 1}/{epochs}: log-lik/token {ll / n_tok:.3f} | state share "
              f"O {share[O]:.3f} I {share[I]:.3f} B {share[B]:.3f}", flush=True)
    return hmm


def decode(hmm, sents, cl, emb, device):
    """Viterbi states -> spans (an I that cannot continue a span starts one)."""
    out = [None] * len(sents)
    for idx, x, c, m in batches(sents, cl, emb, 512, device):
        paths = hmm.viterbi(x.double(), c, m).cpu().numpy()
        for j, i in enumerate(idx):
            tags = ["O" if z == O else ("B-NE" if z == B else "I-NE") for z in paths[j, :len(sents[i])]]
            out[i] = [(s, e) for s, e, _ in iob_spans(tags)]
    return out


# --------------------------------------------------------------------------- #
# refinements
# --------------------------------------------------------------------------- #
def single_word_filter(all_sents, all_spans, ne_dict, sents, sp):
    """Drop a one-word NE whose word is tagged NE in < half its occurrences
    (over the whole decoded corpus) and is not in the coarse dictionary."""
    ne, tot = Counter(), Counter()
    for s, ss in zip(all_sents, all_spans):
        inside = {i for a, b in ss for i in range(a, b)}
        for i, w in enumerate(s):
            tot[w.lower()] += 1; ne[w.lower()] += i in inside
    keep = lambda w: ne[w.lower()] / tot[w.lower()] >= 0.5 or senna_key(w) in ne_dict
    return [[(a, b) for a, b in ss if b - a > 1 or keep(s[a])] for s, ss in zip(sents, sp)]


class Phrases:
    """word2phrase's test, which the paper's eq. 4 cites: (count(a,b) - min_count)
    / (count(a) count(b)) * N > threshold, word2vec defaults min_count 5,
    threshold 100. DECISION: the min_count discount is word2phrase's (eq. 4
    omits it; without it every pair of rare words would qualify)."""

    def __init__(self, sents, min_count=5, threshold=100.0):
        self.uni = Counter(w.lower() for s in sents for w in s)
        self.bi = Counter((a.lower(), b.lower()) for s in sents for a, b in zip(s, s[1:]))
        self.n = sum(self.uni.values()); self.mc = min_count; self.t = threshold

    def __call__(self, a, b):
        a, b = a.lower(), b.lower()
        return (self.bi[(a, b)] - self.mc) / (self.uni[a] * self.uni[b]) * self.n > self.t


def phrase_merge(sents, sp, phrases):
    out = []
    for s, ss in zip(sents, sp):
        merged = []
        for a, b in sorted(ss):
            if merged and merged[-1][1] == a and phrases(s[a - 1], s[a]):
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        out.append(merged)
    return out


# --------------------------------------------------------------------------- #
# 3. DAGMM
# --------------------------------------------------------------------------- #
def span_reps(sents, sp, vec_sent):
    """eq. 5: [x_i; mean(x_i..x_j); x_j] for every span, in order."""
    reps = []
    for s, ss in zip(sents, sp):
        if not ss:
            continue
        X = vec_sent(s)
        reps += [np.concatenate([X[a], X[a:b].mean(0), X[b - 1]]) for a, b in ss]
    return np.stack(reps).astype(np.float32)


class DAGMM(nn.Module):
    def __init__(self, d_in, K=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 75), nn.Tanh(), nn.Linear(75, 15))
        self.dec = nn.Sequential(nn.Linear(15, 75), nn.Tanh(), nn.Linear(75, d_in))
        self.est = nn.Sequential(nn.Linear(17, 10), nn.Tanh(), nn.Dropout(0.5), nn.Linear(10, K))

    def forward(self, u):
        z = self.enc(u); r = self.dec(z)
        rel = (u - r).norm(dim=1, keepdim=True) / u.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos = F.cosine_similarity(u, r, dim=1).unsqueeze(1)
        t = torch.cat([z, rel, cos], 1)
        return r, t, torch.softmax(self.est(t), 1)


def gmm_energy(t, gamma, eps=1e-4):
    """eq. 9-11 on a batch: mixture parameters from gamma, energy E(t), penalty.
    A component with ~no responsibility would divide by ~0; its mass is clamped
    and eps*I keeps every covariance invertible (as DAGMM implementations do)."""
    phi = gamma.mean(0)
    mass = gamma.sum(0).clamp(min=1e-6)
    mu = (gamma.T @ t) / mass.unsqueeze(1)
    d = t.unsqueeze(1) - mu                                         # [N, K, D]
    cov = torch.einsum("nk,nki,nkj->kij", gamma, d, d) / mass[:, None, None]
    cov = cov + eps * torch.eye(t.shape[1], device=t.device)
    mvn = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
    logp = mvn.log_prob(t.unsqueeze(1)) + torch.log(phi + 1e-12)
    return -torch.logsumexp(logp, 1), (1.0 / torch.diagonal(cov, dim1=1, dim2=2)).sum()


def dagmm_cluster(train_u, test_u, device, seed=0, K=4, epochs=200, bs=1024, lr=1e-4,
                  lam1=0.1, lam2=0.005):
    """Fit DAGMM on train spans, return test cluster ids. DECISION: inputs are
    z-scored with train statistics; training follows the DAGMM paper (Adam 1e-4,
    batch 1024, 200 epochs, lambda1 0.1, lambda2 0.005) -- Luo et al. give only
    the layer sizes."""
    torch.manual_seed(seed)
    mean, std = train_u.mean(0), train_u.std(0) + 1e-6
    tr = torch.tensor((train_u - mean) / std, device=device)
    te = torch.tensor((test_u - mean) / std, device=device)
    model = DAGMM(tr.shape[1], K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for k in torch.randperm(len(tr), generator=g).split(bs):
            u = tr[k]
            r, t, gamma = model(u)
            E, P = gmm_energy(t, gamma)
            loss = ((u - r) ** 2).sum(1).mean() + lam1 * E.mean() + lam2 * P
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        _, t_tr, g_tr = model(tr)
        _, t_te, g_te = model(te)
    share = np.bincount(g_tr.argmax(1).cpu().numpy(), minlength=K) / len(tr)
    if share.min() >= 0.01:
        return g_te.argmax(1).cpu().numpy()
    # DECISION: the estimation network's argmax can collapse onto one
    # component (it did: all 16,059 spans in one cluster). DAGMM's premise is
    # density estimation in its learned low-dimensional space t, so then the
    # K-component GMM is fitted on t directly by EM instead.
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(K, covariance_type="full", random_state=seed, reg_covar=1e-4)
    gm.fit(t_tr.cpu().numpy())
    print(f"    DAGMM seed {seed}: estimation-net argmax degenerate (shares "
          f"{np.round(share, 3).tolist()}); GMM fitted on t instead", flush=True)
    return gm.predict(t_te.cpu().numpy())


def attach(sp, ids):
    out, k = [], 0
    for ss in sp:
        out.append([(a, b, int(ids[k + j])) for j, (a, b) in enumerate(ss)]); k += len(ss)
    return out


def typed_eval(gold, sp, ids):
    clustered = attach(sp, ids)
    mapping = match_clusters(gold, clustered)
    res = typed_scores(gold, apply_mapping(clustered, mapping))
    res["mapping"] = {str(k): v for k, v in mapping.items()}
    res["cluster_sizes"] = np.bincount(ids, minlength=4).tolist()
    return res


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dagmm-seeds", default="0,1,2")
    ap.add_argument("--hmm-epochs", type=int, default=20)
    ap.add_argument("--clus-mode", choices=["em", "fixed"], default="em")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--hmm-only", action="store_true",
                    help="stop after span detection (for choosing HMM settings on validation)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data = {s: load(s) for s in ("train", "validation", "test")}
    all_sents = [w for s in data.values() for w in s[0]]
    tr_s, te_s, te_g = data["train"][0], data["test"][0], data["test"][1]
    emb = Embedder({w for s in all_sents for w in s})
    print(f"embeddings: GloVe covers {len(emb.glove):,} corpus words", flush=True)

    ne_dict = cluster_vocabulary(emb, all_sents)
    cl = {s: cluster_tags(data[s][0], ne_dict) for s in data}
    res = {"cluster": span_scores(te_g, [runs_as_spans(c) for c in cl["test"]]),
           "ne_dictionary_size": len(ne_dict)}
    print(f"coarse NE dictionary: {len(ne_dict):,} words | cluster spans F1 "
          f"{res['cluster']['f1']:.3f}", flush=True)

    hmm = train_hmm(tr_s, cl["train"], emb, args.device, epochs=args.hmm_epochs,
                    clus_mode=args.clus_mode, eps=args.eps, diag=args.diag)
    dec = {s: decode(hmm, data[s][0], cl[s], emb, args.device) for s in data}
    res["hmm"] = span_scores(te_g, dec["test"])
    va_g = data["validation"][1]
    res["validation"] = {"cluster": span_scores(va_g, [runs_as_spans(c) for c in cl["validation"]]),
                         "hmm": span_scores(va_g, dec["validation"])}
    print(f"validation spans: cluster F1 {res['validation']['cluster']['f1']:.3f} | "
          f"HMM P {res['validation']['hmm']['precision']:.3f} R {res['validation']['hmm']['recall']:.3f} "
          f"F1 {res['validation']['hmm']['f1']:.3f}", flush=True)
    if args.hmm_only:
        return

    all_dec = [x for s in data for x in dec[s]]
    phrases = Phrases(all_sents)
    final = {}
    for s in data:
        sw = single_word_filter(all_sents, all_dec, ne_dict, data[s][0], dec[s])
        final[s] = phrase_merge(data[s][0], sw, phrases)
        if s == "test":
            res["hmm+single_word"] = span_scores(te_g, sw)
    res["hmm+single_word+phrases"] = span_scores(te_g, final["test"])
    for k in ("hmm", "hmm+single_word", "hmm+single_word+phrases"):
        print(f"{k:26s} spans P {res[k]['precision']:.3f} R {res[k]['recall']:.3f} "
              f"F1 {res[k]['f1']:.3f}", flush=True)

    tr_u = span_reps(tr_s, final["train"], emb.glove_sent)
    te_u = span_reps(te_s, final["test"], emb.glove_sent)
    runs = []
    for seed in [int(x) for x in args.dagmm_seeds.split(",")]:
        r = typed_eval(te_g, final["test"], dagmm_cluster(tr_u, te_u, args.device, seed))
        runs.append(r)
        print(f"DAGMM seed {seed}: typed P {r['precision']:.3f} R {r['recall']:.3f} "
              f"F1 {r['f1']:.3f} | clusters {r['cluster_sizes']} -> {r['mapping']}", flush=True)
    f1 = np.array([r["f1"] for r in runs])
    res["typed"] = {"runs": runs, "f1_mean": float(f1.mean()), "f1_sd": float(f1.std())}
    res["seconds"] = round(time.time() - t0)
    (OUT / "luo2020.json").write_text(json.dumps(res, indent=2))
    (OUT / "luo2020_spans.json").write_text(json.dumps(
        {s: final[s] for s in ("train", "test")}))
    print(f"typed F1 {f1.mean():.4f} ± {f1.std():.4f} over {len(f1)} DAGMM seeds "
          f"(paper, basic: 0.6076) | {res['seconds']} s")


if __name__ == "__main__":
    main()
