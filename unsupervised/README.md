# Unsupervised NER on CoNLL-2003 English

Two systems, scored the same way on the CoNLL-2003 **test** split (3,453 sentences, 5,648 entities):

| file | what |
|---|---|
| `conll.py` | data (train / validation / test kept separate), entity-level scoring (matches `seqeval`), one-to-one cluster→type matching |
| `luo2020.py` | reimplementation of the **basic stage** of Luo, Zhao & Zhan, *Named Entity Recognition Only from Word Embeddings* (EMNLP 2020) |
| `selector_ner.py` | NER from a trained selector: word persistence → spans, encoder span vectors → types |

```bash
python3 unsupervised/luo2020.py --clus-mode fixed --eps 0        # the validation-chosen configuration
python3 unsupervised/selector_ner.py --label bert/mean            # or sbert/mean, e5/mean, ...
```
Outputs go to `outputs/unsupervised/`. Embeddings (GloVe 6B 100d, SENNA 50d) are expected in `data/embeddings/`.

Both systems are unsupervised: no gold tag is used to train anything. Entity types are anonymous clusters, mapped one-to-one to gold types **for evaluation only**, as in Luo et al. Configuration choices that the paper leaves open, or that our method needs, were made on the **validation** split. The test split is used only for the final numbers.

## Baseline: Luo et al. (2020), basic stage

The authors' repository (`github.com/cslydia/uNER`) contains only a README, with no code and no forks, so this follows the paper. The BiLSTM-CRF and RL refinement stages are not reimplemented. Open choices are marked `DECISION` in the code. The ones that change results:

- **Clustering.** K-means on the SENNA vectors of words that *have* one. The 12% of corpus words missing from SENNA all share its `UNKNOWN` vector, and clustered together they *become* the small cluster (span F1 0.04). With them excluded, cluster spans reach 0.61 on test, above the paper's 0.47.
- **HMM.** Trained by Baum-Welch EM, because gradient ascent diverged. If the cluster table is learned, or kept with ε > 0, the 100-d Gaussians take over and the HMM tags about half of all words as entities (validation span F1 0.23–0.26). Only the paper's literal table, with its hard zeros, keeps it anchored (validation 0.573), so that variant is used. The paper reports that the HMM *improves* on clustering (0.47 → 0.76); here it doesn't.
- **DAGMM.** The estimation network's argmax collapses onto one component. When any component gets under 1% of training spans, a 4-component GMM is fitted by EM on DAGMM's learned low-dimensional space instead.

## Selector NER

Persistence *a* of a word is the fraction of budgets ρ ∈ {0.1, …, 0.9} and selector seeds at which the selector keeps it. It's computed through the same word-level path the selectors were trained with, and it matches the selectors' own selection logs. A 2-component Gaussian mixture on *a* (train words) splits entities from the rest, and contiguous entity words become spans. Types come from k-means (K = 4) on the frozen encoder's span vectors. The script also runs Luo et al.'s DAGMM typer on the same spans.
