# PEER: Personalized Evidence Extraction from Reviews

PEER selects `k` evidence sentences from an item's _existing_ reviews that
best anticipate the review a specific user will write for that item **before
that review exists**. Every candidate sentence and all user-history signal
temporally precede the target review (no future-review leakage).

PEER is a two-stage framework: (1) a LightGBM ranker scores every candidate
sentence on a composite relevance label, then (2) a greedy aspect-coverage
selector assembles the final `k`-sentence evidence set.

This repository contains the full experimental pipeline behind the paper:
temporal case construction, aspect extraction, sentence embeddings, the PEER
ranker/selector, task-adapted reproductions of five published
explainable-recommendation baselines (NARRE, HRDR, A2SPR, ERRA-R, PRAG), five
classical heuristic baselines, the evaluation/significance-testing tooling,
the independent-encoder/extractor robustness check, and the LLM case-study
tooling. It does **not** include the paper's raw review data, cached
embeddings, or trained model checkpoints (large binary artifacts) — see
["Getting the data"](#2-getting-the-data) below.

**On the published baselines:** NARRE, HRDR, A2SPR, ERRA-R, and PRAG are
implemented here as **task-adapted reimplementations** under PEER's
sentence-selection setting, not bit-exact reproductions of their original
repositories (those solve rating-prediction or generation tasks, not sentence
selection). See `configs/baselines/manifest.json` for exactly what each
adapter implements and how it differs from the original paper.

## Repository layout

```
peer/                   Core library: ranker features, selector, metrics,
                         aspect extraction, sentiment, embeddings I/O
scripts/                 Every pipeline stage as a standalone CLI script
baselines/
  published/             NARRE, HRDR (neural/), A2SPR, ERRA-R, PRAG
configs/
  baselines/manifest.json  Per-baseline mechanism description + honesty flags
tests/                    pytest suite for the core library
```

`data/`, `embeddings/`, `models/`, `outputs/`, `results/` are created by the
pipeline at runtime and are not included in this package.

## Install

```bash
pip install -e .
pip install -e ".[gpu,lightgbm,llm,dev]"   # optional extras as needed
python -m spacy download en_core_web_sm
```

Requires Python >=3.10. `lightgbm` needs a working OpenMP install (e.g.
`brew install libomp` on macOS); the ranker silently falls back to
scikit-learn's `HistGradientBoostingRegressor` if LightGBM is unavailable
(`peer/models.py`).

## Pipeline overview

```
raw reviews/metadata
  -> build_cases.py                        data/cases/{train,valid,test}.jsonl.gz
  -> cache_embeddings.py                   embeddings/embeddings.{npy,ids.json}
  -> cache_user_history_embeddings.py      models/user_history_embeddings.{npy,ids.json}
  -> extract_aspects.py                    data/processed/aspects/*.parquet
  -> train_peer_retriever.py               models/peer_target_retriever.pt
  -> build_features.py                     data/processed/pairs/{train,valid,test}.parquet
  -> train_ltr.py                          models/peer_ltr.pkl
  -> select_topk.py / run_baselines.py /
     run_published_baselines.py            outputs/predictions/{peer,baselines,published}/*.jsonl
  -> evaluate_evidence.py                  results/evaluation.csv, results/per_case.csv
  -> significance_test.py                  results/significance.csv
```

Everything below assumes the pooled/joint setup (PEER trained once on all
three platforms; see the paper's Implementation Details). Per-platform
variants used only by the personalization swap test are noted separately.

### 1. Cases

```bash
python scripts/build_cases.py --datasets baby yelp googlelocal \
  --raw-dir data/raw --output data/cases --seed 42
```

Expects `data/raw/{dataset}_reviews.jsonl` (+ `{dataset}_metadata.jsonl`
where available); `build_cases.py --help` documents the exact filename
patterns it searches for. Source data: Amazon Reviews 2023 (Baby &
Toddler), the Yelp Open Dataset, and Google Local Reviews.

### 2. Embeddings

Two embedding sets are cached, both keyed by the same sentence/case ids:

```bash
# Blob-encoded embeddings: candidate sentences, ground-truth sentences, and
# per-case metadata/user-history/ground-truth text blobs. Used as-is by
# every baseline (PRAG, ERRA-R, SBERT, BM25, ...).
python scripts/cache_embeddings.py --cases data/cases \
  --device cuda --output embeddings/embeddings

# PEER's own user-history representation: each history sentence encoded
# individually and mean-pooled.
python scripts/cache_user_history_embeddings.py --cases data/cases \
  --device cuda --output models/user_history_embeddings
```

### 3. Aspects

```bash
python scripts/extract_aspects.py --cases data/cases \
  --output data/processed/aspects
```

### 4. PEER's target-interest retriever

```bash
python scripts/train_peer_retriever.py --cases data/cases \
  --embeddings embeddings/embeddings \
  --user-history-override models/user_history_embeddings \
  --output models/peer_target_retriever.pt
```

### 5. Features (pairs parquet)

```bash
python scripts/build_features.py --cases data/cases \
  --aspects data/processed/aspects/sentence_aspects.parquet \
  --embeddings embeddings/embeddings \
  --user-history-override models/user_history_embeddings \
  --peer-retriever models/peer_target_retriever.pt \
  --semantic-weight 0.70 --aspect-weight 0.15 \
  --sentiment-weight 0.10 --coverage-weight 0.05 \
  --output data/processed/pairs
```

The label weights above are the shipped composite-label configuration
(Table "labelweight"). `--user-history-override` merges PEER's per-
sentence-pooled user-history embeddings into `user_sem_sim` and (via the
retriever) `target_emb_sim` only; every other column, and every baseline
that reads embeddings directly, keeps using the blob encoding.

### 6. Ranker

```bash
python scripts/train_ltr.py \
  --train data/processed/pairs/train.parquet \
  --valid data/processed/pairs/valid.parquet \
  --output models/peer_ltr.pkl
```

Uses `peer/models.py::FEATURE_COLUMNS_DEFAULT` -- PEER's 5 reported features
($\phi,\psi,\rho,\alpha,\beta$: `user_sem_sim`, `item_sem_sim`,
`target_emb_sim`, `user_aspect_overlap`, `item_aspect_salience`) -- with no
`--drop-features`. `build_features.py` also computes `metadata_sem_sim`
(read directly by the SBERT/BM25 `+metadata` baselines),
`helpfulness_norm`/`recency_norm` (read by the Popular/Recent baselines),
and `sentiment_match` (a composite-label ingredient, would leak the target
review's sentiment if used as a ranker feature); see `peer/models.py`'s
comments for why each of these, plus every other column
`build_features.py` computes, is excluded from the ranker itself.

### 7. Baselines

Published baselines (PRAG needs its own retriever; NARRE/HRDR need their
own trained scorer):

```bash
python scripts/train_prag_retriever.py --cases data/cases \
  --embeddings embeddings/embeddings --output models/prag_retriever.pt

python scripts/train_published_neural.py --model narre --dataset baby \
  --raw-dir data/raw --cases data/cases --output models/narre_baby.pt
python scripts/train_published_neural.py --model hrdr --dataset baby \
  --raw-dir data/raw --cases data/cases --output models/hrdr_baby.pt
# repeat --dataset for yelp / googlelocal
```

Selection/inference for every method:

```bash
python scripts/select_topk.py --pairs-dir data/processed/pairs \
  --model models/peer_ltr.pkl --splits test --method-name peer_full \
  --k-list 1 3 5 user_avg --lambda-coverage 0.10 \
  --embeddings embeddings/embeddings --output outputs/predictions/peer

python scripts/run_baselines.py --pairs-dir data/processed/pairs \
  --split test --k-list 1 3 5 user_avg \
  --methods random recent popular sbert_user_metadata bm25_user_metadata \
  --embeddings embeddings/embeddings --output outputs/predictions/baselines

python scripts/run_published_baselines.py --cases data/cases \
  --k-list 1 3 5 user_avg \
  --methods prag erra_r a2spr narre hrdr \
  --models-dir models --output outputs/predictions/published
```

(`select_topk.py --lambda-coverage 0.10` is the shipped selector value;
Table "selectortune" documents its sensitivity.) The paper's reported
"SBERT" and "BM25" rows are `sbert_user_metadata` / `bm25_user_metadata`
(history + metadata), not the history-only variants.

### 8. Evaluation

```bash
python scripts/evaluate_evidence.py --pred-dir outputs/predictions \
  --cases data/cases --embeddings embeddings/embeddings.npz \
  --output results/evaluation.csv --per-case-output results/per_case.csv

python scripts/significance_test.py --per-case results/per_case.csv \
  --main-method peer_full \
  --baselines prag erra_r sbert_user_metadata bm25_user_metadata a2spr \
              recent hrdr popular narre random \
  --output results/significance.csv
```

`results/evaluation.csv`, aggregated by `(dataset, k)` and pooled over
`k=user_avg`, is Table "main".

## Secondary tables and figures

Each of these is independent of the others and can be run in any order once
steps 1-8 above are complete.

**Ablation (Table "ablation")**

```bash
python scripts/run_ablation.py --pairs-dir data/processed/pairs \
  --embeddings embeddings/embeddings.npz --k-list user_avg \
  --variants full semantic_only aspect_only minus_phi minus_psi minus_rho \
             minus_alpha minus_beta no_coverage item_only
python scripts/evaluate_evidence.py --pred-dir outputs/predictions/ablation \
  --cases data/cases --embeddings embeddings/embeddings.npz \
  --output results/ablation.csv
```

**Evidence budget + k-budget Pareto (Table "budget", Figure "kbudget-pareto")**

```bash
python scripts/run_k_sensitivity.py --pairs-dir data/processed/pairs \
  --model models/peer_ltr.pkl --embeddings embeddings/embeddings.npz \
  --k-list 1 3 5 7 10 15 20 25 30 35 40 user_avg \
  --output-dir outputs/predictions/k_sensitivity \
  --results-output results/k_sensitivity.csv
python scripts/plot_kbudget_pareto.py --evaluation results/k_sensitivity.csv \
  --output paper/figures/kbudget_pareto.pdf
```

(`run_k_sensitivity.py` covers PEER + the 2 baselines it invokes directly;
run `run_published_baselines.py` with the same `--k-list` for PRAG/ERRA-R
and merge into the same evaluation CSV before plotting.)

**Label-weight sensitivity + Pareto (Tables "labelweight"/"selectortune",
Figure "labelweight-pareto")**

```bash
# valid split -- table + selection of the shipped config
python scripts/search_label_weights.py --pairs-dir data/processed/pairs \
  --cases data/cases --embeddings embeddings/embeddings.npz --split valid \
  --k-list user_avg --output results/label_weight_search_valid.csv

# test split -- robustness Pareto figure only, not used for tuning
python scripts/search_label_weights.py --pairs-dir data/processed/pairs \
  --cases data/cases --embeddings embeddings/embeddings.npz --split test \
  --k-list user_avg --output results/label_weight_search_test.csv

python scripts/plot_labelweight_pareto.py \
  --search-results results/label_weight_search_test.csv \
  --prag <PRAG pooled sem_f1> <PRAG pooled aspect_f1> \
  --erra <ERRA-R pooled sem_f1> <ERRA-R pooled aspect_f1> \
  --output paper/figures/labelweight_pareto.pdf

python scripts/tune_selector.py --pairs-dir data/processed/pairs \
  --cases data/cases --embeddings embeddings/embeddings.npz --split valid \
  --k-list user_avg --output results/selector_tuning.csv
```

**Independent-encoder / independent-extractor robustness (Table
"independent-encoder")**

```bash
python scripts/circularity_check_bge.py --cases data/cases \
  --pred-dir outputs/predictions --k user_avg \
  --methods peer_full prag erra_r sbert_user_metadata a2spr \
            bm25_user_metadata recent hrdr narre popular random \
  --output results/circularity_check_bge

python scripts/circularity_check_yake.py --cases data/cases \
  --pred-dir outputs/predictions --k user_avg \
  --methods peer_full prag erra_r sbert_user_metadata a2spr \
            bm25_user_metadata recent hrdr narre popular random \
  --output results/circularity_check_yake
```

`circularity_check_bge.py` re-encodes every method's already-selected
evidence with an independent sentence encoder (BGE) and is an exact
reproduction of the paper's methodology. `circularity_check_yake.py`
re-tags evidence with YAKE (an independent keyword extractor); **the exact
YAKE hyperparameters used to produce the paper's reported numbers were not
preserved** -- this script uses YAKE's own defaults as a clearly-labeled
reconstruction and should be expected to differ slightly (not
qualitatively) from the published values.

**Personalization swap test (Tables "personalization-div/-gain/
-efficiency/-correlation")**

Needs a _per-platform_ PEER ranker/retriever and a PRAG retriever (the
pooled/joint models from steps 4-6 are for the main comparison; this
analysis follows the paper's original per-platform protocol):

```bash
for ds in baby yelp googlelocal; do
  python scripts/train_peer_retriever.py --cases data/cases_$ds \
    --embeddings embeddings/embeddings_$ds \
    --user-history-override models/user_history_embeddings_$ds \
    --output models/peer_target_retriever_$ds.pt
  python scripts/build_features.py --cases data/cases_$ds \
    --aspects data/processed_$ds/aspects/sentence_aspects.parquet \
    --embeddings embeddings/embeddings_$ds \
    --user-history-override models/user_history_embeddings_$ds \
    --peer-retriever models/peer_target_retriever_$ds.pt \
    --semantic-weight 0.70 --aspect-weight 0.15 \
    --sentiment-weight 0.10 --coverage-weight 0.05 \
    --output data/processed_$ds/pairs
  python scripts/train_ltr.py \
    --train data/processed_$ds/pairs/train.parquet \
    --valid data/processed_$ds/pairs/valid.parquet \
    --output models/peer_ltr_$ds.pkl
  python scripts/train_prag_retriever.py --cases data/cases_$ds \
    --embeddings embeddings/embeddings_$ds --output models/prag_retriever_$ds.pt

  python scripts/personalization_swap_test.py --dataset $ds \
    --cases data/cases_$ds --pairs data/processed_$ds/pairs/test.parquet \
    --aspects data/processed_$ds/aspects/sentence_aspects.parquet \
    --embeddings embeddings/embeddings_$ds \
    --user-history-override models/user_history_embeddings_$ds \
    --peer-ranker models/peer_ltr_$ds.pkl \
    --peer-retriever models/peer_target_retriever_$ds.pt \
    --prag-retriever models/prag_retriever_$ds.pt \
    --output results/personalization_swap_$ds.csv
done

python scripts/compute_personalization_tables.py \
  --results results/personalization_swap_baby.csv \
            results/personalization_swap_yelp.csv \
            results/personalization_swap_googlelocal.csv \
  --output results/personalization_tables.log
```

The population (anchor selection, foreign-history sampling) is seeded
(`--seed 42`, default) but the original population-generation script from
which this was derived was lost to an unrelated machine restart mid-project
and reconstructed from the paper's own methodology description; it
reproduces the paper's reported N almost exactly (exact match on one
platform, within single digits on the other two) but is not guaranteed
bit-identical.

**Case study (Table "case-study")**

```bash
python scripts/export_case_studies.py --pred-dir outputs/predictions \
  --cases data/cases --output outputs/case_studies
```

## Known gaps

The 100-case zero-shot-LLM comparison (Tables "llm-100", "llm-100-sig",
"llm-feature-gap", "llm-feature-gap-subset") requires a stratified 100-case
sample (34/33/33 across platforms) and live LLM calls; `run_llm_baseline.py`
runs the LLM selection step (point it at a pre-filtered 100-case pairs
parquet), but the original sampling/feature-gap-aggregation glue code for
this specific analysis was not preserved and its exact numbers are not
guaranteed to reproduce bit-for-bit even with the same sampling seed, since
LLM API responses are not deterministic across providers/versions.
