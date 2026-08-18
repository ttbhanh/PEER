# PEER: Personalized Evidence Extraction from Reviews

PEER selects `k` evidence sentences from an item's _existing_ reviews that
best anticipate the review a specific user will write for that item **before
that review exists**. Every candidate sentence and all user-history signal
temporally precede the target review (no future-review leakage).

PEER is a two-stage framework: (1) a LightGBM ranker scores every candidate
sentence on a composite relevance label, then (2) a greedy aspect-coverage
selector assembles the final `k`-sentence evidence set.

This is the **core 8-step pipeline** behind the paper's main results table:
temporal case construction, aspect extraction, sentence embeddings, the PEER
ranker/selector, and task-adapted reproductions of five published
explainable-recommendation baselines (NARRE, HRDR, A2SPR, ERRA-R, PRAG) plus
five classical heuristic baselines, through evaluation and significance
testing.

**On the published baselines:** NARRE, HRDR, A2SPR, ERRA-R, and PRAG are
implemented here as **task-adapted reimplementations** under PEER's
sentence-selection setting, not bit-exact reproductions of their original
repositories (those solve rating-prediction or generation tasks, not sentence
selection). See `configs/baselines/manifest.json` for exactly what each
adapter implements and how it differs from the original paper.

## Contents

- [PEER: Personalized Evidence Extraction from Reviews](#peer-personalized-evidence-extraction-from-reviews)
  - [Contents](#contents)
  - [Repository layout](#repository-layout)
  - [Installation](#installation)
  - [Getting the data](#getting-the-data)
  - [Pipeline overview](#pipeline-overview)
    - [1. Cases](#1-cases)
    - [2. Embeddings](#2-embeddings)
    - [3. Aspects](#3-aspects)
    - [4. PEER's target-interest retriever](#4-peers-target-interest-retriever)
    - [5. Features (pairs parquet)](#5-features-pairs-parquet)
    - [6. Ranker](#6-ranker)
    - [7. Baselines](#7-baselines)
    - [8. Evaluation](#8-evaluation)

## Repository layout

```
peer/                   Core library: ranker features, selector, metrics,
                         aspect extraction, sentiment, embeddings I/O
scripts/                 The 8 pipeline-stage scripts below, as standalone CLIs
baselines/
  published/neural/       NARRE and HRDR scorers (used by run_published_baselines.py)
configs/
  baselines/manifest.json  Per-baseline mechanism description + honesty flags
```

`data/`, `embeddings/`, `models/`, `outputs/`, `results/` are created by the
pipeline at runtime and are not included in this package.

## Installation

```bash
pip install -e .
pip install -e ".[gpu,lightgbm,dev]"   # optional extras as needed
python -m spacy download en_core_web_sm
```

Requires Python >=3.10. `lightgbm` needs a working OpenMP install (e.g.
`brew install libomp` on macOS); the ranker silently falls back to
scikit-learn's `HistGradientBoostingRegressor` if LightGBM is unavailable
(`peer/models.py`).

**macOS note:** if a process that has imported `torch` later unpickles a
LightGBM model in the same process, you may hit a segfault inside
`LGBM_BoosterLoadModelFromString` caused by a duplicate OpenMP runtime
between the two libraries. Work around it by setting `OMP_NUM_THREADS=1`
(and, if that alone isn't enough, `KMP_DUPLICATE_LIB_OK=TRUE`) in the
environment before running. This is an environment issue, not a code bug.

**Memory note:** step 5 (`build_features.py`) is the most memory-intensive
stage — on a 64GB machine it has been observed to peak within a few hundred
MB of the physical limit for a _single_ platform alone. **Run it
sequentially, one platform at a time — never in parallel across
platforms** — or you risk OOM. Every other stage is comfortably lighter.

## Getting the data

The paper uses three public review datasets, none of which are redistributed
in this repository:

- **Amazon Reviews 2023 (Baby & Toddler category)** —
  <https://amazon-reviews-2023.github.io/>. Download the `Baby_Products`
  reviews and metadata files.
- **Yelp Open Dataset** — <https://www.yelp.com/dataset>. Requires accepting
  Yelp's dataset terms; provides `yelp_academic_dataset_review.json` and
  `yelp_academic_dataset_business.json` (the metadata source).
- **Google Local Reviews** — the Google Local Reviews (2021) crawl hosted at
  <https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local>; provides
  per-state review and metadata files.

Place (or convert, for Yelp's native JSON / Google Local's native format)
each dataset as **newline-delimited JSON** under `data/raw/`, one reviews
file and one metadata file per platform, named so that
`*{dataset}*review*.jsonl` and `*{dataset}*meta*.jsonl` glob-match them
(`build_cases.py`'s auto-discovery; `dataset` is `baby`, `yelp`, or
`googlelocal`) — e.g. `data/raw/baby_reviews.jsonl` and
`data/raw/baby_metadata.jsonl`. Each review row needs at least the columns
`build_cases.py` reads by default (override with `--user-col` / `--item-col`
/ `--text-col` / `--rating-col` / `--timestamp-col` / `--helpful-col` if your
raw files use different field names):

| Default column | Meaning                                                                                        |
| -------------- | ---------------------------------------------------------------------------------------------- |
| `user_id`      | reviewer identifier                                                                            |
| `parent_asin`  | item identifier (Google Local: pass `--item-col gmap_id`; Yelp: pass `--item-col business_id`) |
| `text`         | the review body                                                                                |
| `rating`       | star rating                                                                                    |
| `timestamp`    | integer epoch (seconds or ms both work; only relative order matters)                           |
| `helpful_vote` | helpfulness count (optional signal for the Popular baseline; defaults to 0 if absent)          |

Each metadata row needs at least the item-id column and enough descriptive
text (title, category, description) for `build_cases.py` to assemble a
per-item `metadata_text` blob — see `--help` for the exact fields it looks
for per dataset.

## Pipeline overview

**PEER is trained independently per platform** — one ranker, one
target-interest retriever, and one aspect vocabulary per platform (Amazon
Baby / Yelp / Google Local in the paper). Training one model per platform
keeps every platform's train/validation/test boundary strictly
self-contained in calendar time. Every step below (1-8) therefore runs in a
`for ds in baby yelp googlelocal; do ... done` loop against that platform's
own data, with entirely separate `data/`, `embeddings/`, and `models/`
artifacts per platform; only the final evaluation/reporting step pools
across platforms, and it does so by **concatenating** each platform's
already-computed per-case results, never by pooling training data.

```
raw reviews/metadata (per platform)
  -> build_cases.py                        data/cases_$ds/{train,valid,test}.jsonl.gz
  -> cache_embeddings.py                   embeddings/embeddings_$ds.{npy,ids.json}
  -> cache_user_history_embeddings.py      models/user_history_embeddings_$ds.{npy,ids.json}
  -> extract_aspects.py                    data/processed_$ds/aspects/*.parquet
  -> train_peer_retriever.py               models/peer_target_retriever_$ds.pt
  -> build_features.py                     data/processed_$ds/pairs/{train,valid,test}.parquet
  -> train_ltr.py                          models/peer_ltr_$ds.pkl
  -> select_topk.py / run_baselines.py /
     run_published_baselines.py            outputs/predictions_$ds/{peer,baselines,published}/*.jsonl
  -> evaluate_evidence.py                  results/evaluation_$ds.csv, results/per_case_$ds.csv
(repeat for ds in baby yelp googlelocal)
  -> concatenate results/per_case_{baby,yelp,googlelocal}.csv -> results/per_case_pooled.csv
  -> significance_test.py (per platform, and again on the pooled concatenation)
```

### 1. Cases

```bash
for ds in baby yelp googlelocal; do
  python scripts/build_cases.py --datasets $ds \
    --raw-dir data/raw --output data/cases_$ds \
    --max-cases-per-dataset 30000 --seed 42
done
```

`--max-cases-per-dataset 30000` reproduces the paper's 90,000-case benchmark
(30,000 eligible cases per platform, reservoir-sampled uniformly across
every eligible case seen in a single streaming pass). Omitting it processes
every eligible case in the raw data, which is far more than 30,000 for the
full raw datasets and does not match the paper's construction.

### 2. Embeddings

```bash
for ds in baby yelp googlelocal; do
  # Blob-encoded embeddings: candidate sentences, ground-truth sentences, and
  # per-case metadata/user-history/ground-truth text blobs. Used as-is by
  # every baseline (PRAG, ERRA-R, SBERT, BM25, ...).
  python scripts/cache_embeddings.py --cases data/cases_$ds \
    --device cuda --output embeddings/embeddings_$ds

  # PEER's own user-history representation: each history sentence encoded
  # individually and mean-pooled.
  python scripts/cache_user_history_embeddings.py --cases data/cases_$ds \
    --device cuda --output models/user_history_embeddings_$ds
done
```

If the HuggingFace cache directory is not writable by your user, point
`HF_HOME` and `SENTENCE_TRANSFORMERS_HOME` at a writable directory before
running this step — `cache_embeddings.py` silently falls back to a
TF-IDF/SVD embedding (128-dim, much weaker) if it cannot load the
sentence-transformer model, so check `embedder/meta.json`'s `"kind"` field
afterward: it should read `"sentence_transformer"`, not a fallback indicator.

### 3. Aspects

```bash
for ds in baby yelp googlelocal; do
  python scripts/extract_aspects.py --cases data/cases_$ds \
    --output data/processed_$ds/aspects
done
```

The canonical aspect vocabulary is built from each platform's own `train`
split only, then applied unchanged (exact-match -> singular-strip ->
head-noun fallback) to canonicalize that platform's `valid` and `test` rows
— never from all three splits pooled together, which would let held-out
reviews' own aspect phrasing influence what counts as a canonical aspect for
their own scoring.

### 4. PEER's target-interest retriever

```bash
for ds in baby yelp googlelocal; do
  python scripts/train_peer_retriever.py --cases data/cases_$ds \
    --embeddings embeddings/embeddings_$ds \
    --user-history-override models/user_history_embeddings_$ds \
    --output models/peer_target_retriever_$ds.pt
done
```

### 5. Features (pairs parquet)

```bash
for ds in baby yelp googlelocal; do
  python scripts/build_features.py --cases data/cases_$ds \
    --aspects data/processed_$ds/aspects/sentence_aspects.parquet \
    --embeddings embeddings/embeddings_$ds \
    --user-history-override models/user_history_embeddings_$ds \
    --peer-retriever models/peer_target_retriever_$ds.pt \
    --semantic-weight 0.70 --aspect-weight 0.15 \
    --sentiment-weight 0.10 --coverage-weight 0.05 \
    --output data/processed_$ds/pairs
done
```

The label weights above are the shipped composite-label configuration.
`--user-history-override` merges PEER's per-sentence-pooled user-history
embeddings into `user_sem_sim` and (via the retriever) `target_emb_sim`
only; every other column, and every baseline that reads embeddings
directly, keeps using the blob encoding. See the memory note under
"Installation" above — run this step for one platform at a time.

### 6. Ranker

```bash
for ds in baby yelp googlelocal; do
  python scripts/train_ltr.py \
    --train data/processed_$ds/pairs/train.parquet \
    --valid data/processed_$ds/pairs/valid.parquet \
    --output models/peer_ltr_$ds.pkl
done
```

Uses `peer/models.py::FEATURE_COLUMNS_DEFAULT` -- PEER's 5 reported features
($\phi,\psi,\rho,\alpha,\beta$: `user_sem_sim`, `item_sem_sim`,
`target_emb_sim`, `user_aspect_overlap`, `item_aspect_salience`) -- with no
`--drop-features`.

### 7. Baselines

Published baselines (PRAG needs its own retriever; NARRE/HRDR need their
own trained scorer) — trained per platform, same as PEER:

```bash
for ds in baby yelp googlelocal; do
  python scripts/train_prag_retriever.py --cases data/cases_$ds \
    --embeddings embeddings/embeddings_$ds --output models/prag_retriever_$ds.pt

  python scripts/train_published_neural.py --model narre --dataset $ds \
    --raw-dir data/raw --cases data/cases_$ds --output models/narre_$ds.pt
  python scripts/train_published_neural.py --model hrdr --dataset $ds \
    --raw-dir data/raw --cases data/cases_$ds --output models/hrdr_$ds.pt
done
```

Selection/inference for every method (also per platform):

```bash
for ds in baby yelp googlelocal; do
  python scripts/select_topk.py --pairs-dir data/processed_$ds/pairs \
    --model models/peer_ltr_$ds.pkl --splits test --method-name peer_full \
    --k-list 1 3 5 user_avg --lambda-coverage 0.10 \
    --embeddings embeddings/embeddings_$ds.npz --output outputs/predictions_$ds/peer

  python scripts/run_baselines.py --pairs-dir data/processed_$ds/pairs \
    --split test --k-list 1 3 5 user_avg \
    --methods random recent popular sbert_user_metadata bm25_user_metadata \
    --embeddings embeddings/embeddings_$ds.npz --output outputs/predictions_$ds/baselines

  python scripts/run_published_baselines.py --cases data/cases_$ds \
    --k-list 1 3 5 user_avg \
    --methods prag erra_r a2spr narre hrdr \
    --models-dir models --dataset $ds --output outputs/predictions_$ds/published
done
```

(`select_topk.py --lambda-coverage 0.10` is the shipped selector value.)
The paper's reported "SBERT" and "BM25" rows are `sbert_user_metadata` /
`bm25_user_metadata` (history + metadata), not the history-only variants.

`run_published_baselines.py` looks up checkpoints as
`{models-dir}/prag_retriever_{dataset}.pt`, `{models-dir}/narre_{dataset}.pt`,
and `{models-dir}/hrdr_{dataset}.pt` — i.e. it expects `--models-dir` to
contain all three platforms' checkpoints together (as produced by the loop
above) and picks out the right one per case via `--dataset`/each case's own
`dataset` field. All three methods print an explicit
`WARNING: no checkpoint...` if a platform's checkpoint is missing (falling
back to all-zero scores for that platform) — always check the log for it
after a run, since a missing checkpoint otherwise fails silently rather than
erroring out.

### 8. Evaluation

```bash
for ds in baby yelp googlelocal; do
  python scripts/evaluate_evidence.py \
    --pred-dir outputs/predictions_$ds \
    --cases data/cases_$ds --embeddings embeddings/embeddings_$ds.npz \
    --output results/evaluation_$ds.csv --per-case-output results/per_case_$ds.csv
done
```

`results/evaluation_$ds.csv`, aggregated by `(dataset, k)` and filtered to
`k=user_avg`, gives that platform's column of Table "main". Build the pooled
column by concatenating the three platforms' `per_case_$ds.csv` files (a
plain `pandas.concat`, matching each platform's equal case count so pooling
is an unweighted average) and re-aggregating; do **not** re-run any pipeline
stage on a pooled/joint split to get this number.

```bash
for ds in baby yelp googlelocal; do
  python scripts/significance_test.py --per-case results/per_case_$ds.csv \
    --main-method peer_full \
    --baselines prag erra_r sbert_user_metadata bm25_user_metadata a2spr \
                recent hrdr popular narre random \
    --output results/significance_$ds.csv
done
# and again on the pooled concatenation for the pooled significance column
python scripts/significance_test.py --per-case results/per_case_pooled.csv \
  --main-method peer_full \
  --baselines prag erra_r sbert_user_metadata bm25_user_metadata a2spr \
              recent hrdr popular narre random \
  --output results/significance_pooled.csv
```
