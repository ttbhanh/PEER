# PEER — Personalized Evidence Extraction from Reviews

PEER selects `k` evidence sentences from an item's *existing* reviews that
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
examples/                 End-to-end reference pipeline (see below)
tests/                    pytest suite for the core library
```

`data/`, `embeddings/`, `models/`, `outputs/`, `results/` are created by the
pipeline at runtime and are not included in this package.

## 1. Installation

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e .                  # core dependencies only
```

Optional extras (install what you need):

```bash
pip install -e '.[gpu]'                # torch, transformers, sentence-transformers,
                                        # accelerate -- needed for real sentence
                                        # embeddings, NARRE/HRDR, and the
                                        # PEER/PRAG retriever training
pip install -e '.[lightgbm]'           # LightGBM ranker backend (falls back to
                                        # scikit-learn RandomForest if absent)
pip install -e '.[aspects]'            # spaCy, for the aspect-extraction pipeline
pip install -e '.[independent-check]'  # YAKE, only for scripts/independent_check.py aspect
pip install -e '.[dev]'                # pytest
```

`requirements.txt` is an alternative flat list covering the same packages if
you prefer `pip install -r requirements.txt`. For spaCy's English model:
`python -m spacy download en_core_web_sm` (optional — the aspect extractor
falls back to a regex/frequency-based method if spaCy isn't installed).

Verify the install:

```bash
pytest -q
```

## 2. Getting the data

The paper evaluates three platforms. None of their raw review dumps are
redistributed here — download them yourself and convert to the schema below.

| Platform | Source | Item-ID field |
|---|---|---|
| Amazon Reviews 2023 (Baby category) | https://amazon-reviews-2023.github.io/ | `parent_asin` |
| Yelp Open Dataset | https://business.yelp.com/data/resources/open-dataset/ | `business_id` |
| Google Local Reviews (2021) | https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/ | `gmap_id` |

Convert each platform's raw dump into two JSONL files per dataset name
(`<name>_reviews.jsonl`, `<name>_metadata.jsonl`) under `data/raw/`, one
review/listing per line, with (at minimum) these fields:

```json
// data/raw/<name>_reviews.jsonl
{"user_id": "U1", "<item-id-field>": "I1", "rating": 5, "text": "...", "timestamp": 1710000000000, "helpful_vote": 2}
```

```json
// data/raw/<name>_metadata.jsonl
{"<item-id-field>": "I1", "title": "...", "description": "...", "category": "..."}
```

`timestamp` must be milliseconds since epoch. `scripts/build_cases.py` reads
whichever field names you tell it via `--user-col`/`--item-col`/`--text-col`/
`--rating-col`/`--timestamp-col`/`--helpful-col` (defaults match the Amazon
schema).

## 3. Reproducing the paper's numbers

```bash
bash examples/reproduce_paper.sh
```

This is a single, documented, ready-to-run script implementing every stage in
order: temporal case construction per platform → concatenation into one
joint pool → aspect extraction → sentence embeddings → PEER's target
retriever + PRAG's retriever (both jointly trained across all three
platforms) → ranker features/label → 5 heuristic baselines → NARRE/HRDR
training (per-platform, since their rating-prediction objective is inherently
platform-specific) → all 5 published baselines → PEER ranker training →
final evidence selection → grouped ablation → evidence-budget sensitivity →
evaluation → bootstrap significance → independent-encoder/extractor
robustness check → illustrative-case export.

Read the script before running it — it takes hours on real data (embedding
generation and NARRE/HRDR training are the slow stages) and expects the raw
files described above under `data/raw/`.

### Exact configuration behind the reported numbers

These are the actual values used, not just script defaults reproduced by
accident — read this section before changing any flag if you want numbers
that match the paper.

- **Case construction** (`scripts/build_cases.py`): `--min-user-history 3
  --min-candidate-reviews 5 --min-candidate-sentences 20
  --max-candidate-sentences 300 --max-user-history-sentences 100
  --max-cases-per-dataset 8000`, then chronologically split 70/10/20 per
  platform (identical 5,600/800/1,600 train/valid/test on each of the three
  platforms, 4,800 pooled test cases).
- **Encoder**: `sentence-transformers/all-mpnet-base-v2` for every
  ranker feature, the composite training label, and the primary sem-F1
  metric.
- **Composite training label weights** (`scripts/build_features.py`):
  semantic=0.70, aspect=0.15, sentiment=0.10, coverage=0.05 — the outcome of
  the label-weight search described in the paper's Discussion section
  ("23 broad configurations" + "a focused five-point semantic-weight sweep");
  see `scripts/search_label_weights.py` to re-run that search.
- **Selector hyperparameters** (`scripts/select_topk.py`):
  `lambda_coverage=0.1`, `mu_redundancy=eta_noise=nu_aspect_repeat=0.0` — the
  paper's noise/redundancy penalty terms are tuned to zero in the final
  configuration (Discussion: PEER optimizes purely for utility + aspect
  coverage rather than trading sem-F1/aspect-F1 for noise/redundancy). See
  `scripts/tune_selector.py` for the 15-point `lambda_coverage` sweep.
- **Ranker**: LightGBM, `learning_rate=0.05, num_leaves=63,
  n_estimators=600, min_child_samples=20, subsample=colsample_bytree=0.9,
  random_state=42` (`peer/models.py`).
- **Ranker features** (9, `peer/models.py::FEATURE_COLUMNS_DEFAULT`):
  `user_sem_sim, item_sem_sim, target_emb_sim, user_aspect_overlap,
  item_aspect_salience, helpfulness_norm, recency_norm, sentence_len_norm,
  aspect_count_norm` — exactly the paper's φ,ψ,ρ,α,β,γ,δ,η,κ.
- **Significance testing**: two-sided paired bootstrap, 2,000 resamples,
  `scripts/significance_test.py`, against PRAG/ERRA-R/A2SPR/BM25-user/HRDR/
  NARRE/Random, per platform and pooled.
- **Seeds**: 42 throughout (case sampling, ranker training, retriever
  training, NARRE/HRDR training, bootstrap resampling).

## 4. Independent-encoder / independent-extractor robustness check

The paper's Discussion section addresses a real circularity risk: PEER's own
ranker features and composite training label share the encoder used for
sem-F1 (`all-mpnet-base-v2`) and the aspect extractor used for aspect-F1
(hybrid spaCy noun-chunking). `scripts/independent_check.py` re-scores every
method's *already-selected* evidence (no re-training, no re-selection)
against two independent measurements never used anywhere in PEER's own
features/label/selector:

```bash
python scripts/independent_check.py semantic \
  --pred-dir outputs/predictions --cases data/cases \
  --output results/independent_semantic_per_case.csv
python scripts/independent_check.py aspect \
  --pred-dir outputs/predictions --cases data/cases \
  --output results/independent_aspect_per_case.csv
```

`semantic` uses `BAAI/bge-base-en-v1.5` (different base architecture and
training recipe from `all-mpnet-base-v2`, no shared checkpoint lineage) to
recompute sem-F1. `aspect` uses YAKE (unsupervised, statistical, no parsing,
no corpus-level vocabulary — the opposite design axis from the shared hybrid
spaCy pipeline) to recompute aspect-F1. Both subcommands also write a paired
bootstrap significance table (PEER vs. each baseline, per platform and
pooled, 2,000 resamples — same protocol as `significance_test.py`).

## 5. LLM case study

The paper's qualitative comparison against general-purpose LLMs
(Sec. "Case Study: PEER vs. General-Purpose LLMs") is a manual, zero-shot
protocol — each LLM is queried independently through its normal chat
interface with a fixed prompt, not a programmatic API loop, so the release
doesn't pin the exact protocol to one vendor's SDK.

```bash
# 1. Export the exact prompt (user history + numbered, ID-tagged candidate
#    pool + instructions) for a given case:
python scripts/llm_case_study.py export --case-id baby_4418 \
  --cases data/cases --output outputs/case_studies/prompt_baby_4418.txt

# 2. Paste the prompt into any LLM chat interface. It returns a JSON array
#    of chosen sentence IDs, e.g. ["baby_r3162241_s7", ...].

# 3. Score the returned selection against the held-out review:
python scripts/llm_case_study.py score --case-id baby_4418 \
  --cases data/cases --embeddings embeddings/embeddings.npz \
  --response '["baby_r3162241_s7", "baby_r924024_s1"]'
```

This is disclosed in the paper as a qualitative illustration (one case, no
significance claim), not an aggregate benchmark — see Discussion/Limitations.

## 6. Testing

```bash
pytest -q
```

Covers the core library (`peer/selectors.py`, `peer/aspects.py`,
`peer/metrics.py`, `peer/sentiment.py`, `peer/aspect_sentiment.py`) in
isolation from the data pipeline.

## Notes and known limitations

- `peer/models.py`'s `RankerModel` prefers LightGBM and falls back to
  scikit-learn's `RandomForestRegressor` if LightGBM isn't installed or
  fails to import (e.g. missing OpenMP runtime on some macOS setups) — pass
  `--backend sklearn_rf` to force the fallback explicitly. The paper's
  reported numbers use the LightGBM backend.
- `scripts/cache_embeddings.py --force-tfidf` produces a deterministic,
  no-download, no-GPU embedding fallback (TF-IDF + truncated SVD) suitable
  for a smoke test of the pipeline's plumbing; it is not what the paper's
  reported numbers use.
- NARRE/HRDR are retrained **once per platform** (never shared across
  platforms), since their rating-regression objective is inherently
  platform-specific — see `scripts/train_published_neural.py`.
- Combining LightGBM and PyTorch (e.g. a script that both scores with the
  ranker and calls the target retriever in a tight loop) has been observed to
  segfault intermittently on some environments, most likely an OpenMP
  threading conflict. If you hit this, split the two library's usage into
  separate process invocations (compute/pickle the torch-side outputs first,
  then load and score with lightgbm in a second process that never imports
  torch).
