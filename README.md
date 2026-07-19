# PEER — Personalized Evidence Extraction from Reviews

PEER selects `k` evidence sentences from an item's _existing_ reviews that best
anticipate the review a specific user will write for that item **before that
review exists**. Every candidate sentence and all user-history signal must
temporally precede the target review's timestamp (no future-review leakage).

PEER is a two-stage architecture: (1) a LightGBM ranker scores every candidate
sentence independently on a composite relevance label, then (2) a greedy
aspect-coverage selector assembles the final `k`-sentence evidence set.

This repository contains the experimental pipeline used in the paper:
case construction, aspect extraction, sentence embeddings, the PEER
ranker/selector, task-adapted reproductions of five published
explainable-recommendation baselines (NARRE, HRDR, A2SPR, ERRA-R, PRAG),
five classical linear-feature baselines, and the evaluation/significance-test
tooling. It does **not** include the paper's raw data, cached embeddings, or
trained model checkpoints (large binary artifacts) — see
["Getting the data"](#2-getting-the-data) below.

> **This release excludes SEER, the single-feature `item_salience` linear
> heuristic, and the five metadata-conditioned linear heuristics
> (BM25-metadata, BM25-user+metadata, SBERT-metadata, SBERT-user+metadata,
> MMR-user+metadata)** that appear in the paper. If you are trying to
> reproduce a table cell involving one of those methods, this codebase will
> not produce it as-is.

**On the published baselines:** NARRE, HRDR, A2SPR, ERRA-R, and PRAG are
implemented here as **task-adapted reimplementations** under PEER's
sentence-selection setting, not bit-exact reproductions of their original
repositories (those solve rating-prediction or generation tasks, not
sentence selection). See `configs/baselines/manifest.json` for the mechanism
each adapter implements and how it differs from the original paper.

## Repository layout

```
peer/                  Core library: ranker features, selector, metrics,
                        aspect extraction, sentiment, embeddings I/O
scripts/                Every pipeline stage as a standalone CLI script
baselines/
  common/               Shared prediction-writing I/O
  published/            NARRE, HRDR (neural/), A2SPR, ERRA-R, PRAG
  llm/                  Optional OpenAI-compatible zero-shot LLM baseline
configs/
  default.yaml          Case-construction / embedding / selector defaults
  baselines/manifest.json  Per-baseline mechanism description + honesty flags
examples/               End-to-end reference pipelines (see below)
tests/                  pytest suite for the published-baseline scorers
```

`data/`, `embeddings/`, `models/`, `outputs/`, `results/` are created by the
pipeline at runtime and are not included in this package (see `.gitignore`).

## 1. Installation

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e .                  # core dependencies only
```

Optional extras (install what you need):

```bash
pip install -e '.[lightgbm]'      # LightGBM ranker backend (falls back to
                                   # scikit-learn HistGradientBoosting if absent)
pip install -e '.[gpu]'           # torch, transformers, sentence-transformers,
                                   # accelerate — needed for real sentence
                                   # embeddings, NARRE/HRDR, and the
                                   # cross-encoder/PRAG-retriever features
pip install -e '.[llm]'           # optional OpenAI-compatible LLM baseline
pip install -e '.[dev]'           # pytest, ruff
```

`requirements.txt` is an alternative flat list covering the same packages
plus `spacy` (used, with a regex fallback, for aspect-phrase extraction) if
you prefer `pip install -r requirements.txt`. For spaCy's English model:
`python -m spacy download en_core_web_sm` (optional — `--method hybrid`
falls back to a regex-only extractor if spaCy isn't installed).

Verify the install:

```bash
pytest -q
```

## 2. Getting the data

PEER was evaluated on three platforms. None of their raw review dumps are
redistributed here — download them yourself and convert to the schema below.

| Platform                            | Source                                                  | Item-ID field |
| ----------------------------------- | ------------------------------------------------------- | ------------- |
| Amazon Reviews 2023 (Baby category) | https://amazon-reviews-2023.github.io/                  | `parent_asin` |
| Yelp Open Dataset                   | https://www.yelp.com/dataset                            | `business_id` |
| Google Local Reviews (2021)         | https://cseweb.ucsd.edu/~jmcauley/datasets/googlelocal/ | `gmap_id`     |

Convert each platform's raw dump into two JSONL files per dataset name
(`<name>_reviews.jsonl`, `<name>_metadata.jsonl`) under `data/raw/`, one
review/listing per line, with (at minimum) these fields:

```json
// data/raw/<name>_reviews.jsonl
{
  "user_id": "U1",
  "<item-id-field>": "I1",
  "rating": 5,
  "text": "...",
  "timestamp": 1710000000000,
  "helpful_vote": 2
}
```

```json
// data/raw/<name>_metadata.jsonl
{
  "<item-id-field>": "I1",
  "title": "...",
  "description": "...",
  "category": "..."
}
```

`timestamp` must be milliseconds since epoch. `scripts/build_cases.py` reads
whichever field names you tell it via `--user-col`/`--item-col`/`--text-col`/
`--rating-col`/`--timestamp-col`/`--helpful-col` (defaults match the Amazon
schema: `user_id`/`parent_asin`/`text`/`rating`/`timestamp`/`helpful_vote`).
For Yelp/Google Local, pass `--item-col business_id` or `--item-col gmap_id`
respectively.

**No data yet?** `scripts/make_demo_data.py` generates small synthetic
reviews so you can smoke-test the whole pipeline without downloading
anything — see the next section.

## 3. Quick start: smoke test on synthetic data

```bash
bash examples/run_demo_full.sh
```

This generates a tiny synthetic dataset, builds temporal cases, extracts
aspects, computes TF-IDF fallback embeddings (no GPU/model download needed),
builds ranker features, trains PEER, runs five classical baselines plus
two published baselines that need no training (A2SPR, ERRA-R), and
writes aggregate metrics to `results/all_results.csv`. Runs in well under a
minute on CPU. Use this to confirm your environment is set up correctly
before pointing the same scripts at real data.

`examples/run_demo.sh` is a similar, slightly smaller variant that also
demonstrates `scripts/run_ablation.py`, `scripts/export_case_studies.py`,
and `scripts/make_tables.py`.

## 4. Full pipeline on real data (single platform)

`examples/run_amazon_3day.sh` is a documented, ready-to-run reference for
one platform (adjust `--datasets`/`--item-col` for Yelp or Google Local).
The stages, in order:

```bash
# 1. Temporal, leakage-free case construction
python scripts/build_cases.py \
  --datasets baby --raw-dir data/raw --output data/cases \
  --min-user-history 3 --min-candidate-reviews 5 \
  --min-candidate-sentences 20 --max-candidate-sentences 300 \
  --max-user-history-sentences 100

# 2. Aspect extraction (hybrid noun-chunk + frequency method)
python scripts/extract_aspects.py --cases data/cases \
  --output data/processed/aspects --method hybrid --min-count 2 --max-vocab 10000

# 3. Sentence embeddings (all-mpnet-base-v2; add --force-tfidf for a
#    no-GPU/no-download fallback)
python scripts/cache_embeddings.py --cases data/cases \
  --model sentence-transformers/all-mpnet-base-v2 \
  --batch-size 1024 --device cuda --fp16 --output embeddings/embeddings.npz

# 4. (Optional but used by the paper's final config) train PEER's own
#    target-embedding retriever and PRAG's retriever, feeding two of the
#    nine ranker features (target_emb_sim, and PRAG's own score)
python scripts/train_peer_retriever.py --cases data/cases \
  --embeddings embeddings/embeddings --output models/peer_target_retriever.pt
python scripts/train_prag_retriever.py --cases data/cases \
  --embeddings embeddings/embeddings --output models/prag_retriever.pt

# 5. Ranker features + composite training label. --semantic-weight/
#    --aspect-weight/--sentiment-weight/--coverage-weight set the label's
#    component weights; see "Reproducing the paper's numbers" below for the
#    exact values behind the reported results.
python scripts/build_features.py --cases data/cases \
  --aspects data/processed/aspects/sentence_aspects.parquet \
  --embeddings embeddings/embeddings.npz \
  --peer-retriever models/peer_target_retriever.pt \
  --output data/processed/pairs

# 6. Classical linear-feature baselines (no training)
python scripts/run_baselines.py --pairs-dir data/processed/pairs --split test \
  --methods random popular recent bm25_user sbert_user \
  --k-list 1 3 5 user_avg --embeddings embeddings/embeddings.npz \
  --output outputs/predictions/baselines

# 7. Published baselines that need no training
python scripts/run_published_baselines.py --cases data/cases --split test \
  --aspects data/processed/aspects/sentence_aspects.parquet \
  --aspect-vocab data/processed/aspects/aspect_vocab.jsonl \
  --embeddings embeddings/embeddings.npz \
  --methods a2spr erra_r prag --models-dir models \
  --k-list 1 3 5 user_avg --output outputs/predictions/published

# 7b. (Optional) NARRE/HRDR need their own rating-regression pretraining
#     first, on a temporally-safe pool of raw reviews:
python scripts/train_published_neural.py --model narre --dataset baby \
  --raw-dir data/raw --cases data/cases --output models/narre_baby.pt
python scripts/train_published_neural.py --model hrdr --dataset baby \
  --raw-dir data/raw --cases data/cases --output models/hrdr_baby.pt
# then re-run step 7 with --methods narre hrdr added

# 8. Train PEER's ranker and select the final k-sentence evidence sets
python scripts/train_ltr.py --train data/processed/pairs/train.parquet \
  --valid data/processed/pairs/valid.parquet --output models/peer_ltr.pkl
python scripts/select_topk.py --pairs-dir data/processed/pairs \
  --model models/peer_ltr.pkl --splits test --method-name peer_full \
  --k-list 1 3 5 user_avg --embeddings embeddings/embeddings.npz \
  --output outputs/predictions/peer

# 9. Evaluate everything and run bootstrap significance tests
python scripts/evaluate_evidence.py --pred-dir outputs/predictions \
  --cases data/cases --embeddings embeddings/embeddings.npz \
  --output results/evaluation.csv --per-case-output results/per_case.csv
python scripts/significance_test.py --per-case results/per_case.csv \
  --main-method peer_full \
  --baselines a2spr bm25_user erra_r hrdr narre prag random sbert_user \
  --metrics sem_f1 aspect_f1 noise redundancy --bootstrap 2000 \
  --output results/significance.csv
```

`scripts/tune_selector.py` and `scripts/search_label_weights.py` reproduce
the paper's hyperparameter searches (selector coverage weight and training
label weights respectively) on the validation split.

## 5. Reproducing the paper's three-platform results

The paper reports Amazon Reviews 2023 (Baby), Yelp Open Dataset, and Google
Local Reviews with **one PEER ranker and one PRAG retriever jointly trained
across all three platforms**, and NARRE/HRDR retrained from scratch on each
platform's own corpus. To reproduce this exactly:

1. Run steps 1–3 above (`build_cases.py` → `extract_aspects.py` →
   `cache_embeddings.py`) **separately for each platform** (`--datasets baby`,
   `--datasets yelp --item-col business_id`,
   `--datasets googlelocal --item-col gmap_id`), each with
   `--max-candidate-sentences 300 --min-user-history 3
--min-candidate-reviews 5 --min-candidate-sentences 20`.
2. Concatenate the three platforms' `cases_{train,valid,test}.jsonl`,
   `sentence_aspects.parquet`, and embeddings (`.npy` + `.ids.json`) files
   into one joint pool before running `build_features.py`/`train_peer_retriever.py`/
   `train_prag_retriever.py`/`train_ltr.py` on the combined pool (steps
   4–5, 8 above) — this is what makes the ranker and retriever _jointly_
   trained rather than per-platform. `select_topk.py` (step 8) and
   `run_published_baselines.py` (step 7) can then be run once on the
   combined test split, or per-platform via their `--dataset` flag if
   memory-constrained.
3. Use the **final label weights** for `build_features.py`:
   `--semantic-weight 0.70 --aspect-weight 0.15 --sentiment-weight 0.10
--coverage-weight 0.05` (this is the outcome of the paper's label-weight
   search, §"Selector Hyperparameter and Label-Weight Search" — see
   `scripts/search_label_weights.py` to re-run that search yourself). The
   selector's own hyperparameters are `select_topk.py`'s defaults
   (`--lambda-coverage 0.1`, all other penalty terms at 0) — nothing else
   needs to be passed.
4. NARRE/HRDR (step 7b) are trained **once per platform**, never shared
   across platforms — their rating-regression objective is inherently
   platform-specific.
5. Run steps 9 (evaluate + significance) once on the combined per-platform
   `per_case.csv`, or separately per platform then concatenate before
   calling `significance_test.py` — it groups by the `dataset` column
   automatically, so a combined `per_case.csv` gives you per-platform _and_
   pooled numbers in one pass (duplicate every row with `dataset` overwritten
   to a constant value, e.g. `"pooled"`, and concatenate that in too, to get
   the pooled figures from the same script).

## 6. Testing

```bash
pytest -q
```

Covers the published-baseline scorers (`baselines/published/scorers.py`) in
isolation from the rest of the pipeline.

## Notes and known limitations

- `peer/models.py`'s `RankerModel` prefers LightGBM and falls back to
  scikit-learn's `HistGradientBoostingRegressor` if LightGBM isn't
  installed or fails to import (e.g. missing OpenMP runtime on some macOS
  setups) — pass `--backend sklearn_hgb` to force the fallback explicitly.
- `scripts/cache_embeddings.py --force-tfidf` produces a deterministic,
  no-download, no-GPU embedding fallback (TF-IDF + truncated SVD) suitable
  for smoke tests; it is not what the paper's reported numbers use.
