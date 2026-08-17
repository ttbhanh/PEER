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
the independent-encoder/extractor robustness check, the oracle upper-bound
analysis, and the LLM case-study tooling. It does **not** include the paper's
raw review data, cached embeddings, or trained model checkpoints (large
binary artifacts) — see ["Getting the data"](#2-getting-the-data) below.

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

**macOS note:** if a process that has imported `torch` later unpickles a
LightGBM model in the same process (e.g. `scripts/personalization_swap_test.py`,
which loads both `RankerModel` and PyTorch retrievers), you may hit a
segfault inside `LGBM_BoosterLoadModelFromString` caused by a duplicate
OpenMP runtime between the two libraries. Work around it by setting
`OMP_NUM_THREADS=1` (and, if that alone isn't enough,
`KMP_DUPLICATE_LIB_OK=TRUE`) in the environment before running such scripts.
This is an environment issue, not a code bug, and does not affect correctness.

## Pipeline overview

**PEER is trained independently per platform** — one ranker, one
target-interest retriever, and one aspect vocabulary per platform (Amazon
Baby / Yelp / Google Local in the paper) — never on a single pooled/joint
split across platforms. This is required for temporal correctness: each raw
platform covers a different real-world calendar range, so pooling
per-platform-relative train/valid/test splits into one joint training set
can let one platform's later-dated training cases temporally precede
another platform's validation/test cases. Training one model per platform
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
    --raw-dir data/raw --output data/cases_$ds --seed 42
done
```

Expects `data/raw/{dataset}_reviews.jsonl` (+ `{dataset}_metadata.jsonl`
where available); `build_cases.py --help` documents the exact filename
patterns it searches for, including `--item-col` (Google Local's raw data
uses `gmap_id` rather than the default `parent_asin`). Source data: Amazon
Reviews 2023 (Baby & Toddler), the Yelp Open Dataset, and Google Local
Reviews.

Sentence splitting keeps sentences of 3-80 words. A review can become a
target case only if the same user has already written more than 3 prior
reviews and there are at least 5 earlier reviews of the target item by other
users, at least 20 resulting candidate sentences, and a nonempty target
review. History is capped in two nested, purely recency-based passes: the
user's own qualifying prior reviews (excluding any review of the target
item) are truncated to the 20 most recent, and their sentences —
concatenated in that same chronological review order — are then truncated
to the last 100 sentences of the concatenation. The candidate pool is capped
similarly: every eligible sentence is sorted by its parent review's
timestamp (descending) and truncated to the top 300.

### 2. Embeddings

Two embedding sets are cached per platform, both keyed by the same
sentence/case ids:

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

If the HuggingFace cache directory is not writable by your user (e.g. a
system-wide cache owned by another account), point `HF_HOME` and
`SENTENCE_TRANSFORMERS_HOME` at a writable directory before running this
step — `cache_embeddings.py` silently falls back to a TF-IDF/SVD embedding
(128-dim, much weaker) if it cannot download/load the sentence-transformer
model, so check `embedder/meta.json`'s `"kind"` field after this step: it
should read `"sentence_transformer"`, not a TF-IDF fallback indicator.

### 3. Aspects

```bash
for ds in baby yelp googlelocal; do
  python scripts/extract_aspects.py --cases data/cases_$ds \
    --output data/processed_$ds/aspects
done
```

**The canonical aspect vocabulary is built from each platform's own
`train` split only**, then applied unchanged (via the same
exact-match -> singular-strip -> head-noun fallback logic) to canonicalize
that platform's `valid` and `test` rows. This is required for validity: an
earlier version of this pipeline built the vocabulary from all three splits
pooled together, which let held-out reviews' own aspect phrasing influence
what counted as a canonical aspect for their own scoring — a train/test
leakage bug now fixed in `extract_aspects.py`.

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

The label weights above are the shipped composite-label configuration
(Table "labelweight"). `--user-history-override` merges PEER's per-
sentence-pooled user-history embeddings into `user_sem_sim` and (via the
retriever) `target_emb_sim` only; every other column, and every baseline
that reads embeddings directly, keeps using the blob encoding.

**Memory note:** this is the most memory-intensive pipeline stage — on a
64GB machine it has been observed to peak within a few hundred MB of the
physical RAM limit for a single platform alone (Yelp's ~20GB embeddings.npy
and Baby's ~15GB embeddings.npy are the largest). **Run this step
sequentially, one platform at a time — do not parallelize it across
platforms** — or you risk OOM. (Earlier, lighter stages such as
`extract_aspects.py`, which is CPU-only text processing, are safe to
parallelize across platforms if you have the cores to spare.)

The composite label combines four raw components that `build_features.py`
also writes out as their own columns (`semantic_to_gt`, `aspect_match_to_gt`,
`sentiment_match`, `coverage_gain`): `label = semantic_weight * semantic_to_gt
+ aspect_weight * aspect_match_to_gt + sentiment_weight * sentiment_match +
coverage_weight * coverage_gain`. Because these components are cached
separately, a label-weight sweep (Table "labelweight") does **not** need to
re-run this expensive step per configuration — see
`scripts/search_label_weights.py` in the secondary-tables section below,
which recombines the cached columns directly.

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
`--drop-features`. `build_features.py` also computes `metadata_sem_sim`
(read directly by the SBERT/BM25 `+metadata` baselines),
`helpfulness_norm`/`recency_norm` (read by the Popular/Recent baselines),
and `sentiment_match` (a composite-label ingredient, would leak the target
review's sentiment if used as a ranker feature); see `peer/models.py`'s
comments for why each of these, plus every other column
`build_features.py` computes, is excluded from the ranker itself.

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

(`select_topk.py --lambda-coverage 0.10` is the shipped selector value;
Table "selectortune" documents its sensitivity.) The paper's reported
"SBERT" and "BM25" rows are `sbert_user_metadata` / `bm25_user_metadata`
(history + metadata), not the history-only variants.

`run_published_baselines.py` looks up checkpoints as
`{models-dir}/prag_retriever_{dataset}.pt`, `{models-dir}/narre_{dataset}.pt`,
and `{models-dir}/hrdr_{dataset}.pt` — i.e. it expects `--models-dir` to
contain all three platforms' checkpoints together (as produced by the loop
above) and picks out the right one per case via `--dataset`/each case's own
`dataset` field. NARRE and HRDR print an explicit `WARNING: no checkpoint...`
if a platform's checkpoint is missing (falling back to all-zero scores for
that platform); PRAG does the same. If you see that warning, evaluation
numbers for that method/platform will be silently near-zero rather than
erroring out — always check the log for it after a run.

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

## Secondary tables and figures

Each of these is independent of the others and can be run in any order once
steps 1-8 above are complete for all three platforms. All of them follow the
same pattern as the main pipeline: run per platform against that platform's
own artifacts, then pool by concatenating/averaging the resulting per-case
or per-config CSVs — never by re-pooling training data.

**Ablation (Table "ablation")**

```bash
for ds in baby yelp googlelocal; do
  python scripts/run_ablation.py --pairs-dir data/processed_$ds/pairs \
    --embeddings embeddings/embeddings_$ds.npz --k-list user_avg \
    --datasets $ds \
    --variants full semantic_only aspect_only minus_phi minus_psi minus_rho \
               minus_alpha minus_beta no_coverage item_only \
    --models-dir models/ablation_$ds --output outputs/predictions_$ds/ablation
  python scripts/evaluate_evidence.py --pred-dir outputs/predictions_$ds/ablation \
    --cases data/cases_$ds --embeddings embeddings/embeddings_$ds.npz \
    --output results/ablation_$ds.csv
done
```

Pass `--datasets $ds` (a single-item list matching the loop) since each
platform trains its own ablation rankers; `run_ablation.py`'s `--datasets`
flag was originally meant to merge multiple platforms' `select_topk.py`
outputs against one *shared* ranker, which no longer applies now that every
platform has its own ranker. To reproduce the paper's validation-split
ablation table specifically (rather than test), add `--splits valid` to the
underlying `select_topk.py` call (edit `run_ablation.py`'s `run()` call or
invoke `train_ltr.py` + `select_topk.py` directly per variant) and evaluate
against `data/processed_$ds/pairs/valid.parquet`-derived predictions.

**Evidence budget + k-budget Pareto (Table "budget", Figure "kbudget-pareto")**

```bash
for ds in baby yelp googlelocal; do
  python scripts/run_k_sensitivity.py --pairs-dir data/processed_$ds/pairs \
    --model models/peer_ltr_$ds.pkl --embeddings embeddings/embeddings_$ds.npz \
    --k-list 1 3 5 7 10 15 20 25 30 35 40 user_avg \
    --output-dir outputs/predictions_$ds/k_sensitivity \
    --results-output results/k_sensitivity_$ds.csv

  python scripts/run_published_baselines.py --cases data/cases_$ds \
    --split test --methods prag erra_r --dataset $ds \
    --k-list 1 3 5 7 10 15 20 25 30 35 40 user_avg \
    --models-dir models --output outputs/predictions_$ds/k_sensitivity
done
python scripts/plot_kbudget_pareto.py --evaluation results/k_sensitivity_pooled.csv \
  --output paper/figures/kbudget_pareto.pdf
```

(`run_k_sensitivity.py` covers PEER + 2 lightweight baselines directly;
merge PRAG/ERRA-R's own predictions into the same directory before
re-running `evaluate_evidence.py`, then concatenate the three platforms'
resulting per-k eval CSVs into `results/k_sensitivity_pooled.csv` before
plotting.)

**Label-weight sensitivity + Pareto (Tables "labelweight"/"selectortune",
Figure "labelweight-pareto")**

`search_label_weights.py` recombines `build_features.py`'s cached label
components directly (see the note at the end of step 5) rather than
re-running feature construction per configuration, so the full 23-config
grid is cheap even though it retrains a ranker per configuration:

```bash
for ds in baby yelp googlelocal; do
  # valid split -- table + selection of the shipped config
  python scripts/search_label_weights.py --pairs-dir data/processed_$ds/pairs \
    --cases data/cases_$ds --embeddings embeddings/embeddings_$ds.npz --split valid \
    --k-list user_avg --output results/label_weight_search_valid_$ds.csv

  # test split -- robustness Pareto figure only, not used for tuning
  python scripts/search_label_weights.py --pairs-dir data/processed_$ds/pairs \
    --cases data/cases_$ds --embeddings embeddings/embeddings_$ds.npz --split test \
    --k-list user_avg --output results/label_weight_search_test_$ds.csv

  python scripts/tune_selector.py --pairs-dir data/processed_$ds/pairs \
    --cases data/cases_$ds --embeddings embeddings/embeddings_$ds.npz --split valid \
    --k-list user_avg --output results/selector_tuning_$ds.csv
done
```

Pool each of the three result CSVs by averaging matching `name`/config rows
across platforms (equal validation/test case counts per platform, so an
unweighted mean), then:

```bash
python scripts/plot_labelweight_pareto.py \
  --search-results results/label_weight_search_test_pooled.csv \
  --prag <PRAG pooled sem_f1> <PRAG pooled aspect_f1> \
  --erra <ERRA-R pooled sem_f1> <ERRA-R pooled aspect_f1> \
  --output paper/figures/labelweight_pareto.pdf
```

**Oracle upper bound (Table "oracle")**

```bash
for ds in baby yelp googlelocal; do
  python scripts/compute_oracle.py --cases data/cases_$ds \
    --aspects data/processed_$ds/aspects/sentence_aspects.parquet \
    --embeddings embeddings/embeddings_$ds \
    --split test --dataset $ds \
    --output results/oracle_$ds.csv
done
```

For every test case, greedily selects `k=user_avg` candidates from that
case's own candidate pool that maximize sem-F1 and aspect-F1 respectively
against the held-out review directly (never as a feature or training
signal for any deployed method) — a `(1-1/e)`-approximate greedy solution to
the underlying submodular coverage objective, so reported values are a
conservative estimate of the true ceiling. Also reports
`recall_pool_aspect`, the fraction of the held-out review's own aspects that
appear anywhere in the candidate pool at all (the ceiling any
content-agnostic aspect metric could reach on this benchmark). Pool by
averaging the three platforms' `oracle_{sem,aspect}_f1`/`recall_pool_aspect`
columns.

**Independent-encoder / independent-extractor robustness (Table
"independent-encoder")**

Re-scores every method's *already-selected* evidence (no reselection, no
retraining) with an encoder/extractor different from the ones used
elsewhere in the pipeline, to test whether the reported gains are tied to
PEER's own sentence encoder or aspect extractor:

```bash
for ds in baby yelp googlelocal; do
  python scripts/circularity_check_bge.py --cases data/cases_$ds \
    --pred-dir outputs/predictions_$ds --split test --k user_avg \
    --methods peer_full prag erra_r sbert_user_metadata a2spr \
              bm25_user_metadata recent hrdr narre popular random \
    --output results/circularity_check_bge_$ds

  python scripts/circularity_check_yake.py --cases data/cases_$ds \
    --pred-dir outputs/predictions_$ds --split test --k user_avg \
    --methods peer_full prag erra_r sbert_user_metadata a2spr \
              bm25_user_metadata recent hrdr narre popular random \
    --output results/circularity_check_yake_$ds
done
```

Each writes a `..._summary.csv` (per-dataset means) and a `..._percase.csv`
(concatenate the three platforms' `_percase.csv` files directly for the
pooled mean/SD and for paired-bootstrap significance testing against the
primary evaluation's per-case numbers). `circularity_check_bge.py` re-encodes
with `BAAI/bge-base-en-v1.5` and is an exact reproduction of the paper's
methodology. `circularity_check_yake.py` re-tags evidence with YAKE (an
independent keyword extractor); **the exact YAKE hyperparameters used to
produce the original numbers were not preserved** — this script uses
YAKE's own defaults (n-gram size <=2, top 10 keywords/sentence, dedup
threshold 0.9) as a clearly-labeled reconstruction and should be expected to
differ slightly (not qualitatively) from previously published values.

**Personalization swap test (Tables "personalization-div/-gain/
-gainrate/-gainmag/-efficiency/-correlation")**

Uses the same per-platform ranker/retriever/aspects from the main pipeline
(no separate training needed) plus a per-platform PRAG retriever:

```bash
for ds in baby yelp googlelocal; do
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

If this segfaults on macOS during `RankerModel.load()` (a `torch`/LightGBM
OpenMP conflict — see the Install section's macOS note above), set
`OMP_NUM_THREADS=1` in the environment before running.

The population (anchor selection, foreign-history sampling) is seeded
(`--seed 42`, default) but the original population-generation script from
which this was derived was lost to an unrelated machine restart mid-project
and reconstructed from the paper's own methodology description; it
reproduces the paper's reported N almost exactly (exact match on one
platform, within single digits on the other two) but is not guaranteed
bit-identical.

**Case study (Table "case-study")**

```bash
python scripts/export_case_studies.py --pred-dir outputs/predictions_$ds \
  --cases data/cases_$ds --output outputs/case_studies
```

The specific case ID(s) featured in the paper's qualitative case study are
chosen by hand for narrative clarity (a non-degenerate candidate pool, a
held-out review with distinctive content, baselines that retrieve
genuinely-on-topic-but-imperfect evidence) from whichever platform's
predictions this exports; re-running the full pipeline from scratch (a
retrained ranker, a rebuilt aspect vocabulary) can change which sentences
PEER selects for any individual case, so a specific case ID from one run of
this pipeline is not guaranteed to illustrate the same narrative point after
a full rebuild. Re-inspect the exported selections before citing a specific
case.

**100-case zero-shot-LLM comparison (Tables "llm-100", "llm-100-sig",
"llm-feature-gap", "llm-feature-gap-subset")**

```bash
python scripts/sample_llm100_cases.py --cases-dir-template 'data/cases_{ds}' \
  --datasets baby yelp googlelocal --n-per-dataset 34 33 33 --seed 42 \
  --output results/llm100_cases.json

python scripts/export_llm100_blind.py --cases results/llm100_cases.json \
  --batch-size 10 --out-dir results/llm100_blind_batches
```

`sample_llm100_cases.py` draws (and saves, for reuse across runs) a
stratified 100-case sample with ground truth included, for scoring only.
`export_llm100_blind.py` splits it into ground-truth-free batch files safe
to hand to an LLM selection process. Two ways to actually run that
selection step:

- **Scripted, via an OpenAI-compatible API:** `scripts/run_llm_baseline.py`
  calls a model directly (`--model`, `--base-url`, `--api-key`) against a
  pairs-parquet-derived candidate pool per case. Fully automated but tied to
  whatever provider/model you configure.
- **Agentic/interactive** (the workflow used for the paper's reported
  numbers): hand each `results/llm100_blind_batches/batch_NN.json` file to
  an LLM-driven selection pass (e.g. an agent tool call per batch, entirely
  blind to ground truth) instructed to write `result_NN.json` in the same
  directory as `[{"case_id": ..., "selected_ids": [...]}, ...]`, matching
  each case's own `k`.

Either way, once every `result_NN.json` exists:

```bash
python scripts/score_llm100.py --cases results/llm100_cases.json \
  --llm-results-dir results/llm100_blind_batches \
  --output results/llm100_results.json
```

(`score_llm100.py`'s defaults already match this README's per-platform path
conventions -- `embeddings/embeddings_{ds}`, `models/peer_target_retriever_{ds}.pt`,
`data/processed_{ds}/aspects/...`, `outputs/predictions_{ds}/peer/...`,
`data/processed_{ds}/pairs/test.parquet`, `results/per_case_{ds}.csv`; pass
the corresponding `--*-template` flag to override any of them if your layout
differs.)

`score_llm100.py` computes sem-F1/aspect-F1 for the LLM's selections,
PEER's own 4 ranker features evaluated on both PEER's and the LLM's picks
(Table "llm-feature-gap"), the same features split by whether the user's
history shares an aspect with the held-out review (Table
"llm-feature-gap-subset"), and PEER/PRAG/ERRA restricted to the identical
100 cases with paired-bootstrap significance of PEER vs. the LLM (Table
"llm-100-sig"). Regardless of which selection path you use, exact
reproduction of the paper's specific numbers is not guaranteed
bit-for-bit — LLM outputs are not deterministic across providers, model
versions, or (for the agentic path) individual runs — but the sampling seed
and saved 100-case file (`results/llm100_cases.json`) are reused so the
*population* being scored is identical every time.

## Known gaps

- **YAKE hyperparameters** used to produce the paper's originally reported
  independent-extractor numbers were not preserved; `circularity_check_yake.py`
  uses YAKE's own defaults as a labeled reconstruction (see above).
- **The personalization swap-test population's exact generation script**
  was lost to an unrelated machine restart and was reconstructed from the
  paper's methodology description; it reproduces the reported N almost
  exactly but is not guaranteed bit-identical (see above).
- **LLM-selection steps** (the case study and the 100-case comparison) are
  inherently non-deterministic across providers/model versions/runs, even
  though the sampling, scoring, and feature-gap-aggregation code is now
  fully preserved (see above) — only the LLM's own selections themselves are
  not guaranteed to reproduce bit-for-bit.
