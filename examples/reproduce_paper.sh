#!/usr/bin/env bash
set -euo pipefail

# Reproduces the paper's reported numbers: one PEER ranker and one PRAG
# retriever jointly trained across all three platforms (Amazon Reviews 2023
# Baby, Yelp Open Dataset, Google Local Reviews), NARRE/HRDR retrained
# per-platform, evaluated on the pooled 4,800-case test split.
#
# Expected raw files (download yourself, see README "Getting the data"):
#   data/raw/baby_reviews.jsonl        data/raw/baby_metadata.jsonl
#   data/raw/yelp_reviews.jsonl        data/raw/yelp_metadata.jsonl
#   data/raw/googlelocal_reviews.jsonl data/raw/googlelocal_metadata.jsonl
#
# Run from the repository root: bash examples/reproduce_paper.sh

# ---------------------------------------------------------------------------
# Stage 1: temporal case construction, once per platform (each has a
# different raw item-id field), into per-platform staging directories.
# ---------------------------------------------------------------------------
CASE_ARGS="--min-user-history 3 --min-candidate-reviews 5 --min-candidate-sentences 20 \
  --max-candidate-sentences 300 --max-user-history-sentences 100 --max-cases-per-dataset 8000"

python scripts/build_cases.py --datasets baby --item-col parent_asin \
  --raw-dir data/raw --output data/cases_baby $CASE_ARGS
python scripts/build_cases.py --datasets yelp --item-col business_id \
  --raw-dir data/raw --output data/cases_yelp $CASE_ARGS
python scripts/build_cases.py --datasets googlelocal --item-col gmap_id \
  --raw-dir data/raw --output data/cases_googlelocal $CASE_ARGS

# Concatenate into one combined pool: this is what makes the ranker, PEER's
# target retriever, and PRAG's retriever *jointly* trained across platforms
# rather than per-platform (only NARRE/HRDR, trained separately below, are
# platform-specific, since their rating-prediction objective inherently is).
mkdir -p data/cases
for split in train valid test; do
  cat data/cases_baby/cases_${split}.jsonl data/cases_yelp/cases_${split}.jsonl \
      data/cases_googlelocal/cases_${split}.jsonl > data/cases/cases_${split}.jsonl
done

# ---------------------------------------------------------------------------
# Stage 2: aspect extraction + sentence embeddings on the combined pool.
# ---------------------------------------------------------------------------
python scripts/extract_aspects.py --cases data/cases \
  --output data/processed/aspects --method hybrid --min-count 2 --max-vocab 10000

python scripts/cache_embeddings.py --cases data/cases \
  --model sentence-transformers/all-mpnet-base-v2 \
  --batch-size 1024 --device cuda --fp16 --output embeddings/embeddings.npz

# ---------------------------------------------------------------------------
# Stage 3: small trained retrievers (PEER's own target-embedding retriever,
# and PRAG's retriever), then ranker features + composite training label.
# ---------------------------------------------------------------------------
python scripts/train_peer_retriever.py --cases data/cases \
  --embeddings embeddings/embeddings --output models/peer_target_retriever.pt
python scripts/train_prag_retriever.py --cases data/cases \
  --embeddings embeddings/embeddings --output models/prag_retriever.pt

# --semantic-weight/--aspect-weight/--sentiment-weight/--coverage-weight
# default to the paper's final label weights (0.70/0.15/0.10/0.05); pass
# different values to re-run scripts/search_label_weights.py's search.
python scripts/build_features.py --cases data/cases \
  --aspects data/processed/aspects/sentence_aspects.parquet \
  --embeddings embeddings/embeddings.npz \
  --peer-retriever models/peer_target_retriever.pt \
  --output data/processed/pairs

# ---------------------------------------------------------------------------
# Stage 4: baselines. NARRE/HRDR need real per-platform training first
# (rating-regression on a temporally-safe, k-core-filtered review pool).
# ---------------------------------------------------------------------------
python scripts/run_baselines.py --pairs-dir data/processed/pairs --split test \
  --k-list 1 3 5 user_avg --embeddings embeddings/embeddings.npz \
  --output outputs/predictions/baselines

python scripts/train_published_neural.py --model narre --dataset baby --item-col parent_asin \
  --raw-dir data/raw --cases data/cases --output models/narre_baby.pt
python scripts/train_published_neural.py --model narre --dataset yelp --item-col business_id \
  --raw-dir data/raw --cases data/cases --output models/narre_yelp.pt
python scripts/train_published_neural.py --model narre --dataset googlelocal --item-col gmap_id \
  --raw-dir data/raw --cases data/cases --output models/narre_googlelocal.pt
python scripts/train_published_neural.py --model hrdr --dataset baby --item-col parent_asin \
  --raw-dir data/raw --cases data/cases --output models/hrdr_baby.pt
python scripts/train_published_neural.py --model hrdr --dataset yelp --item-col business_id \
  --raw-dir data/raw --cases data/cases --output models/hrdr_yelp.pt
python scripts/train_published_neural.py --model hrdr --dataset googlelocal --item-col gmap_id \
  --raw-dir data/raw --cases data/cases --output models/hrdr_googlelocal.pt

python scripts/run_published_baselines.py --cases data/cases --split test \
  --aspects data/processed/aspects/sentence_aspects.parquet \
  --aspect-vocab data/processed/aspects/aspect_vocab.jsonl \
  --embeddings embeddings/embeddings.npz \
  --methods narre hrdr a2spr erra_r prag --models-dir models \
  --k-list 1 3 5 user_avg --output outputs/predictions/published

# ---------------------------------------------------------------------------
# Stage 5: train PEER's own ranker and select the final k-sentence evidence
# sets (lambda_coverage=0.1, all other selector penalty terms at 0 -- the
# paper's shipped configuration, scripts/select_topk.py defaults).
# ---------------------------------------------------------------------------
python scripts/train_ltr.py --train data/processed/pairs/train.parquet \
  --valid data/processed/pairs/valid.parquet --output models/peer_ltr.pkl
python scripts/select_topk.py --pairs-dir data/processed/pairs \
  --model models/peer_ltr.pkl --splits test --method-name peer_full \
  --k-list 1 3 5 user_avg --embeddings embeddings/embeddings.npz \
  --output outputs/predictions/peer

# ---------------------------------------------------------------------------
# Stage 6: ablation + evidence-budget sensitivity + evaluation + significance.
# ---------------------------------------------------------------------------
python scripts/run_ablation.py --pairs-dir data/processed/pairs \
  --embeddings embeddings/embeddings.npz --output outputs/predictions/ablation

python scripts/run_k_sensitivity.py --pairs-dir data/processed/pairs \
  --model models/peer_ltr.pkl --cases data/cases --embeddings embeddings/embeddings.npz \
  --output-dir outputs/predictions/k_sensitivity --results-output results/k_sensitivity.csv

python scripts/evaluate_evidence.py --pred-dir outputs/predictions \
  --cases data/cases --embeddings embeddings/embeddings.npz \
  --output results/evaluation.csv --per-case-output results/per_case.csv

python scripts/significance_test.py --per-case results/per_case.csv \
  --main-method peer_full \
  --baselines prag erra_r a2spr bm25_user hrdr narre random \
  --metrics sem_f1 aspect_f1 noise redundancy --bootstrap 2000 \
  --output results/significance.csv

# ---------------------------------------------------------------------------
# Stage 7 (optional): independent-encoder/extractor robustness check, and
# illustrative-case export.
# ---------------------------------------------------------------------------
python scripts/independent_check.py semantic --pred-dir outputs/predictions --cases data/cases
python scripts/independent_check.py aspect --pred-dir outputs/predictions --cases data/cases

python scripts/export_case_studies.py --per-case results/per_case.csv \
  --pred-dir outputs/predictions --cases data/cases \
  --main-method peer_full --compare-method prag --k user_avg \
  --output outputs/case_studies/case_studies.jsonl

python scripts/make_tables.py --results-dir results --output paper_ready_summary.md
