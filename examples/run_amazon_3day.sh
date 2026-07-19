#!/usr/bin/env bash
set -euo pipefail

# Expected raw files, one pair per dataset:
# data/raw/baby_reviews.jsonl       data/raw/baby_metadata.jsonl
# data/raw/cellphone_reviews.jsonl  data/raw/cellphone_metadata.jsonl
# data/raw/musical_reviews.jsonl    data/raw/musical_metadata.jsonl

DATASETS="baby cellphone musical"

python scripts/build_cases.py \
  --datasets $DATASETS \
  --raw-dir data/raw \
  --output data/cases \
  --min-user-history 3 \
  --min-candidate-reviews 5 \
  --min-candidate-sentences 20 \
  --max-candidate-sentences 300 \
  --max-user-history-sentences 100

python scripts/extract_aspects.py \
  --cases data/cases \
  --output data/processed/aspects \
  --method hybrid \
  --min-count 2 \
  --max-vocab 10000

python scripts/cache_embeddings.py \
  --cases data/cases \
  --model sentence-transformers/all-mpnet-base-v2 \
  --batch-size 1024 \
  --device cuda \
  --fp16 \
  --output embeddings/embeddings.npz

python scripts/build_features.py \
  --cases data/cases \
  --aspects data/processed/aspects/sentence_aspects.parquet \
  --embeddings embeddings/embeddings.npz \
  --output data/processed/pairs

python scripts/run_baselines.py \
  --pairs-dir data/processed/pairs \
  --split test \
  --methods random popular recent bm25_user sbert_user \
  --k-list 1 3 5 user_avg \
  --embeddings embeddings/embeddings.npz \
  --output outputs/predictions/baselines

# a2spr/erra_r need no training; narre/hrdr/prag are skipped here (score
# 0 with a warning if their checkpoints under --models-dir are absent) --
# see scripts/train_published_neural.py and scripts/train_prag_retriever.py
# to train them first if you want the full baseline suite.
python scripts/run_published_baselines.py \
  --cases data/cases \
  --split test \
  --aspects data/processed/aspects/sentence_aspects.parquet \
  --aspect-vocab data/processed/aspects/aspect_vocab.jsonl \
  --embeddings embeddings/embeddings.npz \
  --methods a2spr erra_r \
  --k-list 1 3 5 user_avg \
  --output outputs/predictions/published

python scripts/train_ltr.py \
  --train data/processed/pairs/train.parquet \
  --valid data/processed/pairs/valid.parquet \
  --backend auto \
  --output models/peer_ltr.pkl

python scripts/select_topk.py \
  --pairs-dir data/processed/pairs \
  --model models/peer_ltr.pkl \
  --splits test \
  --method-name peer_full \
  --k-list 1 3 5 user_avg \
  --lambda-coverage 0.2 \
  --mu-redundancy 0.1 \
  --eta-noise 0.1 \
  --embeddings embeddings/embeddings.npz \
  --output outputs/predictions/daper

python scripts/run_ablation.py \
  --variants full no_user no_metadata no_item_salience no_target_emb no_coverage no_sentiment \
  --pairs-dir data/processed/pairs \
  --embeddings embeddings/embeddings.npz \
  --output outputs/predictions/ablation

python scripts/evaluate_evidence.py \
  --pred-dir outputs/predictions \
  --cases data/cases \
  --embeddings embeddings/embeddings.npz \
  --output results/evaluation.csv \
  --per-case-output results/per_case.csv

python scripts/run_k_sensitivity.py \
  --pairs-dir data/processed/pairs \
  --model models/peer_ltr.pkl \
  --embeddings embeddings/embeddings.npz \
  --output-dir outputs/predictions/k_sensitivity \
  --results-output results/k_sensitivity.csv

python scripts/significance_test.py \
  --per-case results/per_case.csv \
  --main-method peer_full \
  --baselines sbert_user bm25_user random \
  --metrics sem_f1 aspect_f1 noise redundancy \
  --bootstrap 1000 \
  --output results/significance.csv

python scripts/export_case_studies.py \
  --per-case results/per_case.csv \
  --pred-dir outputs/predictions \
  --cases data/cases \
  --main-method peer_full \
  --compare-method sbert_user \
  --k user_avg \
  --output outputs/case_studies/case_studies.jsonl

python scripts/make_tables.py --results-dir results --output paper_ready_summary.md
