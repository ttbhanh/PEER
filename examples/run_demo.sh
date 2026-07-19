#!/usr/bin/env bash
set -euo pipefail

python scripts/make_demo_data.py --output data/raw --datasets baby cellphone musical --users 40 --items 15 --reviews-per-user 8
python scripts/build_cases.py --datasets baby cellphone musical --raw-dir data/raw --output data/cases --min-user-history 3 --min-candidate-reviews 2 --min-candidate-sentences 5 --max-candidate-sentences 80 --max-cases-per-dataset 120
python scripts/extract_aspects.py --cases data/cases --output data/processed/aspects --method fallback --min-count 1
python scripts/cache_embeddings.py --cases data/cases --model tfidf-demo --force-tfidf --tfidf-components 32 --batch-size 256 --output embeddings/embeddings.npz
python scripts/build_features.py --cases data/cases --aspects data/processed/aspects/sentence_aspects.parquet --embeddings embeddings/embeddings.npz --output data/processed/pairs
python scripts/run_baselines.py --pairs-dir data/processed/pairs --split test --methods random popular recent bm25_user sbert_user --k-list 1 3 5 user_avg --embeddings embeddings/embeddings.npz --output outputs/predictions/baselines
python scripts/train_ltr.py --train data/processed/pairs/train.parquet --valid data/processed/pairs/valid.parquet --backend sklearn_hgb --output models/peer_ltr.pkl
python scripts/select_topk.py --pairs-dir data/processed/pairs --model models/peer_ltr.pkl --splits test --method-name peer_full --k-list 1 3 5 user_avg --embeddings embeddings/embeddings.npz --output outputs/predictions/daper
python scripts/run_ablation.py --backend sklearn_hgb --variants full no_user no_metadata no_coverage --pairs-dir data/processed/pairs --embeddings embeddings/embeddings.npz --output outputs/predictions/ablation
python scripts/evaluate_evidence.py --pred-dir outputs/predictions --cases data/cases --embeddings embeddings/embeddings.npz --output results/evaluation.csv --per-case-output results/per_case.csv
python scripts/significance_test.py --per-case results/per_case.csv --main-method peer_full --baselines sbert_user bm25_user random --output results/significance.csv || true
python scripts/export_case_studies.py --per-case results/per_case.csv --pred-dir outputs/predictions --cases data/cases --main-method peer_full --compare-method sbert_user --k user_avg --output outputs/case_studies/case_studies.jsonl || true
python scripts/make_tables.py --results-dir results --output paper_ready_summary.md
