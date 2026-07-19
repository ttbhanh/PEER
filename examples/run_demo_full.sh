#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
rm -rf data/cases data/processed embeddings/embeddings.npz outputs/predictions results/*.csv models/*.pkl || true
python scripts/make_demo_data.py --output data/raw --users 12 --items 8 --reviews-per-user 6
python scripts/build_cases.py --raw-dir data/raw --datasets baby cellphone musical --min-user-history 2 --min-candidate-reviews 2 --min-candidate-sentences 5 --max-candidate-sentences 60 --output data/cases
python scripts/extract_aspects.py --cases data/cases --output data/processed/aspects
python scripts/cache_embeddings.py --cases data/cases --force-tfidf --tfidf-components 64 --output embeddings/embeddings.npz
python scripts/build_features.py --cases data/cases --aspects data/processed/aspects/sentence_aspects.parquet --embeddings embeddings/embeddings.npz --output data/processed/pairs
python scripts/run_baselines.py --pairs-dir data/processed/pairs --split test --embeddings embeddings/embeddings.npz --output outputs/predictions/classical
python scripts/run_published_baselines.py --cases data/cases --split test --aspects data/processed/aspects/sentence_aspects.parquet --aspect-vocab data/processed/aspects/aspect_vocab.jsonl --embeddings embeddings/embeddings.npz --methods a2spr erra_r --output outputs/predictions/published
python scripts/train_ltr.py --backend sklearn_hgb --train data/processed/pairs/train.parquet --valid data/processed/pairs/valid.parquet --output models/peer_ltr.pkl
python scripts/select_topk.py --pairs-dir data/processed/pairs --model models/peer_ltr.pkl --splits test --method-name peer_full --embeddings embeddings/embeddings.npz --output outputs/predictions/peer
python scripts/evaluate_evidence.py --pred-dir outputs/predictions --output results/all_results.csv
