from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# metadata_sem_sim / metadata_aspect_overlap deliberately excluded: an ablation
# (results/ablation.csv, no_metadata variant) showed their contribution to
# sem_f1/aspect_f1 was within +-0.003 of peer_full (noise), so the product-
# metadata (title/brand/category) signal is dropped from PEER's own ranker;
# build_features.py still computes those two columns for anyone re-running
# that ablation, they're just no longer in the canonical feature set.
# target_emb_sim: cosine(candidate, estimated target-review embedding) from a
# small PRAG-style retriever conditioned on (user, item) -- see
# scripts/train_peer_retriever.py. Added specifically to close PEER's sem_f1
# gap against the PRAG baseline, whose whole mechanism is ranking by similarity
# to exactly this kind of estimate; here it's one ranker feature among others,
# not the sole ranking criterion.
# cross_encoder_score: a MiniLM cross-encoder (scripts/train_cross_encoder.py)
# fine-tuned to regress max-sentence-level ground-truth similarity, trained on
# hard negatives mined from the ORIGINAL PRAG retriever's own top-ranked-but-
# wrong candidates. build_features.py still computes it (for anyone re-running
# this experiment), but it's deliberately EXCLUDED here: a controlled A/B test
# (same data/weights, this column on vs. off) on the valid split showed it
# *regressed* sem_f1 (0.4200 -> 0.4167) despite slightly improving redundancy --
# the opposite of its purpose. Likely causes, in rough order of suspicion: (a)
# the ranker backend is RandomForest, not LightGBM (LightGBM's macOS wheel needs
# libomp, which isn't loadable here without a system-wide Homebrew path fix
# outside this project's scope -- see git history for scripts/train_ltr.py) --
# RF's greedy per-tree splits may handle this feature's sparsity (populated for
# only the top-50/case candidates, 0 elsewhere) worse than boosting would; (b)
# the training set (~50k pairs, mostly extremes: strong positives + PRAG-hard-
# negatives) may not represent the "ordinary middle" of the candidate
# distribution well enough for the model to rank it reliably at inference time.
# sentiment_match EXCLUDED as of the sentiment-leak fix: scripts/build_features.py
# computes this column as max(sentiment_match(candidate, gt) for gt in
# ground_truth_sentences) -- i.e. it encodes whether the candidate's sentiment
# matches the ACTUAL TARGET REVIEW's sentiment, which only exists for the label
# (composite regression target, train-time only). Unlike semantic_to_gt /
# aspect_match_to_gt (the other two ground-truth-derived label ingredients,
# already correctly excluded here), sentiment_match had also been left in
# FEATURE_COLUMNS_DEFAULT -- a genuine target/label leak into a ranker feature,
# not just the "temporal leakage" (future-review candidates) this project
# otherwise guards carefully against. Measured impact was small (1.6% feature
# importance in the pre-fix model, the lowest of the 10 features) and no other
# baseline reads this column, so the leak wasn't a large driver of PEER's
# reported wins -- but the feature set is fixed here regardless, since the
# correctness issue matters independent of its measured magnitude.
# build_features.py still computes and stores the column (both for the label
# formula, which legitimately needs it, and for anyone auditing the leak).
FEATURE_COLUMNS_DEFAULT = [
    'user_sem_sim', 'item_sem_sim', 'target_emb_sim', 'user_aspect_overlap',
    'item_aspect_salience',
    'helpfulness_norm', 'recency_norm', 'sentence_len_norm', 'aspect_count_norm',
]


def available_feature_columns(df: pd.DataFrame, drop: list[str] | None = None) -> list[str]:
    drop = set(drop or [])
    cols = [c for c in FEATURE_COLUMNS_DEFAULT if c in df.columns and c not in drop]
    if not cols:
        bad = {'case_id', 'sentence_id', 'label', 'split', 'dataset', 'text', 'aspects'}
        cols = [c for c in df.columns if c not in bad and pd.api.types.is_numeric_dtype(df[c])]
    return cols


class RankerModel:
    def __init__(self, backend: str = 'auto', feature_columns: list[str] | None = None):
        self.backend = backend
        self.feature_columns = feature_columns
        self.model: Any = None

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None):
        self.feature_columns = self.feature_columns or available_feature_columns(train_df)
        x = train_df[self.feature_columns].fillna(0.0).values.astype(np.float32)
        y = train_df['label'].values.astype(np.float32)
        if self.backend in {'auto', 'lightgbm'}:
            try:
                import lightgbm as lgb
                params = dict(
                    objective='regression',
                    metric='rmse',
                    learning_rate=0.05,
                    num_leaves=63,
                    n_estimators=600,
                    min_child_samples=20,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=-1,
                )
                self.model = lgb.LGBMRegressor(**params)
                if valid_df is not None and len(valid_df):
                    xv = valid_df[self.feature_columns].fillna(0.0).values.astype(np.float32)
                    yv = valid_df['label'].values.astype(np.float32)
                    self.model.fit(x, y, eval_set=[(xv, yv)], eval_metric='rmse')
                else:
                    self.model.fit(x, y)
                self.backend = 'lightgbm'
                return self
            except Exception:
                if self.backend == 'lightgbm':
                    raise
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1)
        self.model.fit(x, y)
        self.backend = 'sklearn_rf'
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.feature_columns is None:
            raise RuntimeError('Model is not fitted')
        x = df[self.feature_columns].fillna(0.0).values.astype(np.float32)
        return np.asarray(self.model.predict(x), dtype=np.float32)

    def save(self, path: str | Path) -> None:
        joblib.dump({'backend': self.backend, 'feature_columns': self.feature_columns, 'model': self.model}, path)

    @classmethod
    def load(cls, path: str | Path) -> 'RankerModel':
        obj = joblib.load(path)
        m = cls(obj['backend'], obj['feature_columns'])
        m.model = obj['model']
        return m
