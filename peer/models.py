from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# The 9 ranker features described in the paper (semantic x3, aspect x2,
# context x4). Product-metadata and cross-encoder-reranker variants were
# tried during development and are not part of the reported model.
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
        # Fallback if LightGBM is unavailable (e.g. missing OpenMP runtime on
        # some macOS setups). Not what the paper's reported numbers use.
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
