#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from peer.models import FEATURE_COLUMNS_DEFAULT, RankerModel, available_feature_columns
from peer.utils import ensure_dir


def _read_feature_columns(path: str) -> pd.DataFrame:
    """Read only the numeric feature+label columns RankerModel.fit() needs,
    instead of the full pairs table (which also carries text/aspect columns)."""
    available = set(pq.ParquetFile(path).schema_arrow.names)
    wanted = [c for c in FEATURE_COLUMNS_DEFAULT if c in available]
    if 'label' in available:
        wanted.append('label')
    if not wanted:
        return pd.read_parquet(path)
    return pd.read_parquet(path, columns=wanted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/processed/pairs/train.parquet')
    ap.add_argument('--valid', default='data/processed/pairs/valid.parquet')
    ap.add_argument('--backend', default='auto', choices=['auto','lightgbm','sklearn_hgb'])
    ap.add_argument('--drop-features', nargs='*', default=[])
    ap.add_argument('--output', default='models/peer_ltr.pkl')
    args = ap.parse_args()

    train = _read_feature_columns(args.train)
    valid = _read_feature_columns(args.valid) if Path(args.valid).exists() else None
    feature_cols = available_feature_columns(train, drop=args.drop_features)
    print('Feature columns:', feature_cols)
    model = RankerModel(args.backend, feature_cols).fit(train, valid)
    ensure_dir(Path(args.output).parent)
    model.save(args.output)
    print(f'Saved model -> {args.output}')


if __name__ == '__main__':
    main()
