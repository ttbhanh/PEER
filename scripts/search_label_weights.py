#!/usr/bin/env python
from __future__ import annotations

"""Composite training-label weight search (paper, Discussion: "The label
search combines 23 broad configurations with a focused five-point
semantic-weight sweep"), run on the VALID split only so the reported test
numbers aren't themselves the product of tuning on them.

semantic_to_gt/aspect_match_to_gt/sentiment_match/coverage_gain are all
pre-computed columns in the pairs parquet (scripts/build_features.py), so
recomputing `label` for a new weight vector is a single in-memory linear
combination -- no need to re-run the retriever/embedding pipeline. The only
real cost per candidate is retraining the LightGBM ranker (label is its
regression target) and running selection+eval on the valid cases, with the
selector's own hyperparameters held fixed at their shipped values
(scripts/select_topk.py defaults) so any downstream difference is
attributable to the label weights alone.

Two-stage protocol matching the paper's claim:
  1. `--semantic-grid 0.5 0.6 0.7 0.8` (default) x 5 aspect:sentiment:coverage
     ratio patterns + 3 reference points = 23 configs (the "broad" search).
  2. A follow-up run with a 5-point `--semantic-grid` centered on the stage-1
     winner (e.g. `--semantic-grid 0.65 0.675 0.70 0.725 0.75`) is the
     "focused five-point semantic-weight sweep".
"""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import time
from pathlib import Path

import pandas as pd

from peer.metrics import aggregate_metrics
from peer.models import RankerModel
from peer.utils import ensure_dir

from scripts.select_topk import LABEL_COMPONENT_COLUMNS, load_embeddings, parse_k_list, read_pairs_columns
from scripts.tune_selector import build_case_inputs, load_case_gt, run_config

RATIO_PATTERNS = {
    'orig_3_2_1': (3, 2, 1),  # shipped ratio: aspect > sentiment > coverage
    'equal': (1, 1, 1),
    'aspect_heavy': (3, 1, 1),
    'sentiment_heavy': (1, 3, 1),
    'coverage_heavy': (1, 1, 3),
}


def compute_label(df: pd.DataFrame, w: dict) -> pd.Series:
    return (w['semantic'] * df['semantic_to_gt'] + w['aspect'] * df['aspect_match_to_gt']
            + w['sentiment'] * df['sentiment_match'] + w['coverage'] * df['coverage_gain'])


def make_configs(semantic_grid: list[float]) -> list[dict]:
    configs = []
    for sem in semantic_grid:
        rest = 1.0 - sem
        for name, (a, s, c) in RATIO_PATTERNS.items():
            tot = a + s + c
            configs.append({
                'name': f'sem{sem}_{name}',
                'semantic': sem, 'aspect': rest * a / tot,
                'sentiment': rest * s / tot, 'coverage': rest * c / tot,
            })
    configs.append({'name': 'shipped', 'semantic': 0.70, 'aspect': 0.15, 'sentiment': 0.10, 'coverage': 0.05})
    configs.append({'name': 'equal_weight_baseline', 'semantic': 0.25, 'aspect': 0.25, 'sentiment': 0.25, 'coverage': 0.25})
    configs.append({'name': 'pure_semantic', 'semantic': 1.0, 'aspect': 0.0, 'sentiment': 0.0, 'coverage': 0.0})
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs-dir', default='data/processed/pairs')
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--split', default='valid')
    ap.add_argument('--k-list', nargs='+', default=['1', '3', '5', 'user_avg'])
    ap.add_argument('--backend', default='lightgbm')
    ap.add_argument('--lambda-coverage', type=float, default=0.1,
                     help='Fixed selector hyperparameter, matching the shipped scripts/select_topk.py default')
    ap.add_argument('--allowed-item-top-n', type=int, default=15)
    ap.add_argument('--semantic-grid', nargs='+', type=float, default=[0.5, 0.6, 0.7, 0.8],
                     help='Stage 1 (broad): 4 values x 5 ratio patterns + 3 reference points = 23 configs. '
                          'Stage 2 (focused): pass a 5-point grid centered on the stage-1 winner.')
    ap.add_argument('--output', default='results/label_weight_search.csv')
    args = ap.parse_args()

    print('Loading train/eval splits...')
    train_df = read_pairs_columns(Path(args.pairs_dir) / 'train.parquet',
                                   extra=LABEL_COMPONENT_COLUMNS, include_case_columns=False)
    eval_df = read_pairs_columns(Path(args.pairs_dir) / f'{args.split}.parquet',
                                  extra=LABEL_COMPONENT_COLUMNS)
    embs = load_embeddings(args.embeddings)
    case_gt = load_case_gt(args.cases)
    k_list = parse_k_list(args.k_list)

    for col in LABEL_COMPONENT_COLUMNS:
        assert col in train_df.columns, f'missing {col} -- rerun build_features.py first'

    configs = make_configs(args.semantic_grid)
    print(f'{len(configs)} label-weight configs to evaluate on {args.split} ({eval_df["case_id"].nunique()} cases)')

    results = []
    for i, w in enumerate(configs):
        t0 = time.time()
        train_df['label'] = compute_label(train_df, w)
        eval_df['label'] = compute_label(eval_df, w)

        model = RankerModel(backend=args.backend)
        model.fit(train_df, eval_df)
        eval_df['score'] = model.predict(eval_df)

        cases = build_case_inputs(eval_df, embs, args)
        rows = run_config(cases, case_gt, embs, k_list, args.lambda_coverage)
        agg = pd.DataFrame(aggregate_metrics(rows, ['k']))
        agg_avg = agg.drop(columns=['k', 'n_cases']).mean(numeric_only=True)

        rec = {**{k: v for k, v in w.items() if k != 'name'}, 'name': w['name'],
               'backend': model.backend, 'sec': round(time.time() - t0, 1), **agg_avg.to_dict()}
        ua = agg[agg['k'] == 'user_avg']
        if len(ua):
            rec['sem_f1_uavg'] = float(ua['sem_f1'].iloc[0])
            rec['aspect_f1_uavg'] = float(ua['aspect_f1'].iloc[0])
        results.append(rec)
        print(f'[{i+1}/{len(configs)}] {w["name"]:24s} sem={w["semantic"]:.2f} asp={w["aspect"]:.2f} '
              f'sent={w["sentiment"]:.2f} cov={w["coverage"]:.2f}  '
              f'sem_f1(uavg)={rec.get("sem_f1_uavg", float("nan")):.4f} '
              f'aspect_f1(uavg)={rec.get("aspect_f1_uavg", float("nan")):.4f}  ({rec["sec"]}s)')

        out_df = pd.DataFrame(results)
        ensure_dir(Path(args.output).parent)
        out_df.to_csv(args.output, index=False)

    print(f'\nWrote {len(results)} configs -> {args.output}')
    print('\nTop 5 by sem_f1_uavg:')
    print(pd.DataFrame(results).sort_values('sem_f1_uavg', ascending=False).head(5)[
        ['name', 'semantic', 'aspect', 'sentiment', 'coverage', 'sem_f1_uavg', 'aspect_f1_uavg']
    ].to_string(index=False))


if __name__ == '__main__':
    main()
