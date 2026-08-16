#!/usr/bin/env python
from __future__ import annotations

"""Systematic search over the composite training-label weights (semantic/
aspect/sentiment/coverage), run on the VALID split only. Recomputing `label`
for a new weight vector is a cheap linear combination of pre-computed
columns; the real cost per config is retraining the ranker and re-running
selection+eval with the selector's own hyperparameters held fixed."""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from peer.metrics import aggregate_metrics
from peer.models import FEATURE_COLUMNS_DEFAULT, RankerModel
from peer.utils import ensure_dir

from scripts.select_topk import LABEL_COMPONENT_COLUMNS, load_embeddings, parse_k_list, read_pairs_columns
from scripts.tune_selector import build_case_inputs, load_case_gt, run_config


def compute_label(df: pd.DataFrame, w: dict) -> pd.Series:
    return (w['semantic'] * df['semantic_to_gt'] + w['aspect'] * df['aspect_match_to_gt']
            + w['sentiment'] * df['sentiment_match'] + w['coverage'] * df['coverage_gain'])


def make_configs() -> list[dict]:
    semantic_grid = [0.5, 0.6, 0.7, 0.8]
    ratio_patterns = {
        'orig_3_2_1': (3, 2, 1),      # shipped ratio: aspect > sentiment > coverage
        'equal': (1, 1, 1),
        'aspect_heavy': (3, 1, 1),
        'sentiment_heavy': (1, 3, 1),
        'coverage_heavy': (1, 1, 3),
    }
    configs = []
    for sem in semantic_grid:
        rest = 1.0 - sem
        for name, (a, s, c) in ratio_patterns.items():
            tot = a + s + c
            configs.append({
                'name': f'sem{sem}_{name}',
                'semantic': sem, 'aspect': rest * a / tot,
                'sentiment': rest * s / tot, 'coverage': rest * c / tot,
            })
    configs.append({'name': 'shipped_manual', 'semantic': 0.70, 'aspect': 0.15, 'sentiment': 0.10, 'coverage': 0.05})
    configs.append({'name': 'original_pre_semweight', 'semantic': 0.40, 'aspect': 0.30, 'sentiment': 0.20, 'coverage': 0.10})
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
    ap.add_argument('--drop-features', nargs='*', default=[],
                     help='Feature columns to exclude from the ranker for every retrain in the grid (e.g. the context group).')
    ap.add_argument('--lambda-coverage', type=float, default=0.1)
    ap.add_argument('--mu-redundancy', type=float, default=0.0)
    ap.add_argument('--eta-noise', type=float, default=0.0)
    ap.add_argument('--nu-aspect-repeat', type=float, default=0.0)
    ap.add_argument('--gate-percentile', type=float, default=0.0)
    ap.add_argument('--demand-user-weight', type=float, default=0.0)
    ap.add_argument('--demand-item-weight', type=float, default=0.0)
    ap.add_argument('--demand-min-weight', type=float, default=0.15)
    ap.add_argument('--max-per-aspect', default=3)
    ap.add_argument('--allowed-item-top-n', type=int, default=15)
    ap.add_argument('--output', default='results/label_weight_search.csv')
    args = ap.parse_args()

    print('Loading train/eval splits...')
    train_df = read_pairs_columns(Path(args.pairs_dir) / 'train.parquet',
                                   extra=LABEL_COMPONENT_COLUMNS, include_case_columns=False)
    eval_df = read_pairs_columns(Path(args.pairs_dir) / f'{args.split}.parquet',
                                  extra=LABEL_COMPONENT_COLUMNS)
    case_gt = load_case_gt(args.cases)
    needed_ids = set(eval_df['sentence_id'].astype(str))
    for case_id in eval_df['case_id'].unique():
        gt_info = case_gt.get(case_id)
        if gt_info:
            needed_ids.update(gt_info['gt_sentence_ids'])
    embs = load_embeddings(args.embeddings, needed_ids)
    k_list = parse_k_list(args.k_list)

    for col in ['semantic_to_gt', 'aspect_match_to_gt', 'sentiment_match', 'coverage_gain']:
        assert col in train_df.columns, f'missing {col} -- rerun build_features.py with the coverage_gain patch first'

    selector_cfg = dict(
        gate_percentile=args.gate_percentile,
        max_per_aspect=None if str(args.max_per_aspect).lower() in {'none', 'null'} else int(args.max_per_aspect),
        demand_min_weight=args.demand_min_weight,
        nu_aspect_repeat=args.nu_aspect_repeat,
        lambda_coverage=args.lambda_coverage,
        mu_redundancy=args.mu_redundancy,
        eta_noise=args.eta_noise,
    )

    configs = make_configs()
    print(f'{len(configs)} label-weight configs to evaluate on {args.split} '
          f'({eval_df["case_id"].nunique()} cases)')

    results = []
    for i, w in enumerate(configs):
        t0 = time.time()
        train_df['label'] = compute_label(train_df, w)
        eval_df['label'] = compute_label(eval_df, w)

        feature_cols = [c for c in FEATURE_COLUMNS_DEFAULT if c not in args.drop_features]
        model = RankerModel(backend=args.backend, feature_columns=feature_cols)
        model.fit(train_df, eval_df)
        eval_df['score'] = model.predict(eval_df)

        cases = build_case_inputs(eval_df, embs, args)
        rows = run_config(cases, case_gt, embs, k_list, selector_cfg)
        agg = pd.DataFrame(aggregate_metrics(rows, ['k']))
        agg_avg = agg.drop(columns=['k', 'n_cases']).mean(numeric_only=True)

        rec = {**{k: v for k, v in w.items() if k != 'name'}, 'name': w['name'],
               'backend': model.backend, 'sec': round(time.time() - t0, 1),
               **agg_avg.to_dict()}
        ua = agg[agg['k'] == 'user_avg']
        if len(ua):
            rec['sem_f1_uavg'] = float(ua['sem_f1'].iloc[0])
            rec['aspect_f1_uavg'] = float(ua['aspect_f1'].iloc[0])
        results.append(rec)
        print(f'[{i+1}/{len(configs)}] {w["name"]:28s} sem={w["semantic"]:.2f} asp={w["aspect"]:.2f} '
              f'sent={w["sentiment"]:.2f} cov={w["coverage"]:.2f}  '
              f'sem_f1(uavg)={rec.get("sem_f1_uavg", float("nan")):.4f} '
              f'aspect_f1(uavg)={rec.get("aspect_f1_uavg", float("nan")):.4f}  ({rec["sec"]}s)')

        out_df = pd.DataFrame(results)
        ensure_dir(Path(args.output).parent)
        out_df.to_csv(args.output, index=False)

    print(f'\nWrote {len(results)} configs -> {args.output}')
    out_df = pd.DataFrame(results)
    print('\nTop 5 by sem_f1_uavg:')
    print(out_df.sort_values('sem_f1_uavg', ascending=False).head(5)[
        ['name', 'semantic', 'aspect', 'sentiment', 'coverage', 'sem_f1_uavg', 'aspect_f1_uavg']
    ].to_string(index=False))


if __name__ == '__main__':
    main()
