#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from peer.utils import ensure_dir


def bootstrap_diff(df, main, baseline, metric, k, dataset, n_boot=1000, seed=42):
    sub = df[(df['k'].astype(str) == str(k)) & (df['dataset'] == dataset)]
    a = sub[sub['method'] == main][['case_id', metric]].rename(columns={metric:'main'})
    b = sub[sub['method'] == baseline][['case_id', metric]].rename(columns={metric:'base'})
    m = a.merge(b, on='case_id')
    if len(m) == 0:
        return None
    diff = (m['main'] - m['base']).values
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(diff, size=len(diff), replace=True)
        boots.append(np.mean(sample))
    boots = np.asarray(boots)
    return {
        'dataset': dataset,
        'k': k,
        'metric': metric,
        'main': main,
        'baseline': baseline,
        'n_cases': len(diff),
        'main_mean': float(m['main'].mean()),
        'baseline_mean': float(m['base'].mean()),
        'diff_mean': float(np.mean(diff)),
        'ci_low': float(np.percentile(boots, 2.5)),
        'ci_high': float(np.percentile(boots, 97.5)),
        'p_two_sided': float(2 * min(np.mean(boots <= 0), np.mean(boots >= 0))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-case', default='results/per_case.csv')
    ap.add_argument('--main-method', default='peer_full')
    ap.add_argument('--baselines', nargs='+', default=['sbert_user_metadata','mmr_user_metadata','bm25_user','random'])
    ap.add_argument('--metrics', nargs='+', default=['sem_f1','aspect_f1','noise','redundancy'])
    ap.add_argument('--k-list', nargs='+', default=None)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--output', default='results/significance.csv')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    df = pd.read_csv(args.per_case)
    datasets = sorted(df['dataset'].dropna().unique())
    k_list = args.k_list or sorted(df['k'].astype(str).unique())
    rows = []
    for ds in datasets:
        for k in k_list:
            for metric in args.metrics:
                for base in args.baselines:
                    res = bootstrap_diff(df, args.main_method, base, metric, str(k), ds, args.bootstrap, args.seed)
                    if res:
                        rows.append(res)
    out = pd.DataFrame(rows)
    ensure_dir(Path(args.output).parent)
    out.to_csv(args.output, index=False)
    print(f'Wrote significance results -> {args.output}')
    print(out.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
