#!/usr/bin/env python
from __future__ import annotations

"""Grid search + Pareto selection for greedy_coverage_select's hyperparameters,
run on the VALID split only. For each config, computes mean sem_f1/aspect_f1/
noise/redundancy (k in {3,5,user_avg}), finds the Pareto frontier over those
4 objectives, then among Pareto-optimal configs picks the one maximizing
F_beta(aspect_p, aspect_r; beta=1.5), subject to --noise-ceiling/--sem-f1-floor."""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from peer.aspect_demand import build_demand_from_weights
from peer.aspects import aspect_f1 as aspect_f1_fn
from peer.metrics import aggregate_metrics, aspect_noise, redundancy, semantic_prf
from peer.models import RankerModel
from peer.selectors import greedy_coverage_select, precision_gate, resolve_k
from peer.utils import read_jsonl, resolve_cases_path

from scripts.select_topk import group_to_candidates, load_embeddings, parse_k_list, read_pairs_columns, to_list


def load_case_gt(cases_dir: str):
    m = {}
    for split in ['train', 'valid', 'test']:
        p = resolve_cases_path(cases_dir, split)
        if p is None:
            continue
        for c in read_jsonl(p):
            m[c['case_id']] = {
                'gt_sentence_ids': [f"{c['case_id']}_gt_{i}" for i in range(len(c['ground_truth_sentences']))],
                'ground_truth_text': c['ground_truth_text'],
            }
    return m


def eval_selection(case_id, dataset, k, selected, gt_aspects, user_aspects, embs, gt_info):
    selected_ids = [s['sentence_id'] for s in selected]
    selected_texts = [s['text'] for s in selected]
    selected_aspects = list(dict.fromkeys(a for s in selected for a in to_list(s.get('aspects'))))
    gt_ids = gt_info.get('gt_sentence_ids', [])
    gt_text = gt_info.get('ground_truth_text', '')
    e_emb = np.asarray([embs[sid] for sid in selected_ids if sid in embs], dtype=np.float32)
    g_emb = np.asarray([embs[sid] for sid in gt_ids if sid in embs], dtype=np.float32)
    sem_p = sem_r = sem_f1 = 0.0
    if len(e_emb) and len(g_emb):
        sem_p, sem_r, sem_f1 = semantic_prf(e_emb, g_emb)
    asp_p, asp_r, asp_f1 = aspect_f1_fn(selected_aspects, gt_aspects)
    pred_text = ' '.join(selected_texts)
    return {
        'dataset': dataset, 'k': str(k),
        'sem_p': sem_p, 'sem_r': sem_r, 'sem_f1': sem_f1,
        'aspect_p': asp_p, 'aspect_r': asp_r, 'aspect_f1': asp_f1,
        'noise': aspect_noise(selected_aspects, gt_aspects, user_aspects),
        'redundancy': redundancy(e_emb) if len(e_emb) else 0.0,
    }


def build_case_inputs(df: pd.DataFrame, embs, args):
    """Per-case data that's independent of selector hyperparameters: candidates,
    graded weights, allowed set, gt/user aspects, demand-eligible availability."""
    cases = []
    for (case_id, dataset), group in df.groupby(['case_id', 'dataset']):
        cands = group_to_candidates(group, embs)
        first = group.iloc[0]
        user_aspects_ranked = to_list(first.get('user_aspects'))
        user_aspects = set(user_aspects_ranked)
        n_user = len(user_aspects_ranked)
        user_relevance = {a: 1.0 - i / max(1, n_user) for i, a in enumerate(user_aspects_ranked)}
        item_aspect_cnt: Counter = Counter()
        for row_aspects in group['aspects']:
            item_aspect_cnt.update(to_list(row_aspects))
        top_item_aspects = {a for a, _ in item_aspect_cnt.most_common(args.allowed_item_top_n)}
        allowed = user_aspects | top_item_aspects
        max_item_count = max(item_aspect_cnt.values()) if item_aspect_cnt else 1
        weights = {
            a: args.demand_user_weight * user_relevance.get(a, 0.0)
               + args.demand_item_weight * (item_aspect_cnt[a] / max_item_count)
            for a in (user_aspects | set(item_aspect_cnt))
        }
        cases.append({
            'case_id': case_id, 'dataset': dataset,
            'cands': cands, 'allowed': allowed, 'weights': weights, 'item_aspect_cnt': item_aspect_cnt,
            'gt_aspects': to_list(first.get('gt_aspects')), 'user_aspects': list(user_aspects),
            'user_avg_k': int(first.get('user_avg_k', 3)),
        })
    return cases


def run_config(cases, case_gt, embs, k_list, cfg):
    rows = []
    for c in cases:
        gt_info = case_gt.get(c['case_id'], {})
        for kv in k_list:
            k = resolve_k(kv, c['user_avg_k'], len(c['cands']))
            gated = precision_gate(c['cands'], k, percentile=cfg['gate_percentile'])
            aspect_demand = build_demand_from_weights(
                c['weights'], c['item_aspect_cnt'], k,
                min_weight=cfg['demand_min_weight'], max_per_aspect=cfg['max_per_aspect'],
            ) if k >= 2 else None
            selected = greedy_coverage_select(
                gated, k,
                lambda_coverage=cfg['lambda_coverage'], mu_redundancy=cfg['mu_redundancy'],
                eta_noise=cfg['eta_noise'], nu_aspect_repeat=cfg['nu_aspect_repeat'],
                allowed_aspects=c['allowed'], aspect_demand=aspect_demand,
            )
            rows.append(eval_selection(c['case_id'], c['dataset'], kv, selected, c['gt_aspects'], c['user_aspects'], embs, gt_info))
    return rows


def f_beta(p: float, r: float, beta: float) -> float:
    if p <= 0 or r <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r + 1e-12)


def is_dominated(a: dict, b: dict) -> bool:
    """True if config b dominates a: at least as good on all 4 objectives, strictly better on >=1."""
    geq = b['sem_f1'] >= a['sem_f1'] and b['aspect_f1'] >= a['aspect_f1'] and b['noise'] <= a['noise'] and b['redundancy'] <= a['redundancy']
    gt = b['sem_f1'] > a['sem_f1'] or b['aspect_f1'] > a['aspect_f1'] or b['noise'] < a['noise'] or b['redundancy'] < a['redundancy']
    return geq and gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs-dir', default='data/processed/pairs')
    ap.add_argument('--split', default='valid')
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--model', default='models/peer_ltr.pkl')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--k-list', nargs='+', default=['3', '5', 'user_avg'])
    ap.add_argument('--demand-user-weight', type=float, default=0.6)
    ap.add_argument('--demand-item-weight', type=float, default=0.4)
    ap.add_argument('--allowed-item-top-n', type=int, default=15)
    ap.add_argument('--gate-percentile-grid', nargs='+', type=float, default=[0.3])
    ap.add_argument('--max-per-aspect-grid', nargs='+', default=[3])
    ap.add_argument('--demand-min-weight-grid', nargs='+', type=float, default=[0.15])
    ap.add_argument('--nu-aspect-repeat-grid', nargs='+', type=float, default=[0.05])
    ap.add_argument('--lambda-coverage-grid', nargs='+', type=float, default=[0.2])
    ap.add_argument('--mu-redundancy-grid', nargs='+', type=float, default=[0.1])
    ap.add_argument('--eta-noise-grid', nargs='+', type=float, default=[0.1])
    ap.add_argument('--sem-f1-floor', type=float, default=None,
                     help='Reject configs with mean sem_f1 below this (default: read from --sem-f1-floor-file if given, else no floor)')
    ap.add_argument('--noise-ceiling', type=float, default=None,
                     help='Reject configs with mean noise above this (default: no ceiling)')
    ap.add_argument('--beta', type=float, default=1.5)
    ap.add_argument('--rank-by', default='f_beta', choices=['f_beta', 'sem_f1'],
                     help='Which column picks the winner among constraint-satisfying Pareto configs')
    ap.add_argument('--output', default='results/selector_tuning.csv')
    ap.add_argument('--dataset', default=None,
                     help='Restrict to one dataset (e.g. baby/yelp/googlelocal) to bound memory; run once per '
                          'dataset and combine (weighted by case count) to verify a config on the full split.')
    args = ap.parse_args()

    df = read_pairs_columns(Path(args.pairs_dir) / f'{args.split}.parquet')
    if args.dataset is not None:
        df = df[df['dataset'] == args.dataset]
    model = RankerModel.load(args.model)
    df['score'] = model.predict(df)
    case_gt = load_case_gt(args.cases)
    needed_ids = set(df['sentence_id'].astype(str))
    for case_id in df['case_id'].unique():
        gt_info = case_gt.get(case_id)
        if gt_info:
            needed_ids.update(gt_info['gt_sentence_ids'])
    embs = load_embeddings(args.embeddings, needed_ids)
    k_list = parse_k_list(args.k_list)

    print(f'Building per-case inputs for {df["case_id"].nunique()} {args.split} cases...')
    cases = build_case_inputs(df, embs, args)

    max_per_aspect_grid = [None if str(v).lower() in {'none', 'null'} else int(v) for v in args.max_per_aspect_grid]
    grid = list(itertools.product(
        args.gate_percentile_grid,
        max_per_aspect_grid,
        args.demand_min_weight_grid,
        args.nu_aspect_repeat_grid,
        args.lambda_coverage_grid,
        args.mu_redundancy_grid,
        args.eta_noise_grid,
    ))
    print(f'{len(grid)} configs in grid')
    results = []
    for gate_percentile, max_per_aspect, demand_min_weight, nu_aspect_repeat, lambda_coverage, mu_redundancy, eta_noise in tqdm(grid, desc='Grid search'):
        cfg = dict(
            gate_percentile=gate_percentile, max_per_aspect=max_per_aspect,
            demand_min_weight=demand_min_weight, nu_aspect_repeat=nu_aspect_repeat,
            lambda_coverage=lambda_coverage, mu_redundancy=mu_redundancy, eta_noise=eta_noise,
        )
        rows = run_config(cases, case_gt, embs, k_list, cfg)
        agg = pd.DataFrame(rows)
        mean = agg[['sem_f1', 'aspect_p', 'aspect_r', 'aspect_f1', 'noise', 'redundancy']].mean()
        rec = {**cfg, **mean.to_dict()}
        rec['f_beta'] = f_beta(rec['aspect_p'], rec['aspect_r'], args.beta)
        results.append(rec)

    res_df = pd.DataFrame(results)
    res_df['max_per_aspect'] = res_df['max_per_aspect'].astype('object').where(res_df['max_per_aspect'].notna(), 'none')

    recs = res_df.to_dict('records')
    for r in recs:
        r['pareto'] = not any(is_dominated(r, other) for other in recs if other is not r)
    res_df['pareto'] = [r['pareto'] for r in recs]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    res_df.sort_values(args.rank_by, ascending=False).to_csv(args.output, index=False)
    print(f'Wrote grid results -> {args.output}')
    print('\n=== All configs, sorted by sem_f1 ===')
    print(res_df.sort_values('sem_f1', ascending=False).to_string(index=False))

    pool = res_df[res_df['pareto']].copy()
    if args.sem_f1_floor is not None:
        pool = pool[pool['sem_f1'] >= args.sem_f1_floor]
    if args.noise_ceiling is not None:
        pool = pool[pool['noise'] <= args.noise_ceiling]
    if pool.empty:
        print('WARNING: no Pareto config satisfies the constraints; falling back to full grid.')
        pool = res_df.copy()
        if args.sem_f1_floor is not None:
            pool = pool[pool['sem_f1'] >= args.sem_f1_floor]
        if args.noise_ceiling is not None:
            pool = pool[pool['noise'] <= args.noise_ceiling]
    winner = pool.sort_values(args.rank_by, ascending=False).iloc[0]
    print('\n=== Pareto-optimal configs ===')
    print(res_df[res_df['pareto']].sort_values('sem_f1', ascending=False).to_string(index=False))
    print(f'\n=== Winner (max {args.rank_by} among constraint-satisfying Pareto configs) ===')
    print(winner.to_string())


if __name__ == '__main__':
    main()
