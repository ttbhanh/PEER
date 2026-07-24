#!/usr/bin/env python
from __future__ import annotations

"""Selector hyperparameter search (paper, Discussion: "the selector search
covers 15 values"), run on the VALID split only. The shipped selector
(scripts/select_topk.py) fixes mu_redundancy=eta_noise=nu_aspect_repeat=0 and
sweeps lambda_coverage alone; this script reproduces that sweep."""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from peer.aspects import aspect_f1 as aspect_f1_fn
from peer.metrics import aggregate_metrics, aspect_noise, redundancy, semantic_prf
from peer.models import RankerModel
from peer.selectors import greedy_coverage_select, resolve_k
from peer.utils import read_jsonl

from scripts.select_topk import group_to_candidates, load_embeddings, parse_k_list, read_pairs_columns, to_list


def load_case_gt(cases_dir: str):
    m = {}
    for split in ['train', 'valid', 'test']:
        p = Path(cases_dir) / f'cases_{split}.jsonl'
        if not p.exists():
            continue
        for c in read_jsonl(p):
            m[c['case_id']] = {
                'gt_sentence_ids': [f"{c['case_id']}_gt_{i}" for i in range(len(c['ground_truth_sentences']))],
                'ground_truth_text': c['ground_truth_text'],
            }
    return m


def eval_selection(dataset, k, selected, gt_aspects, user_aspects, embs, gt_info):
    selected_ids = [s['sentence_id'] for s in selected]
    selected_texts = [s['text'] for s in selected]
    selected_aspects = list(dict.fromkeys(a for s in selected for a in to_list(s.get('aspects'))))
    gt_ids = gt_info.get('gt_sentence_ids', [])
    e_emb = np.asarray([embs[sid] for sid in selected_ids if sid in embs], dtype=np.float32)
    g_emb = np.asarray([embs[sid] for sid in gt_ids if sid in embs], dtype=np.float32)
    sem_p = sem_r = sem_f1 = 0.0
    if len(e_emb) and len(g_emb):
        sem_p, sem_r, sem_f1 = semantic_prf(e_emb, g_emb)
    asp_p, asp_r, asp_f1 = aspect_f1_fn(selected_aspects, gt_aspects)
    return {
        'dataset': dataset, 'k': str(k),
        'sem_p': sem_p, 'sem_r': sem_r, 'sem_f1': sem_f1,
        'aspect_p': asp_p, 'aspect_r': asp_r, 'aspect_f1': asp_f1,
        'noise': aspect_noise(selected_aspects, gt_aspects, user_aspects),
        'redundancy': redundancy(e_emb) if len(e_emb) else 0.0,
    }


def build_case_inputs(df: pd.DataFrame, embs, args):
    cases = []
    for (case_id, dataset), group in df.groupby(['case_id', 'dataset']):
        cands = group_to_candidates(group, embs)
        first = group.iloc[0]
        user_aspects = set(to_list(first.get('user_aspects')))
        item_aspect_cnt: Counter = Counter()
        for row_aspects in group['aspects']:
            item_aspect_cnt.update(to_list(row_aspects))
        top_item_aspects = {a for a, _ in item_aspect_cnt.most_common(args.allowed_item_top_n)}
        allowed = user_aspects | top_item_aspects
        cases.append({
            'case_id': case_id, 'dataset': dataset,
            'cands': cands, 'allowed': allowed,
            'gt_aspects': to_list(first.get('gt_aspects')), 'user_aspects': list(user_aspects),
            'user_avg_k': int(first.get('user_avg_k', 3)),
        })
    return cases


def run_config(cases, case_gt, embs, k_list, lambda_coverage):
    rows = []
    for c in cases:
        gt_info = case_gt.get(c['case_id'], {})
        for kv in k_list:
            k = resolve_k(kv, c['user_avg_k'], len(c['cands']))
            selected = greedy_coverage_select(
                c['cands'], k, lambda_coverage=lambda_coverage,
                mu_redundancy=0.0, eta_noise=0.0, nu_aspect_repeat=0.0,
                allowed_aspects=c['allowed'],
            )
            rows.append(eval_selection(c['dataset'], kv, selected, c['gt_aspects'], c['user_aspects'], embs, gt_info))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs-dir', default='data/processed/pairs')
    ap.add_argument('--split', default='valid')
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--model', default='models/peer_ltr.pkl')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--k-list', nargs='+', default=['user_avg'])
    ap.add_argument('--allowed-item-top-n', type=int, default=15)
    ap.add_argument('--lambda-grid', nargs='+', type=float,
                     default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.85, 1.0],
                     help='15-point default grid, see paper Discussion ("the selector search covers 15 values")')
    ap.add_argument('--output', default='results/selector_tuning.csv')
    args = ap.parse_args()

    df = read_pairs_columns(Path(args.pairs_dir) / f'{args.split}.parquet')
    model = RankerModel.load(args.model)
    df['score'] = model.predict(df)
    embs = load_embeddings(args.embeddings)
    case_gt = load_case_gt(args.cases)
    k_list = parse_k_list(args.k_list)

    print(f'Building per-case inputs for {df["case_id"].nunique()} {args.split} cases...')
    cases = build_case_inputs(df, embs, args)

    results = []
    for lam in tqdm(args.lambda_grid, desc='lambda_coverage sweep'):
        rows = run_config(cases, case_gt, embs, k_list, lam)
        mean = pd.DataFrame(rows)[['sem_f1', 'aspect_p', 'aspect_r', 'aspect_f1', 'noise', 'redundancy']].mean()
        results.append({'lambda_coverage': lam, **mean.to_dict()})

    res_df = pd.DataFrame(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(args.output, index=False)
    print(f'Wrote grid results -> {args.output}')
    print(res_df.sort_values('sem_f1', ascending=False).to_string(index=False))


if __name__ == '__main__':
    main()
