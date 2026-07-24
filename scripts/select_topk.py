#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from peer.models import FEATURE_COLUMNS_DEFAULT, RankerModel
from peer.selectors import resolve_k, topk, greedy_coverage_select
from peer.utils import ensure_dir, write_jsonl

# Columns this pipeline (select_topk.py/tune_selector.py/search_label_weights.py)
# actually uses, instead of every column in the pairs parquet -- at full scale
# (millions of rows), reading only these columns is the difference between
# fitting in memory and not.
PIPELINE_CASE_COLUMNS = [
    'case_id', 'dataset', 'sentence_id', 'text', 'aspects',
    'user_aspects', 'gt_aspects', 'user_avg_k', 'ground_truth_text',
]
LABEL_COMPONENT_COLUMNS = ['semantic_to_gt', 'aspect_match_to_gt', 'sentiment_match', 'coverage_gain']


def read_pairs_columns(path, extra: list[str] | None = None, include_case_columns: bool = True) -> pd.DataFrame:
    import pyarrow.parquet as pq
    # .schema_arrow (not .schema) is needed to get real top-level column names
    # for list<string> columns like 'aspects'.
    available = set(pq.ParquetFile(path).schema_arrow.names)
    base = [*PIPELINE_CASE_COLUMNS, *FEATURE_COLUMNS_DEFAULT] if include_case_columns else list(FEATURE_COLUMNS_DEFAULT)
    wanted = [c for c in [*base, *(extra or [])] if c in available]
    if not wanted:
        return pd.read_parquet(path)
    return pd.read_parquet(path, columns=wanted)


def load_embeddings(path: str | None):
    if not path:
        return {}
    p = Path(path)
    npy_path = p.with_suffix('.npy')
    if not npy_path.exists():
        return {}
    emb = np.load(npy_path, mmap_mode='r')
    with open(p.with_suffix('.ids.json')) as f:
        ids = json.load(f)
    return {str(ids[i]): emb[i] for i in range(len(ids))}


def to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass
    return [x]


def parse_k_list(values):
    return ['user_avg' if str(v).lower() in {'user', 'user_avg', 'user_avg_k'} else int(v) for v in values]


def group_to_candidates(group: pd.DataFrame, embs):
    cands = []
    for r in group.to_dict('records'):
        c = {'sentence_id': r['sentence_id'], 'text': r['text'], 'score': float(r['score']), 'aspects': to_list(r.get('aspects'))}
        if r['sentence_id'] in embs:
            c['embedding'] = embs[r['sentence_id']]
        cands.append(c)
    return cands


def pred_row(case_id, dataset, method, kname, selected, group):
    first = group.iloc[0]
    return {
        'case_id': case_id,
        'dataset': dataset,
        'method': method,
        'k': str(kname),
        'selected_sentence_ids': [s['sentence_id'] for s in selected],
        'selected_texts': [s['text'] for s in selected],
        'selected_aspects': [a for s in selected for a in to_list(s.get('aspects'))],
        'ground_truth_text': first['ground_truth_text'],
        'gt_aspects': to_list(first.get('gt_aspects')),
        'user_aspects': to_list(first.get('user_aspects')),
        'user_avg_k': int(first.get('user_avg_k', 3)),
    }


def run_split(args, split: str):
    df = read_pairs_columns(Path(args.pairs_dir) / f'{split}.parquet')
    if getattr(args, 'dataset', None) is not None:
        df = df[df['dataset'] == args.dataset]
    model = RankerModel.load(args.model)
    df['score'] = model.predict(df)
    embs = load_embeddings(args.embeddings)
    preds = []
    for (case_id, dataset), group in tqdm(df.groupby(['case_id', 'dataset']), desc=f'PEER {split}'):
        cands = group_to_candidates(group, embs)
        first = group.iloc[0]
        user_aspects = set(to_list(first.get('user_aspects')))
        # allowed_aspects for the selector's noise term: this user's own aspects,
        # union with the item's own top-N most-discussed aspects (a grounded
        # proxy for "in-domain for this product").
        item_aspect_cnt: Counter = Counter()
        for row_aspects in group['aspects']:
            item_aspect_cnt.update(to_list(row_aspects))
        top_item_aspects = {a for a, _ in item_aspect_cnt.most_common(args.allowed_item_top_n)}
        allowed = user_aspects | top_item_aspects
        for kv in parse_k_list(args.k_list):
            k = resolve_k(kv, int(first.get('user_avg_k', 3)), len(cands))
            if args.selector == 'topk':
                selected = topk(cands, k)
            else:
                selected = greedy_coverage_select(
                    cands, k,
                    lambda_coverage=args.lambda_coverage,
                    mu_redundancy=args.mu_redundancy,
                    eta_noise=args.eta_noise,
                    nu_aspect_repeat=args.nu_aspect_repeat,
                    allowed_aspects=allowed,
                )
            preds.append(pred_row(case_id, dataset, args.method_name, kv, selected, group))
    out_dir = ensure_dir(args.output)
    suffix = f'__{args.dataset}' if getattr(args, 'dataset', None) is not None else ''
    path = out_dir / f'{args.method_name}_{split}{suffix}.jsonl'
    write_jsonl(preds, path)
    print(f'Wrote {len(preds)} predictions -> {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs-dir', default='data/processed/pairs')
    ap.add_argument('--model', default='models/peer_ltr.pkl')
    ap.add_argument('--splits', nargs='+', default=['test'])
    ap.add_argument('--selector', default='greedy', choices=['greedy', 'topk'])
    ap.add_argument('--method-name', default='peer_full')
    ap.add_argument('--k-list', nargs='+', default=['1', '3', '5', 'user_avg'])
    # Shipped values (paper, Sec. "Discussion and Limitations"): redundancy and
    # noise penalties are tuned to 0 in the final config, so PEER optimizes
    # purely for utility + aspect coverage rather than trading sem-F1/aspect-F1
    # for better noise/redundancy numbers.
    ap.add_argument('--lambda-coverage', type=float, default=0.1)
    ap.add_argument('--mu-redundancy', type=float, default=0.0)
    ap.add_argument('--eta-noise', type=float, default=0.0)
    ap.add_argument('--nu-aspect-repeat', type=float, default=0.0)
    ap.add_argument('--allowed-item-top-n', type=int, default=15,
                     help='How many of the item\'s most-discussed aspects count as "allowed" for the noise term')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--output', default='outputs/predictions/peer')
    ap.add_argument('--dataset', default=None,
                     help='Restrict to one dataset instead of the whole split at once (memory-constrained '
                          'environments). Run once per dataset (output filenames get a __{dataset} suffix) '
                          'and concatenate the jsonl files afterward.')
    args = ap.parse_args()
    for split in args.splits:
        run_split(args, split)


if __name__ == '__main__':
    main()
