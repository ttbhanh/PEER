#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rank_bm25 import BM25Okapi
except Exception:
    class BM25Okapi:
        def __init__(self, tokenized_corpus, k1=1.5, b=0.75):
            import math
            from collections import Counter
            self.corpus = tokenized_corpus
            self.k1 = k1
            self.b = b
            self.doc_len = [len(d) for d in tokenized_corpus]
            self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
            self.freqs = [Counter(d) for d in tokenized_corpus]
            df = Counter()
            for d in tokenized_corpus:
                df.update(set(d))
            n = max(1, len(tokenized_corpus))
            self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        def get_scores(self, query_tokens):
            scores = []
            for f, dl in zip(self.freqs, self.doc_len):
                s = 0.0
                for q in query_tokens:
                    tf = f.get(q, 0)
                    if tf == 0:
                        continue
                    idf = self.idf.get(q, 0.0)
                    denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
                    s += idf * tf * (self.k1 + 1) / denom
                scores.append(s)
            return scores
from tqdm import tqdm

from peer.selectors import resolve_k, topk
from peer.text import tokenize
from peer.utils import ensure_dir, write_jsonl


def load_embeddings(path: str | None):
    if not path:
        return {}
    p = Path(path)
    npy_path = p.with_suffix('.npy')
    if not npy_path.exists():
        return {}
    # Plain .npy (mmap-able) + sibling ids.json, not .npz -- see
    # peer/embeddings.py::_npy_and_ids_paths.
    emb = np.load(npy_path, mmap_mode='r')
    with open(p.with_suffix('.ids.json')) as f:
        ids = json.load(f)
    return {str(ids[i]): emb[i] for i in range(len(ids))}


def to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    try:
        import pandas as pd
        if pd.isna(x):
            return []
    except Exception:
        pass
    return [x]


def parse_k_list(values):
    out = []
    for v in values:
        if str(v).lower() in {'user_avg', 'user_avg_k', 'user'}:
            out.append('user_avg')
        else:
            out.append(int(v))
    return out


def method_scores(method: str, group: pd.DataFrame) -> np.ndarray:
    n = len(group)
    if method == 'random':
        return np.random.random(n)
    if method == 'popular':
        return group.get('helpfulness_norm', pd.Series(np.zeros(n))).values.astype(float)
    if method == 'recent':
        return group.get('recency_norm', pd.Series(np.zeros(n))).values.astype(float)
    if method == 'sbert_user':
        return group['user_sem_sim'].values.astype(float)
    if method == 'bm25_user':
        tokenized = [tokenize(t) for t in group['text'].tolist()]
        bm25 = BM25Okapi(tokenized)
        first = group.iloc[0]
        q = to_list(first.get('user_aspects'))
        q_toks = tokenize(' '.join(q))
        return np.asarray(bm25.get_scores(q_toks), dtype=float) if q_toks else np.zeros(n)
    raise ValueError(f'Unknown baseline method: {method}')


def group_to_candidates(group: pd.DataFrame, scores: np.ndarray, embs: dict[str, np.ndarray]):
    cands = []
    for score, r in zip(scores, group.to_dict('records')):
        c = {
            'sentence_id': r['sentence_id'],
            'text': r['text'],
            'score': float(score),
            'aspects': to_list(r.get('aspects')),
        }
        if r['sentence_id'] in embs:
            c['embedding'] = embs[r['sentence_id']]
        cands.append(c)
    return cands


def prediction_row(case_id, dataset, method, kname, selected, group):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs-dir', default='data/processed/pairs')
    ap.add_argument('--split', default='test', choices=['train','valid','test'])
    ap.add_argument('--methods', nargs='+', default=['random','popular','recent','bm25_user','sbert_user'])
    ap.add_argument('--k-list', nargs='+', default=['1','3','5','user_avg'])
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--output', default='outputs/predictions/baselines')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = ensure_dir(args.output)
    df = pd.read_parquet(Path(args.pairs_dir) / f'{args.split}.parquet')
    embs = load_embeddings(args.embeddings)
    k_list = parse_k_list(args.k_list)

    for method in args.methods:
        preds = []
        for (case_id, dataset), group in tqdm(df.groupby(['case_id','dataset']), desc=f'{method}'):
            scores = method_scores(method, group)
            cands = group_to_candidates(group, scores, embs)
            for kv in k_list:
                k = resolve_k(kv, int(group.iloc[0].get('user_avg_k', 3)), len(cands))
                selected = topk(cands, k)
                preds.append(prediction_row(case_id, dataset, method, kv, selected, group))
        path = out_dir / f'{method}_{args.split}.jsonl'
        write_jsonl(preds, path)
        print(f'Wrote {len(preds)} predictions -> {path}')


if __name__ == '__main__':
    main()
