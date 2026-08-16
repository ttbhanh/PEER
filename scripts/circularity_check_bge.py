#!/usr/bin/env python
from __future__ import annotations

"""Independent-encoder robustness check (Table "Robustness under Independent
Evaluation", sem_f1 half): every method's *already-selected* evidence
(no reselection, no retraining) is re-encoded and re-scored against the
held-out review with BAAI/bge-base-en-v1.5 instead of the primary
all-mpnet-base-v2 encoder used by the ranking features and the training
label. Tests whether the reported gains are tied to the specific encoder
used elsewhere in the pipeline.
"""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd

from peer.utils import ensure_dir, read_jsonl, resolve_cases_path
from peer.metrics import semantic_prf
from peer.embeddings import TextEmbedder


def load_predictions(pred_dir: _ProjectPath, method: str, split: str, k: str) -> list[dict]:
    for sub in ['peer', 'published', 'baselines']:
        p = pred_dir / sub / f'{method}_{split}.jsonl'
        if p.exists():
            return [r for r in read_jsonl(p) if str(r.get('k')) == str(k)]
    raise FileNotFoundError(f'No predictions for method={method} under {pred_dir}/*/{method}_{split}.jsonl')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--pred-dir', default='outputs/predictions')
    ap.add_argument('--split', default='test')
    ap.add_argument('--k', default='user_avg')
    ap.add_argument('--methods', nargs='+', required=True,
                     help='Prediction filenames (without _{split}.jsonl) to score, e.g. peer_full prag erra_r')
    ap.add_argument('--model', default='BAAI/bge-base-en-v1.5')
    ap.add_argument('--device', default=None)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--output', default='results/circularity_check_bge')
    args = ap.parse_args()

    pred_dir = _ProjectPath(args.pred_dir)
    embedder = TextEmbedder(args.model, device=args.device, fp16=False, fallback_tfidf=False)
    print(f'Embedder kind: {embedder.kind}', flush=True)

    case_gt: dict[str, list[str]] = {}
    cases_path = resolve_cases_path(args.cases, args.split)
    for c in read_jsonl(cases_path):
        case_gt[c['case_id']] = c['ground_truth_sentences']
    print(f'test cases with gt: {len(case_gt)}', flush=True)

    method_rows: dict[str, list[dict]] = {}
    texts_needed: set[str] = set()
    for method in args.methods:
        rows = load_predictions(pred_dir, method, args.split, args.k)
        method_rows[method] = rows
        for r in rows:
            texts_needed.update(r.get('selected_texts') or [])
            texts_needed.update(case_gt.get(r['case_id'], []))
        print(f'{method}: {len(rows)} rows loaded', flush=True)

    texts_needed.discard('')
    texts_list = sorted(texts_needed)
    print(f'unique texts to embed: {len(texts_list)}', flush=True)
    embs_arr = embedder.encode(texts_list, batch_size=args.batch_size)
    text2emb = {t: embs_arr[i] for i, t in enumerate(texts_list)}
    print('embedding done', flush=True)

    summary_rows, percase_rows = [], []
    for method, rows in method_rows.items():
        per_ds: dict[str, list[float]] = {}
        for r in rows:
            ds = r['dataset']
            sel = [t for t in (r.get('selected_texts') or []) if t in text2emb]
            gt = [t for t in case_gt.get(r['case_id'], []) if t in text2emb]
            e_emb = np.asarray([text2emb[t] for t in sel], dtype=np.float32)
            g_emb = np.asarray([text2emb[t] for t in gt], dtype=np.float32)
            _, _, f1 = semantic_prf(e_emb, g_emb) if len(e_emb) and len(g_emb) else (0.0, 0.0, 0.0)
            per_ds.setdefault(ds, []).append(f1)
            percase_rows.append({'dataset': ds, 'method': method, 'case_id': r['case_id'], 'sem_f1_bge': f1})
        for ds, vals in per_ds.items():
            summary_rows.append({'dataset': ds, 'method': method, 'n_cases': len(vals), 'sem_f1_bge': float(np.mean(vals))})
        all_vals = [v for vs in per_ds.values() for v in vs]
        print(f'{method}: pooled sem_f1(BGE)={np.mean(all_vals):.4f} (n={len(all_vals)})', flush=True)

    out = _ProjectPath(args.output)
    ensure_dir(out.parent)
    pd.DataFrame(summary_rows).to_csv(f'{out}_summary.csv', index=False)
    pd.DataFrame(percase_rows).to_csv(f'{out}_percase.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
