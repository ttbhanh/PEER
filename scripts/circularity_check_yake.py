#!/usr/bin/env python
from __future__ import annotations

"""Independent-extractor robustness check (Table "Robustness under
Independent Evaluation", aspect_f1 half): every method's already-selected
evidence is re-tagged with YAKE, an unsupervised statistical keyword
extractor, instead of the primary spaCy+lexicon aspect pipeline used
elsewhere. Tests whether the reported aspect_f1 gains are tied to the
specific extractor used for feature construction and training labels.

NOTE: the exact YAKE hyperparameters used to produce the numbers reported in
the paper were not preserved. This script uses YAKE's own defaults
(n-gram size <=2, top 10 keywords/sentence, deduplication threshold 0.9) as
a reasonable, clearly-labeled reconstruction -- expect small (not
qualitative) numeric differences from the paper's reported values, similar
in spirit to the paper's own personalization swap-test population, which
carries the same caveat (see README).
"""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import time

import numpy as np
import pandas as pd
import yake

from peer.utils import ensure_dir, read_jsonl, resolve_cases_path
from peer.aspects import aspect_f1


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
    ap.add_argument('--ngram-max', type=int, default=2)
    ap.add_argument('--top-n', type=int, default=10)
    ap.add_argument('--dedup-threshold', type=float, default=0.9)
    ap.add_argument('--output', default='results/circularity_check_yake')
    args = ap.parse_args()

    pred_dir = _ProjectPath(args.pred_dir)
    extractor = yake.KeywordExtractor(n=args.ngram_max, dedupLim=args.dedup_threshold, top=args.top_n)

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
    print(f'unique sentences to YAKE-tag: {len(texts_list)}', flush=True)

    text2aspects: dict[str, list[str]] = {}
    t0 = time.time()
    for i, t in enumerate(texts_list):
        kws = extractor.extract_keywords(t)
        text2aspects[t] = [kw.lower() for kw, _ in kws]
        if (i + 1) % 50000 == 0:
            print(f'tagged {i+1}/{len(texts_list)} ({(i+1)/(time.time()-t0):.0f}/s)', flush=True)
    print(f'YAKE tagging done in {time.time()-t0:.1f}s', flush=True)

    summary_rows, percase_rows = [], []
    for method, rows in method_rows.items():
        per_ds: dict[str, list[float]] = {}
        for r in rows:
            ds = r['dataset']
            pred_aspects = [a for t in (r.get('selected_texts') or []) for a in text2aspects.get(t, [])]
            gt_aspects = [a for t in case_gt.get(r['case_id'], []) for a in text2aspects.get(t, [])]
            _, _, f1 = aspect_f1(pred_aspects, gt_aspects)
            per_ds.setdefault(ds, []).append(f1)
            percase_rows.append({'dataset': ds, 'method': method, 'case_id': r['case_id'], 'aspect_f1_yake': f1})
        for ds, vals in per_ds.items():
            summary_rows.append({'dataset': ds, 'method': method, 'n_cases': len(vals), 'aspect_f1_yake': float(np.mean(vals))})
        all_vals = [v for vs in per_ds.values() for v in vs]
        print(f'{method}: pooled aspect_f1(YAKE)={np.mean(all_vals):.4f} (n={len(all_vals)})', flush=True)

    out = _ProjectPath(args.output)
    ensure_dir(out.parent)
    pd.DataFrame(summary_rows).to_csv(f'{out}_summary.csv', index=False)
    pd.DataFrame(percase_rows).to_csv(f'{out}_percase.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
