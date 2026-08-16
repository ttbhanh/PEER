#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd

from peer.utils import ensure_dir, read_jsonl, resolve_cases_path, write_jsonl


def load_predictions(pred_dir: Path):
    rows = []
    for f in sorted(pred_dir.rglob('*.jsonl')):
        rows.extend(read_jsonl(f))
    return rows


def load_cases(cases_dir: Path):
    cases = {}
    for split in ['train','valid','test']:
        p = resolve_cases_path(cases_dir, split)
        if p is not None:
            for c in read_jsonl(p):
                cases[c['case_id']] = c
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-case', default='results/per_case.csv')
    ap.add_argument('--pred-dir', default='outputs/predictions')
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--main-method', default='peer_full')
    ap.add_argument('--compare-method', default='sbert_user_metadata')
    ap.add_argument('--k', default='user_avg')
    ap.add_argument('--num-success', type=int, default=3)
    ap.add_argument('--num-failure', type=int, default=1)
    ap.add_argument('--output', default='outputs/case_studies/case_studies.jsonl')
    args = ap.parse_args()

    metrics = pd.read_csv(args.per_case)
    preds = load_predictions(Path(args.pred_dir))
    pred_map = {(p['case_id'], p['method'], str(p['k'])): p for p in preds}
    cases = load_cases(Path(args.cases))
    rows = []
    for ds in sorted(metrics['dataset'].unique()):
        sub_main = metrics[(metrics['dataset']==ds) & (metrics['method']==args.main_method) & (metrics['k'].astype(str)==str(args.k))]
        sub_base = metrics[(metrics['dataset']==ds) & (metrics['method']==args.compare_method) & (metrics['k'].astype(str)==str(args.k))]
        merged = sub_main[['case_id','sem_f1','aspect_f1']].merge(sub_base[['case_id','sem_f1','aspect_f1']], on='case_id', suffixes=('_main','_base'))
        merged['gain'] = (merged['sem_f1_main'] + merged['aspect_f1_main']) - (merged['sem_f1_base'] + merged['aspect_f1_base'])
        selected = pd.concat([merged.sort_values('gain', ascending=False).head(args.num_success), merged.sort_values('gain', ascending=True).head(args.num_failure)])
        for _, r in selected.iterrows():
            cid = r['case_id']
            case = cases.get(cid, {})
            main_pred = pred_map.get((cid, args.main_method, str(args.k)), {})
            base_pred = pred_map.get((cid, args.compare_method, str(args.k)), {})
            rows.append({
                'dataset': ds,
                'case_id': cid,
                'gain': float(r['gain']),
                'user_id': case.get('user_id'),
                'item_id': case.get('item_id'),
                'user_history_sample': case.get('user_history_sentences', [])[-5:],
                'metadata_text': case.get('metadata_text','')[:1000],
                'ground_truth_review': case.get('ground_truth_text',''),
                'ground_truth_aspects': main_pred.get('gt_aspects', []),
                f'{args.compare_method}_selected': base_pred.get('selected_texts', []),
                f'{args.main_method}_selected': main_pred.get('selected_texts', []),
            })
    ensure_dir(Path(args.output).parent)
    write_jsonl(rows, args.output)
    print(f'Wrote {len(rows)} case studies -> {args.output}')


if __name__ == '__main__':
    main()
