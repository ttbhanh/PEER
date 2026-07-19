#!/usr/bin/env python
from __future__ import annotations

"""Extract evidence sets for offline evaluation + a blind human user study,
comparing PEER against a fixed set of baselines (default: PRAG, ERRA-R) on a
stratified sample of test cases.

Produces three files in --output:
  - offline_eval_labeled.csv : method names visible, joined with per-case
    metrics from results/per_case.csv -- for automated/internal analysis.
  - blind_study.csv          : method identity replaced with a per-case
    randomized option label (A/B/C/...) -- safe to hand to human raters
    without revealing which system produced which evidence set.
  - blind_study_mapping.csv  : study_case_id -> {option letter: method} key
    to decode blind_study.csv results after collecting ratings. NOT for
    raters.

Sampling and the per-case option-letter shuffle are both seeded (--seed) for
reproducibility -- re-running with the same seed and inputs regenerates an
identical sample and mapping.
"""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import csv
import random
from pathlib import Path

import pandas as pd

from peer.utils import ensure_dir, read_jsonl


def load_predictions(pred_dir: Path, method: str, k: str, split: str = 'test') -> dict[str, dict]:
    for sub in ['peer', 'published', 'baselines']:
        p = pred_dir / sub / f'{method}_{split}.jsonl'
        if p.exists():
            rows = [r for r in read_jsonl(p) if str(r.get('k')) == str(k)]
            return {r['case_id']: r for r in rows}
    raise FileNotFoundError(f'No predictions found for method={method} under {pred_dir}/*/{method}_{split}.jsonl')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=['peer_full', 'prag', 'erra_r'],
                     help='Method prediction files to compare (must exist under outputs/predictions/{peer,published,baselines}/)')
    ap.add_argument('--k', default='user_avg')
    ap.add_argument('--per-dataset', type=int, default=50,
                     help='Cases to sample per dataset (stratified)')
    ap.add_argument('--pred-dir', default='outputs/predictions')
    ap.add_argument('--cases', default='data/cases/cases_test.jsonl')
    ap.add_argument('--per-case-metrics', default='results/per_case.csv')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='outputs/user_study')
    args = ap.parse_args()

    rng = random.Random(args.seed)

    cases = {c['case_id']: c for c in read_jsonl(args.cases)}
    by_dataset: dict[str, list[str]] = {}
    for cid, c in cases.items():
        by_dataset.setdefault(c['dataset'], []).append(cid)

    sampled_ids: list[str] = []
    for ds, ids in sorted(by_dataset.items()):
        ids_sorted = sorted(ids)  # deterministic order before sampling
        rng.shuffle(ids_sorted)
        take = ids_sorted[:args.per_dataset]
        sampled_ids.extend(take)
        print(f'{ds}: sampled {len(take)}/{len(ids)} cases')
    sampled_ids = sorted(sampled_ids)  # stable overall order

    preds = {m: load_predictions(Path(args.pred_dir), m, args.k) for m in args.methods}
    missing = [m for m in args.methods for cid in sampled_ids if cid not in preds[m]]
    if missing:
        raise RuntimeError(f'{len(missing)} (method,case) pairs missing predictions, e.g. {missing[:5]}')

    metrics_df = pd.read_csv(args.per_case_metrics)
    metrics_df = metrics_df[(metrics_df.k.astype(str) == str(args.k)) & (metrics_df.method.isin(args.methods))]
    metrics_idx = {(r.case_id, r.method): r for r in metrics_df.itertuples()}

    out_dir = ensure_dir(args.output)
    offline_rows = []
    blind_rows = []
    mapping_rows = []

    for study_idx, cid in enumerate(sampled_ids):
        c = cases[cid]
        options = list(args.methods)
        rng.shuffle(options)
        letters = [chr(ord('A') + i) for i in range(len(options))]
        study_case_id = f'study_{study_idx:04d}'

        mapping_row = {'study_case_id': study_case_id, 'case_id': cid}
        # item_metadata_text deliberately excluded from study context: PEER's
        # own method drops product metadata entirely (see peer/models.py --
        # an ablation showed metadata's contribution was within noise), so
        # showing raters metadata PEER never sees would bias the comparison
        # in favor of PRAG/ERRA-R, whose mechanisms do use it.
        blind_row = {
            'study_case_id': study_case_id,
            'dataset': c['dataset'],
            'user_history_text': c['user_history_text'],
        }
        for letter, method in zip(letters, options):
            mapping_row[f'option_{letter}_method'] = method
            pred = preds[method][cid]
            evidence_text = ' '.join(pred.get('selected_texts') or [])
            blind_row[f'option_{letter}_evidence'] = evidence_text
            # empty rating placeholders -- adjust dimensions/scale to your study design
            blind_row[f'option_{letter}_relevance_1to5'] = ''
            blind_row[f'option_{letter}_coverage_1to5'] = ''
            blind_row[f'option_{letter}_redundancy_1to5'] = ''

            m = metrics_idx.get((cid, method))
            offline_rows.append({
                'case_id': cid, 'dataset': c['dataset'], 'method': method, 'k': args.k,
                'user_history_text': c['user_history_text'],
                'selected_texts': evidence_text,
                'selected_aspects': ';'.join(pred.get('selected_aspects') or []),
                'ground_truth_text': c['ground_truth_text'],
                'sem_f1': getattr(m, 'sem_f1', None), 'aspect_f1': getattr(m, 'aspect_f1', None),
                'noise': getattr(m, 'noise', None), 'redundancy': getattr(m, 'redundancy', None),
            })
        mapping_rows.append(mapping_row)
        blind_rows.append(blind_row)

    pd.DataFrame(offline_rows).to_csv(out_dir / 'offline_eval_labeled.csv', index=False, quoting=csv.QUOTE_ALL)
    pd.DataFrame(blind_rows).to_csv(out_dir / 'blind_study.csv', index=False, quoting=csv.QUOTE_ALL)
    pd.DataFrame(mapping_rows).to_csv(out_dir / 'blind_study_mapping.csv', index=False, quoting=csv.QUOTE_ALL)

    print(f'\nWrote {len(sampled_ids)} cases x {len(args.methods)} methods:')
    print(f'  {out_dir / "offline_eval_labeled.csv"}  ({len(offline_rows)} rows, method names visible)')
    print(f'  {out_dir / "blind_study.csv"}  ({len(blind_rows)} rows, blinded/randomized -- safe for raters)')
    print(f'  {out_dir / "blind_study_mapping.csv"}  ({len(mapping_rows)} rows, decode key -- NOT for raters)')


if __name__ == '__main__':
    main()
