#!/usr/bin/env python
from __future__ import annotations

"""Draws (and saves, for reuse) the stratified 100-case sample used by the
zero-shot-LLM-at-scale robustness check (Section "LLM comparison at scale").
34/33/33 cases are drawn uniformly at random (fixed seed, no cherry-picking)
from each platform's own test split. For every sampled case we save
everything a case study or a blind zero-shot selection task needs: metadata
text, user history text, the full candidate pool (id+text), k=user_avg, and
the ground-truth review (kept in this file for scoring only -- never shown to
a blind LLM-selection prompt, which should instead be built from the
'blind' export of this same file, e.g. via scripts/export_llm100_blind.py)."""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
import random

from peer.utils import ensure_dir, read_jsonl, resolve_cases_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases-dir-template', default='data/cases_{ds}',
                     help='{ds}-templated cases directory, i.e. scripts/build_cases.py --output per platform')
    ap.add_argument('--datasets', nargs='+', default=['baby', 'yelp', 'googlelocal'])
    ap.add_argument('--n-per-dataset', nargs='+', type=int, default=[34, 33, 33])
    ap.add_argument('--split', default='test')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='results/llm100_cases.json')
    args = ap.parse_args()

    assert len(args.datasets) == len(args.n_per_dataset)
    rng = random.Random(args.seed)

    sampled = []
    for ds, n in zip(args.datasets, args.n_per_dataset):
        cases_dir = args.cases_dir_template.format(ds=ds)
        path = resolve_cases_path(cases_dir, args.split)
        all_cases = read_jsonl(path)
        idx = list(range(len(all_cases)))
        rng.shuffle(idx)
        chosen = idx[:n]
        for i in chosen:
            c = all_cases[i]
            sampled.append({
                'case_id': c['case_id'],
                'dataset': ds,
                'item_id': c['item_id'],
                'user_id': c['user_id'],
                'k': int(c.get('user_avg_k', 3)),
                'metadata_text': c.get('metadata_text', ''),
                'user_history_text': c.get('user_history_text', ''),
                'candidates': [{'id': cnd['sentence_id'], 'text': cnd['text']} for cnd in c['candidate_sentences']],
                'ground_truth_sentences': c['ground_truth_sentences'],
            })
        print(f'{ds}: sampled {len(chosen)}/{n} requested (pool size {len(all_cases)})')

    out_path = _ProjectPath(args.output)
    ensure_dir(out_path.parent)
    with open(out_path, 'w') as f:
        json.dump(sampled, f, indent=1)
    print(f'wrote {len(sampled)} cases -> {out_path}')


if __name__ == '__main__':
    main()
